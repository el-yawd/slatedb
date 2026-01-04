//! Contiguous vector file storage with O(1) access
//!
//! Per the LSM-VEC paper: "All vectors are placed in a contiguous on-disk array,
//! sorted by their corresponding ID. This layout allows constant-time retrieval
//! via offset computation."
//!
//! Layout:
//! ```text
//! [Header: 64 bytes][Vector 0][Vector 1][Vector 2]...
//! ```
//!
//! Each vector is `dimension * sizeof(f32)` bytes.
//! Access: `offset = HEADER_SIZE + id * dimension * 4`

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use parking_lot::Mutex;

use crate::error::{LsmVecError, Result};
use crate::types::{Dimension, NodeId};

/// Header size in bytes
const HEADER_SIZE: u64 = 64;

/// Magic number for file identification
const MAGIC: [u8; 8] = *b"LSMVEC01";

/// File format version
const VERSION: u32 = 1;

/// Contiguous vector file storage
///
/// Provides O(1) random access to vectors by ID using offset computation.
pub struct VectorFile {
    file: Mutex<File>,
    dimension: Dimension,
    /// Size of each vector in bytes: dimension * sizeof(f32)
    vector_byte_size: u64,
}

impl VectorFile {
    /// Open or create a vector file
    ///
    /// # Arguments
    /// * `path` - Path to the vector file
    /// * `dimension` - Vector dimension (must match existing file if opening)
    pub fn open(path: &Path, dimension: Dimension) -> Result<Self> {
        let exists = path.exists();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let vector_byte_size = (dimension * std::mem::size_of::<f32>()) as u64;

        let vf = Self {
            file: Mutex::new(file),
            dimension,
            vector_byte_size,
        };

        if exists {
            vf.verify_header()?;
        } else {
            vf.write_header(0)?;
        }

        Ok(vf)
    }

    /// Write file header
    fn write_header(&self, count: u64) -> Result<()> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(0))?;

        // Magic (8 bytes)
        file.write_all(&MAGIC)?;
        // Version (4 bytes)
        file.write_all(&VERSION.to_le_bytes())?;
        // Dimension (4 bytes)
        file.write_all(&(self.dimension as u32).to_le_bytes())?;
        // Count (8 bytes)
        file.write_all(&count.to_le_bytes())?;
        // Reserved (40 bytes)
        file.write_all(&[0u8; 40])?;

        file.sync_all()?;
        Ok(())
    }

    /// Verify file header matches expected configuration
    fn verify_header(&self) -> Result<()> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(0))?;

        // Read magic
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(LsmVecError::InvalidFile("Invalid magic number".into()));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(LsmVecError::InvalidFile(format!(
                "Unsupported version: {}",
                version
            )));
        }

        // Read dimension
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let file_dimension = u32::from_le_bytes(dim_bytes) as usize;
        if file_dimension != self.dimension {
            return Err(LsmVecError::InvalidDimension {
                expected: self.dimension,
                actual: file_dimension,
            });
        }

        Ok(())
    }

    /// Get the number of vectors in the file
    pub fn count(&self) -> Result<u64> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(16))?; // Offset of count field

        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Update the count in the header
    fn update_count(&self, count: u64) -> Result<()> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(16))?;
        file.write_all(&count.to_le_bytes())?;
        Ok(())
    }

    /// Compute file offset for a vector ID
    #[inline]
    fn offset(&self, id: NodeId) -> u64 {
        HEADER_SIZE + id * self.vector_byte_size
    }

    /// Read vector by ID - O(1) access
    ///
    /// Returns `None` if the ID is beyond the current count.
    pub fn get(&self, id: NodeId) -> Result<Option<Vec<f32>>> {
        let offset = self.offset(id);

        let mut file = self.file.lock();

        // Check bounds
        let file_size = file.seek(SeekFrom::End(0))?;
        if offset + self.vector_byte_size > file_size {
            return Ok(None);
        }

        file.seek(SeekFrom::Start(offset))?;

        let mut buf = vec![0u8; self.vector_byte_size as usize];
        file.read_exact(&mut buf)?;

        // Convert bytes to f32 array (little-endian)
        let vector: Vec<f32> = buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(Some(vector))
    }

    /// Write vector at specific ID - O(1) access
    ///
    /// The ID must be <= current count (append or overwrite existing).
    pub fn put(&self, id: NodeId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(LsmVecError::InvalidDimension {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        let offset = self.offset(id);

        // Convert f32 array to bytes (little-endian)
        let buf: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        {
            let mut file = self.file.lock();
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(&buf)?;
        }

        // Update count if this is a new ID
        let current_count = self.count()?;
        if id >= current_count {
            self.update_count(id + 1)?;
        }

        Ok(())
    }

    /// Append a vector and return its assigned ID
    pub fn append(&self, vector: &[f32]) -> Result<NodeId> {
        let id = self.count()?;
        self.put(id, vector)?;
        Ok(id)
    }

    /// Sync file to disk
    pub fn sync(&self) -> Result<()> {
        self.file.lock().sync_all()?;
        Ok(())
    }

    /// Get the dimension of vectors in this file
    pub fn dimension(&self) -> Dimension {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_append() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let vf = VectorFile::open(&path, 4).unwrap();
        assert_eq!(vf.count().unwrap(), 0);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let id1 = vf.append(&v1).unwrap();
        assert_eq!(id1, 0);
        assert_eq!(vf.count().unwrap(), 1);

        let v2 = vec![5.0, 6.0, 7.0, 8.0];
        let id2 = vf.append(&v2).unwrap();
        assert_eq!(id2, 1);
        assert_eq!(vf.count().unwrap(), 2);

        // Read back
        assert_eq!(vf.get(0).unwrap(), Some(v1));
        assert_eq!(vf.get(1).unwrap(), Some(v2));
        assert_eq!(vf.get(2).unwrap(), None);
    }

    #[test]
    fn test_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        // Create and write
        {
            let vf = VectorFile::open(&path, 3).unwrap();
            vf.append(&[1.0, 2.0, 3.0]).unwrap();
            vf.append(&[4.0, 5.0, 6.0]).unwrap();
            vf.sync().unwrap();
        }

        // Reopen and read
        {
            let vf = VectorFile::open(&path, 3).unwrap();
            assert_eq!(vf.count().unwrap(), 2);
            assert_eq!(vf.get(0).unwrap(), Some(vec![1.0, 2.0, 3.0]));
            assert_eq!(vf.get(1).unwrap(), Some(vec![4.0, 5.0, 6.0]));
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        // Create with dimension 4
        {
            let vf = VectorFile::open(&path, 4).unwrap();
            vf.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
            vf.sync().unwrap();
        }

        // Try to open with dimension 3
        let result = VectorFile::open(&path, 3);
        assert!(matches!(result, Err(LsmVecError::InvalidDimension { .. })));
    }

    #[test]
    fn test_overwrite() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let vf = VectorFile::open(&path, 2).unwrap();
        vf.append(&[1.0, 2.0]).unwrap();
        vf.append(&[3.0, 4.0]).unwrap();

        // Overwrite ID 0
        vf.put(0, &[10.0, 20.0]).unwrap();
        assert_eq!(vf.get(0).unwrap(), Some(vec![10.0, 20.0]));
        assert_eq!(vf.get(1).unwrap(), Some(vec![3.0, 4.0]));
        assert_eq!(vf.count().unwrap(), 2);
    }
}
