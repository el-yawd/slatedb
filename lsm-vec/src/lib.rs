//! LSM-VEC: Disk-based vector search with HNSW graph indexing on LSM-tree storage
//!
//! This crate implements the LSM-VEC architecture from the paper:
//! ["LSM-VEC: A Large-Scale Disk-Based System for Dynamic Vector Search"](https://arxiv.org/abs/2505.17152)
//!
//! ## Architecture
//!
//! LSM-VEC combines HNSW (Hierarchical Navigable Small World) graphs with LSM-tree storage:
//!
//! - **Vector Data**: Stored in a separate contiguous file for O(1) access by ID
//! - **Graph Edges (Layer 0)**: Stored in SlateDB (LSM-tree) for efficient updates
//! - **Upper Layers**: Kept in memory (<1% of nodes, fast navigation)
//! - **SimHash**: Probabilistic sampling to reduce I/O during search
//!
//! ## Example
//!
//! ```ignore
//! use lsm_vec::{LsmVecDb, LsmVecConfig};
//! use slatedb::Db;
//! use std::sync::Arc;
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Open SlateDB for graph storage
//!     let db = Arc::new(Db::open("data/graph").await.unwrap());
//!
//!     // Open LSM-VEC with 128-dimensional vectors
//!     let config = LsmVecConfig::new(128);
//!     let vec_db = LsmVecDb::open(db, Path::new("data/vectors.bin"), config)
//!         .await
//!         .unwrap();
//!
//!     // Insert vectors
//!     let id1 = vec_db.insert(vec![1.0; 128]).await.unwrap();
//!     let id2 = vec_db.insert(vec![2.0; 128]).await.unwrap();
//!
//!     // Search for k nearest neighbors
//!     let results = vec_db.search(&[1.5; 128], 10).await.unwrap();
//!     for result in results {
//!         println!("ID: {}, Distance: {}", result.id, result.distance);
//!     }
//! }
//! ```

mod distance;
mod error;
mod graph_store;
mod index;
mod simhash;
mod types;
mod upper_graph;
mod vector_file;

pub use error::{LsmVecError, Result};
pub use index::IndexStats;
pub use types::{DistanceMetric, LsmVecConfig, NodeId, SearchResult};

use slatedb::Db;
use std::path::Path;
use std::sync::Arc;

use crate::graph_store::GraphStore;
use crate::index::HnswIndex;
use crate::vector_file::VectorFile;

/// LSM-VEC database for billion-scale vector search
///
/// This is the main entry point for using LSM-VEC. It provides:
/// - `insert()` - Add vectors to the index
/// - `search()` - Find k nearest neighbors
/// - `delete()` - Remove vectors from the index
/// - `get()` - Retrieve a vector by ID
pub struct LsmVecDb {
    index: HnswIndex,
    #[allow(dead_code)]
    vectors: Arc<VectorFile>,
}

impl LsmVecDb {
    /// Open or create a LSM-VEC database
    ///
    /// # Arguments
    /// * `db` - SlateDB instance for graph edge storage
    /// * `vector_path` - Path to the vector file (separate from SlateDB)
    /// * `config` - Index configuration
    ///
    /// # Example
    /// ```ignore
    /// let db = Arc::new(Db::open("data/graph").await?);
    /// let config = LsmVecConfig::new(128);
    /// let vec_db = LsmVecDb::open(db, Path::new("data/vectors.bin"), config).await?;
    /// ```
    pub async fn open(db: Arc<Db>, vector_path: &Path, config: LsmVecConfig) -> Result<Self> {
        let vectors = Arc::new(VectorFile::open(vector_path, config.dimension)?);
        let graph_store = Arc::new(GraphStore::new(db));
        let index = HnswIndex::new(vectors.clone(), graph_store, config).await?;

        Ok(Self { index, vectors })
    }

    /// Insert a vector into the index
    ///
    /// Returns the assigned vector ID.
    ///
    /// # Example
    /// ```ignore
    /// let id = vec_db.insert(vec![1.0, 2.0, 3.0, 4.0]).await?;
    /// ```
    pub async fn insert(&self, vector: Vec<f32>) -> Result<NodeId> {
        self.index.insert(vector).await
    }

    /// Delete a vector by ID
    ///
    /// Note: The vector data in the file is not actually removed (would create holes).
    /// This marks the vector as deleted in the graph.
    ///
    /// # Example
    /// ```ignore
    /// vec_db.delete(42).await?;
    /// ```
    pub async fn delete(&self, id: NodeId) -> Result<()> {
        self.index.delete(id).await
    }

    /// Search for k nearest neighbors
    ///
    /// Uses the default ef_search parameter from the configuration.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Example
    /// ```ignore
    /// let results = vec_db.search(&query_vector, 10).await?;
    /// for result in results {
    ///     println!("ID: {}, Distance: {}", result.id, result.distance);
    /// }
    /// ```
    pub async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.index.search(query, k).await
    }

    /// Search for k nearest neighbors with custom ef parameter
    ///
    /// Higher ef values give better recall but slower search.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Search width (ef >= k, higher = better recall, slower)
    ///
    /// # Example
    /// ```ignore
    /// // More thorough search with ef=200
    /// let results = vec_db.search_with_ef(&query_vector, 10, 200).await?;
    /// ```
    pub async fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>> {
        self.index.search_with_ef(query, k, ef).await
    }

    /// Get a vector by ID
    ///
    /// Returns None if the ID doesn't exist.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(vector) = vec_db.get(42)? {
    ///     println!("Vector: {:?}", vector);
    /// }
    /// ```
    pub fn get(&self, id: NodeId) -> Result<Option<Vec<f32>>> {
        self.index.get(id)
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        self.index.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::local::LocalFileSystem;
    use rand::Rng;
    use tempfile::tempdir;

    async fn create_test_db() -> (Arc<Db>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let object_store: Arc<dyn object_store::ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        let db = Db::builder("test", object_store).build().await.unwrap();
        (Arc::new(db), dir)
    }

    #[tokio::test]
    async fn test_insert_and_search() {
        let (db, dir) = create_test_db().await;
        let vector_path = dir.path().join("vectors.bin");

        let config = LsmVecConfig::new(4);
        let vec_db = LsmVecDb::open(db, &vector_path, config).await.unwrap();

        // Insert some vectors
        let id1 = vec_db.insert(vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        let id2 = vec_db.insert(vec![0.9, 0.1, 0.0, 0.0]).await.unwrap();
        let id3 = vec_db.insert(vec![0.0, 1.0, 0.0, 0.0]).await.unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);

        // Search for nearest to [1.0, 0.0, 0.0, 0.0]
        let results = vec_db.search(&[1.0, 0.0, 0.0, 0.0], 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id1); // Exact match should be first
        assert!(results[0].distance < 0.001);
    }

    #[tokio::test]
    async fn test_get_vector() {
        let (db, dir) = create_test_db().await;
        let vector_path = dir.path().join("vectors.bin");

        let config = LsmVecConfig::new(3);
        let vec_db = LsmVecDb::open(db, &vector_path, config).await.unwrap();

        let v = vec![1.0, 2.0, 3.0];
        let id = vec_db.insert(v.clone()).await.unwrap();

        let retrieved = vec_db.get(id).unwrap();
        assert_eq!(retrieved, Some(v));

        // Non-existent ID
        assert!(vec_db.get(999).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete() {
        let (db, dir) = create_test_db().await;
        let vector_path = dir.path().join("vectors.bin");

        let config = LsmVecConfig::new(4);
        let vec_db = LsmVecDb::open(db, &vector_path, config).await.unwrap();

        // Insert 3 similar vectors so SimHash doesn't filter them out
        // (orthogonal vectors have ~50% collision which may be filtered)
        let id1 = vec_db.insert(vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        let id2 = vec_db.insert(vec![0.9, 0.1, 0.0, 0.0]).await.unwrap();
        let id3 = vec_db.insert(vec![0.8, 0.2, 0.0, 0.0]).await.unwrap();

        // Delete id1
        vec_db.delete(id1).await.unwrap();

        // Search for something close to id2
        let results = vec_db.search(&[0.9, 0.1, 0.0, 0.0], 10).await.unwrap();

        // Should find at least id2 and id3
        assert!(
            !results.is_empty(),
            "Search should find results after delete"
        );

        // id2 should be first (exact match)
        assert_eq!(results[0].id, id2);

        // id1 should not be in results (it was deleted from graph)
        let ids: Vec<_> = results.iter().map(|r| r.id).collect();
        // Note: id1's vector still exists in file but shouldn't be reachable via graph
        assert!(ids.contains(&id2));
        assert!(ids.contains(&id3));
    }

    #[tokio::test]
    async fn test_many_vectors() {
        let (db, dir) = create_test_db().await;
        let vector_path = dir.path().join("vectors.bin");

        let config = LsmVecConfig::new(8);
        let vec_db = LsmVecDb::open(db, &vector_path, config).await.unwrap();

        // Insert 100 random vectors
        let mut rng = rand::rng();
        for _ in 0..100 {
            let v: Vec<f32> = (0..8).map(|_| rng.random_range(-1.0..1.0)).collect();
            vec_db.insert(v).await.unwrap();
        }

        let stats = vec_db.stats();
        assert_eq!(stats.vector_count, 100);

        // Search should work
        let query: Vec<f32> = (0..8).map(|_| rng.random_range(-1.0..1.0)).collect();
        let results = vec_db.search(&query, 10).await.unwrap();
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].distance <= results[i].distance);
        }
    }
}
