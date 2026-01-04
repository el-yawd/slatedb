# LSM-VEC Implementation Plan

Implementation of LSM-VEC based on the paper: [LSM-VEC: A Large-Scale Disk-Based System for Dynamic Vector Search](https://arxiv.org/abs/2505.17152)

---

## Paper Key Points

From the paper:

> "LSM-VEC stores vector data **separately** from the graph index. All vectors are placed in a **contiguous on-disk array, sorted by their corresponding ID**. This layout allows **constant-time retrieval via offset computation**, avoiding redundant data storage."

This means:
1. **Graph edges** → Stored in LSM-tree (SlateDB)
2. **Vector data** → Separate contiguous file with O(1) access

Since all vectors have the same dimension `d`, accessing vector with ID `i`:
```
offset = i * d * sizeof(f32)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LSM-VEC API                              │
│  insert() | delete() | search() | get()                        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    HNSW Index Manager                           │
│  - Entry point tracking                                         │
│  - Level assignment (exponential distribution)                  │
│  - Search coordination across layers                            │
│  - SimHash-based probabilistic sampling                         │
└─────────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
┌──────────────────────┐          ┌──────────────────────────────┐
│  Upper Layers (1+)   │          │      Layer 0 (Disk)          │
│  In-Memory Graph     │          │   Graph edges via SlateDB    │
│  (<1% of nodes)      │          │   Key: node_id → neighbors   │
└──────────────────────┘          └──────────────────────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────────────┐
                               │         SlateDB (LSM-Tree)       │
                               │  - Graph edges as KV pairs       │
                               │  - Node metadata                 │
                               │  - Index metadata                │
                               └──────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 Vector File (Separate from LSM)                 │
│  Contiguous array: [vec_0][vec_1][vec_2]...[vec_n]              │
│  Access: offset = id * dimension * 4 bytes                      │
│  O(1) random read by ID                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Types

### File: `src/types.rs`

```rust
/// Unique identifier for a vector/node
pub type NodeId = u64;

/// Layer identifier in HNSW graph
pub type LayerId = u8;

/// Vector dimensionality
pub type Dimension = usize;

/// Distance metric
#[derive(Clone, Copy, Debug, Default)]
pub enum DistanceMetric {
    #[default]
    L2,
    Cosine,
}

/// Configuration for LSM-VEC index
#[derive(Clone, Debug)]
pub struct LsmVecConfig {
    /// Vector dimensionality (fixed for all vectors)
    pub dimension: Dimension,

    /// Distance metric
    pub metric: DistanceMetric,

    /// Max neighbors at layer 0 (M_max, default: 32)
    pub m_max: usize,

    /// Max neighbors at upper layers (M, default: 16)
    pub m: usize,

    /// Level multiplier: 1/ln(M)
    pub m_level: f64,

    /// Construction search width (ef_construction, default: 200)
    pub ef_construction: usize,

    /// Default query search width (ef_search, default: 100)
    pub ef_search: usize,

    /// SimHash bits (m, default: 64)
    pub simhash_bits: usize,

    /// SimHash threshold epsilon (default: 0.1)
    pub simhash_epsilon: f64,
}

impl Default for LsmVecConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            dimension: 128,
            metric: DistanceMetric::L2,
            m_max: 32,
            m,
            m_level: 1.0 / (m as f64).ln(),
            ef_construction: 200,
            ef_search: 100,
            simhash_bits: 64,
            simhash_epsilon: 0.1,
        }
    }
}

/// Search result
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: NodeId,
    pub distance: f32,
}

/// Node metadata
#[derive(Clone, Debug)]
pub struct NodeMetadata {
    pub id: NodeId,
    pub max_layer: LayerId,
}
```

---

## Phase 2: Vector File Storage

Separate contiguous file for O(1) vector access.

### File: `src/vector_file.rs`

```rust
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use crate::types::{NodeId, Dimension};
use crate::error::LsmVecError;

/// Vector file header (stored at offset 0)
#[repr(C)]
struct VectorFileHeader {
    magic: [u8; 8],      // "LSMVEC\0\0"
    version: u32,        // File format version
    dimension: u32,      // Vector dimension
    count: u64,          // Number of vectors (also next ID to assign)
    _reserved: [u8; 40], // Future use, pad to 64 bytes
}

const HEADER_SIZE: u64 = 64;
const MAGIC: [u8; 8] = *b"LSMVEC\0\0";
const VERSION: u32 = 1;

/// Contiguous vector file storage
///
/// Layout:
/// [Header: 64 bytes][Vector 0][Vector 1][Vector 2]...
///
/// Each vector is `dimension * 4` bytes (f32 array)
/// Access: offset = HEADER_SIZE + id * dimension * 4
pub struct VectorFile {
    file: Mutex<File>,
    dimension: Dimension,
    vector_size: u64,  // dimension * 4 bytes
}

impl VectorFile {
    /// Open or create vector file
    pub fn open(path: &Path, dimension: Dimension) -> Result<Self, LsmVecError> {
        let exists = path.exists();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let vector_size = (dimension * std::mem::size_of::<f32>()) as u64;

        let mut vf = Self {
            file: Mutex::new(file),
            dimension,
            vector_size,
        };

        if exists {
            vf.verify_header()?;
        } else {
            vf.write_header(0)?;
        }

        Ok(vf)
    }

    fn write_header(&self, count: u64) -> Result<(), LsmVecError> {
        let header = VectorFileHeader {
            magic: MAGIC,
            version: VERSION,
            dimension: self.dimension as u32,
            count,
            _reserved: [0; 40],
        };

        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(0))?;

        // Write header as raw bytes
        let header_bytes: [u8; 64] = unsafe { std::mem::transmute(header) };
        file.write_all(&header_bytes)?;
        file.sync_all()?;

        Ok(())
    }

    fn verify_header(&self) -> Result<(), LsmVecError> {
        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(0))?;

        let mut header_bytes = [0u8; 64];
        file.read_exact(&mut header_bytes)?;

        let header: VectorFileHeader = unsafe { std::mem::transmute(header_bytes) };

        if header.magic != MAGIC {
            return Err(LsmVecError::InvalidFile("Invalid magic number".into()));
        }
        if header.version != VERSION {
            return Err(LsmVecError::InvalidFile("Unsupported version".into()));
        }
        if header.dimension as usize != self.dimension {
            return Err(LsmVecError::InvalidDimension {
                expected: self.dimension,
                actual: header.dimension as usize,
            });
        }

        Ok(())
    }

    /// Read vector count from header
    pub fn count(&self) -> Result<u64, LsmVecError> {
        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(16))?; // offset of count field

        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Compute file offset for a vector ID
    fn offset(&self, id: NodeId) -> u64 {
        HEADER_SIZE + id * self.vector_size
    }

    /// Read vector by ID - O(1) access
    pub fn get(&self, id: NodeId) -> Result<Option<Vec<f32>>, LsmVecError> {
        let offset = self.offset(id);

        let mut file = self.file.lock().unwrap();

        // Check if ID is within bounds
        let file_size = file.seek(SeekFrom::End(0))?;
        if offset + self.vector_size > file_size {
            return Ok(None);
        }

        file.seek(SeekFrom::Start(offset))?;

        let mut buf = vec![0u8; self.vector_size as usize];
        file.read_exact(&mut buf)?;

        // Convert bytes to f32 array
        let vector: Vec<f32> = buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(Some(vector))
    }

    /// Write vector at ID - O(1) access
    /// ID must be <= current count (append or overwrite)
    pub fn put(&self, id: NodeId, vector: &[f32]) -> Result<(), LsmVecError> {
        if vector.len() != self.dimension {
            return Err(LsmVecError::InvalidDimension {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        let offset = self.offset(id);

        // Convert f32 array to bytes
        let buf: Vec<u8> = vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(&buf)?;

        // Update count if this is a new ID
        let current_count = {
            file.seek(SeekFrom::Start(16))?;
            let mut count_buf = [0u8; 8];
            file.read_exact(&mut count_buf)?;
            u64::from_le_bytes(count_buf)
        };

        if id >= current_count {
            let new_count = id + 1;
            file.seek(SeekFrom::Start(16))?;
            file.write_all(&new_count.to_le_bytes())?;
        }

        Ok(())
    }

    /// Append vector and return assigned ID
    pub fn append(&self, vector: &[f32]) -> Result<NodeId, LsmVecError> {
        let id = self.count()?;
        self.put(id, vector)?;
        Ok(id)
    }

    /// Sync to disk
    pub fn sync(&self) -> Result<(), LsmVecError> {
        self.file.lock().unwrap().sync_all()?;
        Ok(())
    }

    /// Get dimension
    pub fn dimension(&self) -> Dimension {
        self.dimension
    }
}
```

---

## Phase 3: Distance Functions

### File: `src/distance.rs`

```rust
use crate::types::DistanceMetric;

pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match metric {
        DistanceMetric::L2 => l2_distance(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}
```

---

## Phase 4: SimHash Sampler

Per paper: probabilistic sampling to reduce I/O during search.

### File: `src/simhash.rs`

```rust
use bitvec::prelude::*;
use parking_lot::RwLock;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::collections::HashMap;

use crate::types::{Dimension, NodeId};

/// SimHash for probabilistic neighbor filtering
/// Hash(x) = [sgn(x·a_1), sgn(x·a_2), ..., sgn(x·a_m)]
pub struct SimHashSampler {
    /// Random projection vectors from N(0, I_d)
    projections: Vec<Vec<f32>>,
    num_bits: usize,
    dimension: Dimension,

    /// Cached hashes for all nodes
    hashes: RwLock<HashMap<NodeId, BitVec>>,
}

impl SimHashSampler {
    pub fn new(dimension: Dimension, num_bits: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let projections: Vec<Vec<f32>> = (0..num_bits)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.sample(StandardNormal))
                    .collect()
            })
            .collect();

        Self {
            projections,
            num_bits,
            dimension,
            hashes: RwLock::new(HashMap::new()),
        }
    }

    /// Compute hash for vector
    pub fn compute_hash(&self, vector: &[f32]) -> BitVec {
        debug_assert_eq!(vector.len(), self.dimension);

        let mut hash = BitVec::with_capacity(self.num_bits);
        for proj in &self.projections {
            let dot: f32 = vector.iter().zip(proj.iter()).map(|(x, p)| x * p).sum();
            hash.push(dot >= 0.0);
        }
        hash
    }

    pub fn store_hash(&self, id: NodeId, hash: BitVec) {
        self.hashes.write().insert(id, hash);
    }

    pub fn get_hash(&self, id: NodeId) -> Option<BitVec> {
        self.hashes.read().get(&id).cloned()
    }

    pub fn remove_hash(&self, id: NodeId) {
        self.hashes.write().remove(&id);
    }

    /// Count collisions (matching bits)
    pub fn collision_count(h1: &BitVec, h2: &BitVec) -> usize {
        h1.iter().zip(h2.iter()).filter(|(a, b)| *a == *b).count()
    }

    /// Compute threshold for epsilon
    pub fn threshold(&self, epsilon: f64) -> usize {
        ((self.num_bits as f64) * (1.0 - epsilon * 0.5)) as usize
    }

    /// Filter candidates by SimHash threshold
    pub fn filter_candidates(
        &self,
        query_hash: &BitVec,
        candidates: &[NodeId],
        threshold: usize,
    ) -> Vec<NodeId> {
        let hashes = self.hashes.read();
        candidates
            .iter()
            .filter(|&id| {
                hashes.get(id)
                    .map(|h| Self::collision_count(query_hash, h) >= threshold)
                    .unwrap_or(true)
            })
            .copied()
            .collect()
    }
}
```

---

## Phase 5: Graph Storage (SlateDB)

Only graph edges and metadata in SlateDB, NOT vectors.

### File: `src/graph_store.rs`

```rust
use bytes::{Bytes, BytesMut, BufMut, Buf};
use slatedb::Db;
use std::sync::Arc;

use crate::types::{NodeId, NodeMetadata, LayerId};
use crate::error::LsmVecError;

mod prefix {
    pub const EDGE: u8 = b'e';      // Graph edges (layer 0)
    pub const META: u8 = b'm';      // Node metadata
    pub const INDEX: u8 = b'i';     // Index metadata
}

/// Graph storage in SlateDB (edges only, no vectors)
pub struct GraphStore {
    db: Arc<Db>,
}

impl GraphStore {
    pub fn new(db: Arc<Db>) -> Self {
        Self { db }
    }

    // Key encoding
    fn edge_key(id: NodeId) -> Bytes {
        let mut key = BytesMut::with_capacity(9);
        key.put_u8(prefix::EDGE);
        key.put_u64(id);
        key.freeze()
    }

    fn meta_key(id: NodeId) -> Bytes {
        let mut key = BytesMut::with_capacity(9);
        key.put_u8(prefix::META);
        key.put_u64(id);
        key.freeze()
    }

    fn index_key(name: &str) -> Bytes {
        let mut key = BytesMut::with_capacity(1 + name.len());
        key.put_u8(prefix::INDEX);
        key.put_slice(name.as_bytes());
        key.freeze()
    }

    // ========== Edge Operations (Layer 0) ==========

    pub async fn put_neighbors(&self, id: NodeId, neighbors: &[NodeId]) -> Result<(), LsmVecError> {
        let key = Self::edge_key(id);
        let mut buf = BytesMut::with_capacity(neighbors.len() * 8);
        for &n in neighbors {
            buf.put_u64(n);
        }
        self.db.put(&key, &buf.freeze()).await?;
        Ok(())
    }

    pub async fn get_neighbors(&self, id: NodeId) -> Result<Option<Vec<NodeId>>, LsmVecError> {
        let key = Self::edge_key(id);
        match self.db.get(&key).await? {
            Some(bytes) => {
                let mut buf = &bytes[..];
                let mut neighbors = Vec::with_capacity(bytes.len() / 8);
                while buf.remaining() >= 8 {
                    neighbors.push(buf.get_u64());
                }
                Ok(Some(neighbors))
            }
            None => Ok(None),
        }
    }

    pub async fn delete_neighbors(&self, id: NodeId) -> Result<(), LsmVecError> {
        let key = Self::edge_key(id);
        self.db.delete(&key).await?;
        Ok(())
    }

    // ========== Node Metadata ==========

    pub async fn put_metadata(&self, meta: &NodeMetadata) -> Result<(), LsmVecError> {
        let key = Self::meta_key(meta.id);
        let mut buf = BytesMut::with_capacity(9);
        buf.put_u64(meta.id);
        buf.put_u8(meta.max_layer);
        self.db.put(&key, &buf.freeze()).await?;
        Ok(())
    }

    pub async fn get_metadata(&self, id: NodeId) -> Result<Option<NodeMetadata>, LsmVecError> {
        let key = Self::meta_key(id);
        match self.db.get(&key).await? {
            Some(bytes) => {
                let mut buf = &bytes[..];
                Ok(Some(NodeMetadata {
                    id: buf.get_u64(),
                    max_layer: buf.get_u8(),
                }))
            }
            None => Ok(None),
        }
    }

    pub async fn delete_metadata(&self, id: NodeId) -> Result<(), LsmVecError> {
        let key = Self::meta_key(id);
        self.db.delete(&key).await?;
        Ok(())
    }

    // ========== Index Metadata ==========

    pub async fn put_entry_point(&self, id: NodeId, layer: LayerId) -> Result<(), LsmVecError> {
        let key = Self::index_key("entry_point");
        let mut buf = BytesMut::with_capacity(9);
        buf.put_u64(id);
        buf.put_u8(layer);
        self.db.put(&key, &buf.freeze()).await?;
        Ok(())
    }

    pub async fn get_entry_point(&self) -> Result<Option<(NodeId, LayerId)>, LsmVecError> {
        let key = Self::index_key("entry_point");
        match self.db.get(&key).await? {
            Some(bytes) => {
                let mut buf = &bytes[..];
                Ok(Some((buf.get_u64(), buf.get_u8())))
            }
            None => Ok(None),
        }
    }
}
```

---

## Phase 6: In-Memory Upper Layers

### File: `src/upper_graph.rs`

```rust
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::types::{NodeId, LayerId};

pub const MAX_LAYERS: usize = 16;

/// In-memory graph for upper layers (layer >= 1)
/// Per paper: <1% of nodes exist in upper layers
pub struct UpperGraph {
    layers: Vec<RwLock<HashMap<NodeId, Vec<NodeId>>>>,
    entry_point: AtomicU64,
    entry_point_layer: AtomicU8,
    has_entry_point: AtomicU8,
}

impl UpperGraph {
    pub fn new() -> Self {
        Self {
            layers: (0..MAX_LAYERS).map(|_| RwLock::new(HashMap::new())).collect(),
            entry_point: AtomicU64::new(0),
            entry_point_layer: AtomicU8::new(0),
            has_entry_point: AtomicU8::new(0),
        }
    }

    pub fn get_entry_point(&self) -> Option<(NodeId, LayerId)> {
        if self.has_entry_point.load(Ordering::Acquire) == 0 {
            return None;
        }
        Some((
            self.entry_point.load(Ordering::Acquire),
            self.entry_point_layer.load(Ordering::Acquire),
        ))
    }

    pub fn set_entry_point(&self, id: NodeId, layer: LayerId) {
        self.entry_point.store(id, Ordering::Release);
        self.entry_point_layer.store(layer, Ordering::Release);
        self.has_entry_point.store(1, Ordering::Release);
    }

    pub fn get_neighbors(&self, id: NodeId, layer: LayerId) -> Option<Vec<NodeId>> {
        if layer == 0 || layer as usize >= MAX_LAYERS {
            return None;
        }
        self.layers[layer as usize].read().get(&id).cloned()
    }

    pub fn set_neighbors(&self, id: NodeId, layer: LayerId, neighbors: Vec<NodeId>) {
        if layer == 0 || layer as usize >= MAX_LAYERS {
            return;
        }
        self.layers[layer as usize].write().insert(id, neighbors);
    }

    pub fn remove_node(&self, id: NodeId) {
        for layer in &self.layers {
            layer.write().remove(&id);
        }
    }
}

impl Default for UpperGraph {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## Phase 7: HNSW Index

### File: `src/index.rs`

```rust
use ordered_float::OrderedFloat;
use rand::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use crate::distance::compute_distance;
use crate::error::LsmVecError;
use crate::graph_store::GraphStore;
use crate::simhash::SimHashSampler;
use crate::types::*;
use crate::upper_graph::UpperGraph;
use crate::vector_file::VectorFile;

#[derive(Clone, PartialEq, Eq)]
struct Candidate {
    id: NodeId,
    distance: OrderedFloat<f32>,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// LSM-VEC HNSW Index
pub struct HnswIndex {
    /// Vector file (separate from LSM, O(1) access)
    vectors: Arc<VectorFile>,

    /// Graph store (SlateDB, edges only)
    graph_store: Arc<GraphStore>,

    /// In-memory upper layers
    upper_graph: UpperGraph,

    /// SimHash sampler
    sampler: SimHashSampler,

    /// Config
    config: LsmVecConfig,
}

impl HnswIndex {
    pub async fn new(
        vectors: Arc<VectorFile>,
        graph_store: Arc<GraphStore>,
        config: LsmVecConfig,
    ) -> Result<Self, LsmVecError> {
        let sampler = SimHashSampler::new(config.dimension, config.simhash_bits, 42);

        let mut index = Self {
            vectors,
            graph_store,
            upper_graph: UpperGraph::new(),
            sampler,
            config,
        };

        // Load entry point
        if let Some((id, layer)) = index.graph_store.get_entry_point().await? {
            index.upper_graph.set_entry_point(id, layer);
        }

        Ok(index)
    }

    /// Random level using exponential distribution
    fn random_level(&self) -> LayerId {
        let mut rng = thread_rng();
        let f: f64 = rng.gen();
        let level = (-f.ln() * self.config.m_level).floor() as LayerId;
        level.min(15)
    }

    /// Get vector by ID (O(1) from file)
    fn get_vector(&self, id: NodeId) -> Result<Option<Vec<f32>>, LsmVecError> {
        self.vectors.get(id)
    }

    /// Get neighbors at layer
    async fn get_neighbors(&self, id: NodeId, layer: LayerId) -> Result<Vec<NodeId>, LsmVecError> {
        if layer == 0 {
            Ok(self.graph_store.get_neighbors(id).await?.unwrap_or_default())
        } else {
            Ok(self.upper_graph.get_neighbors(id, layer).unwrap_or_default())
        }
    }

    /// Set neighbors at layer
    async fn set_neighbors(&self, id: NodeId, layer: LayerId, neighbors: Vec<NodeId>) -> Result<(), LsmVecError> {
        if layer == 0 {
            self.graph_store.put_neighbors(id, &neighbors).await?;
        } else {
            self.upper_graph.set_neighbors(id, layer, neighbors);
        }
        Ok(())
    }

    /// Search single layer
    async fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[NodeId],
        layer: LayerId,
        ef: usize,
        use_simhash: bool,
    ) -> Result<Vec<Candidate>, LsmVecError> {
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut visited: HashSet<NodeId> = HashSet::new();

        let query_hash = if use_simhash {
            Some(self.sampler.compute_hash(query))
        } else {
            None
        };
        let threshold = self.sampler.threshold(self.config.simhash_epsilon);

        // Initialize
        for &ep in entry_points {
            if visited.insert(ep) {
                if let Some(vec) = self.get_vector(ep)? {
                    let dist = compute_distance(query, &vec, self.config.metric);
                    let c = Candidate { id: ep, distance: OrderedFloat(dist) };
                    candidates.push(Reverse(c.clone()));
                    results.push(c);
                }
            }
        }

        while let Some(Reverse(current)) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }

            let mut neighbors = self.get_neighbors(current.id, layer).await?;

            // SimHash filtering for layer 0
            if layer == 0 {
                if let Some(ref qh) = query_hash {
                    neighbors = self.sampler.filter_candidates(qh, &neighbors, threshold);
                }
            }

            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    if let Some(vec) = self.get_vector(neighbor)? {
                        let dist = compute_distance(query, &vec, self.config.metric);
                        let should_add = results.len() < ef
                            || dist < results.peek().map(|c| c.distance.0).unwrap_or(f32::MAX);

                        if should_add {
                            let c = Candidate { id: neighbor, distance: OrderedFloat(dist) };
                            candidates.push(Reverse(c.clone()));
                            results.push(c);
                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<_> = results.into_iter().collect();
        result_vec.sort_by_key(|c| c.distance);
        Ok(result_vec)
    }

    /// k-NN search
    pub async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, LsmVecError> {
        self.search_with_ef(query, k, self.config.ef_search).await
    }

    pub async fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>, LsmVecError> {
        let (entry_point, max_layer) = match self.upper_graph.get_entry_point() {
            Some(ep) => ep,
            None => return Ok(vec![]),
        };

        let mut current_best = vec![entry_point];

        // Upper layers (greedy, ef=1)
        for layer in (1..=max_layer).rev() {
            let results = self.search_layer(query, &current_best, layer, 1, false).await?;
            if !results.is_empty() {
                current_best = results.into_iter().map(|c| c.id).collect();
            }
        }

        // Layer 0 (full ef, with SimHash)
        let results = self.search_layer(query, &current_best, 0, ef.max(k), true).await?;

        Ok(results
            .into_iter()
            .take(k)
            .map(|c| SearchResult { id: c.id, distance: c.distance.0 })
            .collect())
    }

    /// Insert vector
    pub async fn insert(&self, vector: Vec<f32>) -> Result<NodeId, LsmVecError> {
        // 1. Append to vector file (get ID)
        let id = self.vectors.append(&vector)?;

        // 2. Compute & store SimHash
        let hash = self.sampler.compute_hash(&vector);
        self.sampler.store_hash(id, hash);

        // 3. Random level
        let node_layer = self.random_level();

        // 4. Store metadata
        self.graph_store.put_metadata(&NodeMetadata { id, max_layer: node_layer }).await?;

        // 5. Insert into graph
        let entry_point = self.upper_graph.get_entry_point();

        if let Some((ep, max_layer)) = entry_point {
            let mut current_best = vec![ep];

            // Navigate to node_layer
            for layer in (node_layer.saturating_add(1)..=max_layer).rev() {
                let results = self.search_layer(&vector, &current_best, layer, 1, false).await?;
                if !results.is_empty() {
                    current_best = results.into_iter().map(|c| c.id).collect();
                }
            }

            // Insert at each layer
            let start_layer = node_layer.min(max_layer);
            for layer in (0..=start_layer).rev() {
                let results = self.search_layer(
                    &vector,
                    &current_best,
                    layer,
                    self.config.ef_construction,
                    layer == 0,
                ).await?;

                let max_neighbors = if layer == 0 { self.config.m_max } else { self.config.m };
                let neighbors = self.select_neighbors(&vector, results, max_neighbors)?;

                self.set_neighbors(id, layer, neighbors.clone()).await?;

                // Update neighbors' edges
                for &neighbor in &neighbors {
                    let mut neighbor_edges = self.get_neighbors(neighbor, layer).await?;
                    neighbor_edges.push(id);

                    if neighbor_edges.len() > max_neighbors {
                        if let Some(neighbor_vec) = self.get_vector(neighbor)? {
                            neighbor_edges = self.prune_neighbors(&neighbor_vec, neighbor_edges, max_neighbors)?;
                        }
                    }
                    self.set_neighbors(neighbor, layer, neighbor_edges).await?;
                }

                current_best = neighbors;
            }

            if node_layer > max_layer {
                self.upper_graph.set_entry_point(id, node_layer);
                self.graph_store.put_entry_point(id, node_layer).await?;
            }
        } else {
            self.upper_graph.set_entry_point(id, node_layer);
            self.graph_store.put_entry_point(id, node_layer).await?;
        }

        Ok(id)
    }

    /// Select neighbors heuristic
    fn select_neighbors(&self, query: &[f32], candidates: Vec<Candidate>, m: usize) -> Result<Vec<NodeId>, LsmVecError> {
        let mut result = Vec::with_capacity(m);

        for candidate in candidates {
            if result.len() >= m {
                break;
            }

            let candidate_vec = match self.get_vector(candidate.id)? {
                Some(v) => v,
                None => continue,
            };

            let mut is_good = true;
            for &selected_id in &result {
                if let Some(selected_vec) = self.get_vector(selected_id)? {
                    let dist_to_selected = compute_distance(&candidate_vec, &selected_vec, self.config.metric);
                    if dist_to_selected < candidate.distance.0 {
                        is_good = false;
                        break;
                    }
                }
            }

            if is_good {
                result.push(candidate.id);
            }
        }

        Ok(result)
    }

    /// Prune neighbors
    fn prune_neighbors(&self, node_vec: &[f32], neighbors: Vec<NodeId>, max: usize) -> Result<Vec<NodeId>, LsmVecError> {
        let mut scored: Vec<Candidate> = neighbors
            .into_iter()
            .filter_map(|id| {
                self.get_vector(id).ok().flatten().map(|vec| {
                    let dist = compute_distance(node_vec, &vec, self.config.metric);
                    Candidate { id, distance: OrderedFloat(dist) }
                })
            })
            .collect();

        scored.sort_by_key(|c| c.distance);
        Ok(scored.into_iter().take(max).map(|c| c.id).collect())
    }

    /// Delete vector
    pub async fn delete(&self, id: NodeId) -> Result<(), LsmVecError> {
        let meta = match self.graph_store.get_metadata(id).await? {
            Some(m) => m,
            None => return Ok(()),
        };

        // Remove edges and reconnect neighbors at each layer
        for layer in 0..=meta.max_layer {
            let neighbors = self.get_neighbors(id, layer).await?;

            for &neighbor in &neighbors {
                let mut neighbor_edges = self.get_neighbors(neighbor, layer).await?;
                neighbor_edges.retain(|&n| n != id);

                // Collect reconnection candidates
                let mut candidates: HashSet<NodeId> = HashSet::new();
                candidates.extend(&neighbors);
                candidates.extend(&neighbor_edges);
                candidates.remove(&neighbor);
                candidates.remove(&id);

                if let Some(neighbor_vec) = self.get_vector(neighbor)? {
                    let max_neighbors = if layer == 0 { self.config.m_max } else { self.config.m };

                    let mut scored: Vec<Candidate> = candidates
                        .into_iter()
                        .filter_map(|c| {
                            self.get_vector(c).ok().flatten().map(|vec| {
                                let dist = compute_distance(&neighbor_vec, &vec, self.config.metric);
                                Candidate { id: c, distance: OrderedFloat(dist) }
                            })
                        })
                        .collect();
                    scored.sort_by_key(|c| c.distance);

                    let new_edges: Vec<NodeId> = scored.into_iter().take(max_neighbors).map(|c| c.id).collect();
                    self.set_neighbors(neighbor, layer, new_edges).await?;
                }
            }

            if layer == 0 {
                self.graph_store.delete_neighbors(id).await?;
            }
        }

        self.upper_graph.remove_node(id);
        self.sampler.remove_hash(id);
        self.graph_store.delete_metadata(id).await?;
        // Note: vector data in file is NOT deleted (would create holes)
        // Could mark as deleted or implement compaction later

        Ok(())
    }

    /// Get vector by ID
    pub fn get(&self, id: NodeId) -> Result<Option<Vec<f32>>, LsmVecError> {
        self.get_vector(id)
    }
}
```

---

## Phase 8: Error Types

### File: `src/error.rs`

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LsmVecError {
    #[error("Storage error: {0}")]
    Storage(#[from] slatedb::SlateDBError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Invalid file: {0}")]
    InvalidFile(String),

    #[error("Node not found: {0}")]
    NodeNotFound(u64),

    #[error("Index is empty")]
    EmptyIndex,
}
```

---

## Phase 9: Public API

### File: `src/lib.rs`

```rust
//! LSM-VEC: Disk-based vector search with HNSW on LSM-tree
//!
//! Paper: https://arxiv.org/abs/2505.17152

mod distance;
mod error;
mod graph_store;
mod index;
mod simhash;
mod types;
mod upper_graph;
mod vector_file;

pub use error::LsmVecError;
pub use types::{DistanceMetric, LsmVecConfig, NodeId, SearchResult};

use slatedb::Db;
use std::path::Path;
use std::sync::Arc;

use crate::graph_store::GraphStore;
use crate::index::HnswIndex;
use crate::vector_file::VectorFile;

/// LSM-VEC database
pub struct LsmVecDb {
    index: HnswIndex,
    #[allow(dead_code)]
    vectors: Arc<VectorFile>,
}

impl LsmVecDb {
    /// Open LSM-VEC database
    ///
    /// - `db`: SlateDB instance for graph edges
    /// - `vector_path`: Path to vector file (separate from LSM)
    /// - `config`: Index configuration
    pub async fn open(
        db: Arc<Db>,
        vector_path: &Path,
        config: LsmVecConfig,
    ) -> Result<Self, LsmVecError> {
        let vectors = Arc::new(VectorFile::open(vector_path, config.dimension)?);
        let graph_store = Arc::new(GraphStore::new(db));
        let index = HnswIndex::new(vectors.clone(), graph_store, config).await?;

        Ok(Self { index, vectors })
    }

    /// Insert vector, returns ID
    pub async fn insert(&self, vector: Vec<f32>) -> Result<NodeId, LsmVecError> {
        self.index.insert(vector).await
    }

    /// Delete by ID
    pub async fn delete(&self, id: NodeId) -> Result<(), LsmVecError> {
        self.index.delete(id).await
    }

    /// k-NN search
    pub async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, LsmVecError> {
        self.index.search(query, k).await
    }

    /// Search with custom ef
    pub async fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>, LsmVecError> {
        self.index.search_with_ef(query, k, ef).await
    }

    /// Get vector by ID
    pub fn get(&self, id: NodeId) -> Result<Option<Vec<f32>>, LsmVecError> {
        self.index.get(id)
    }
}
```

---

## File Structure

```
lsm-vec/
├── Cargo.toml
├── IMPLEMENTATION_PLAN.md
└── src/
    ├── lib.rs           # Public API
    ├── types.rs         # Types & config
    ├── error.rs         # Errors
    ├── distance.rs      # Distance functions
    ├── vector_file.rs   # Contiguous vector file (O(1) access)
    ├── simhash.rs       # SimHash sampler
    ├── graph_store.rs   # SlateDB graph edges
    ├── upper_graph.rs   # In-memory upper layers
    └── index.rs         # HNSW index
```

---

## Implementation Milestones

### Milestone 1: Foundation
- [ ] `types.rs` - Core types
- [ ] `error.rs` - Error types
- [ ] `distance.rs` - Distance functions

### Milestone 2: Vector File
- [ ] `vector_file.rs` - Contiguous file with O(1) access
- [ ] Header format
- [ ] Read/write operations

### Milestone 3: SimHash
- [ ] `simhash.rs` - Random projections
- [ ] Hash computation
- [ ] Candidate filtering

### Milestone 4: Graph Storage
- [ ] `graph_store.rs` - SlateDB wrapper for edges
- [ ] `upper_graph.rs` - In-memory upper layers

### Milestone 5: HNSW Index
- [ ] `index.rs` - Search algorithm
- [ ] Insert algorithm
- [ ] Delete algorithm

### Milestone 6: Public API
- [ ] `lib.rs` - LsmVecDb wrapper

### Milestone 7: Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Benchmarks

---

## Key Design Decisions

1. **Vectors separate from LSM**: Per paper, O(1) access via offset
2. **Graph edges in SlateDB**: Efficient for updates, compaction
3. **Upper layers in memory**: <1% of nodes, fast navigation
4. **SimHash filtering**: Reduces I/O during layer 0 search
5. **No custom caching**: SlateDB handles caching

---

## Future: Vector Type in SlateDB

For better integration, could add a `VectorStore` type to SlateDB that:
- Manages contiguous vector files alongside SSTs
- Participates in compaction (reordering)
- Handles object storage for vectors
- Integrates with SlateDB's caching

This would be a separate effort after the basic LSM-VEC works.
