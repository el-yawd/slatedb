//! Core types and configuration for LSM-VEC

/// Unique identifier for a vector/node
pub type NodeId = u64;

/// Layer identifier in HNSW graph
pub type LayerId = u8;

/// Vector dimensionality
pub type Dimension = usize;

/// Distance metric for similarity computation
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance
    #[default]
    L2,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

/// Configuration for LSM-VEC index
#[derive(Clone, Debug)]
pub struct LsmVecConfig {
    /// Vector dimensionality (fixed for all vectors)
    pub dimension: Dimension,

    /// Distance metric
    pub metric: DistanceMetric,

    /// Max neighbors at layer 0 (M_max in paper, default: 32)
    pub m_max: usize,

    /// Max neighbors at upper layers (M in paper, default: 16)
    pub m: usize,

    /// Level multiplier for random level assignment: 1/ln(M)
    pub m_level: f64,

    /// Search width during construction (ef_construction, default: 200)
    pub ef_construction: usize,

    /// Default search width for queries (ef_search, default: 100)
    pub ef_search: usize,

    /// Number of SimHash bits (m in paper, default: 64)
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

impl LsmVecConfig {
    /// Create a new configuration with the specified dimension
    pub fn new(dimension: Dimension) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set M parameter (max neighbors at upper layers)
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_level = 1.0 / (m as f64).ln();
        self
    }

    /// Set M_max parameter (max neighbors at layer 0)
    pub fn with_m_max(mut self, m_max: usize) -> Self {
        self.m_max = m_max;
        self
    }

    /// Set ef_construction parameter
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set ef_search parameter
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }
}

/// Search result containing node ID and distance
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// Node ID
    pub id: NodeId,
    /// Distance from query
    pub distance: f32,
}

/// Node metadata stored for each vector
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeMetadata {
    /// Node ID
    pub id: NodeId,
    /// Maximum layer this node exists in
    pub max_layer: LayerId,
}
