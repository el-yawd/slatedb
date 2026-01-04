//! Error types for LSM-VEC

use thiserror::Error;

/// LSM-VEC error type
#[derive(Error, Debug)]
pub enum LsmVecError {
    /// Storage error from SlateDB
    #[error("Storage error: {0}")]
    Storage(#[from] slatedb::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid vector dimension
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Invalid file format
    #[error("Invalid file: {0}")]
    InvalidFile(String),

    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(u64),

    /// Index is empty
    #[error("Index is empty")]
    EmptyIndex,
}

/// Result type for LSM-VEC operations
pub type Result<T> = std::result::Result<T, LsmVecError>;
