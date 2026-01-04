//! Graph storage in SlateDB (edges only, no vectors)
//!
//! Per the LSM-VEC paper, graph edges are stored in the LSM-tree while vectors
//! are stored separately. This module handles the graph edge storage.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use slatedb::Db;
use std::sync::Arc;

use crate::error::Result;
use crate::types::{LayerId, NodeId, NodeMetadata};

/// Key prefixes for different data types in SlateDB
mod prefix {
    /// Graph edges (layer 0): e{node_id} -> [neighbor_ids]
    pub const EDGE: u8 = b'e';
    /// Node metadata: m{node_id} -> NodeMetadata
    pub const META: u8 = b'm';
    /// Index metadata: i{key} -> value
    pub const INDEX: u8 = b'i';
}

/// Graph storage backed by SlateDB
///
/// Stores graph edges and metadata, but NOT vector data.
pub struct GraphStore {
    db: Arc<Db>,
}

impl GraphStore {
    /// Create a new graph store
    pub fn new(db: Arc<Db>) -> Self {
        Self { db }
    }

    // ========== Key Encoding ==========

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

    /// Store neighbors for a node (layer 0 graph edges)
    pub async fn put_neighbors(&self, id: NodeId, neighbors: &[NodeId]) -> Result<()> {
        let key = Self::edge_key(id);
        let mut buf = BytesMut::with_capacity(neighbors.len() * 8);
        for &n in neighbors {
            buf.put_u64(n);
        }
        self.db.put(&key, &buf.freeze()).await?;
        Ok(())
    }

    /// Get neighbors for a node (layer 0 graph edges)
    pub async fn get_neighbors(&self, id: NodeId) -> Result<Option<Vec<NodeId>>> {
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

    /// Delete neighbors for a node
    pub async fn delete_neighbors(&self, id: NodeId) -> Result<()> {
        let key = Self::edge_key(id);
        self.db.delete(&key).await?;
        Ok(())
    }

    // ========== Node Metadata ==========

    /// Store node metadata
    pub async fn put_metadata(&self, meta: &NodeMetadata) -> Result<()> {
        let key = Self::meta_key(meta.id);
        let mut buf = BytesMut::with_capacity(9);
        buf.put_u64(meta.id);
        buf.put_u8(meta.max_layer);
        self.db.put(&key, &buf.freeze()).await?;
        Ok(())
    }

    /// Get node metadata
    pub async fn get_metadata(&self, id: NodeId) -> Result<Option<NodeMetadata>> {
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

    /// Delete node metadata
    pub async fn delete_metadata(&self, id: NodeId) -> Result<()> {
        let key = Self::meta_key(id);
        self.db.delete(&key).await?;
        Ok(())
    }

    // ========== Index Metadata ==========

    /// Store entry point (the starting node for searches)
    pub async fn put_entry_point(&self, id: NodeId, layer: LayerId) -> Result<()> {
        let key = Self::index_key("entry_point");
        let mut buf = BytesMut::with_capacity(9);
        buf.put_u64(id);
        buf.put_u8(layer);
        self.db.put(&key, &buf.freeze()).await?;
        Ok(())
    }

    /// Get entry point
    pub async fn get_entry_point(&self) -> Result<Option<(NodeId, LayerId)>> {
        let key = Self::index_key("entry_point");
        match self.db.get(&key).await? {
            Some(bytes) => {
                let mut buf = &bytes[..];
                Ok(Some((buf.get_u64(), buf.get_u8())))
            }
            None => Ok(None),
        }
    }

    /// Delete entry point
    pub async fn delete_entry_point(&self) -> Result<()> {
        let key = Self::index_key("entry_point");
        self.db.delete(&key).await?;
        Ok(())
    }
}
