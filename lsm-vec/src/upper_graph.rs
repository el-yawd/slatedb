//! In-memory graph for HNSW upper layers
//!
//! Per the LSM-VEC paper: "less than 1% of all nodes reside above the bottom layer,
//! which makes them suitable for in-memory storage even at billion-scale."
//!
//! This module stores the upper layers (layer >= 1) in memory for fast navigation.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::types::{LayerId, NodeId};

/// Maximum number of layers in the HNSW graph
pub const MAX_LAYERS: usize = 16;

/// In-memory graph for HNSW upper layers (layer >= 1)
///
/// The bottom layer (layer 0) is stored on disk via SlateDB.
/// Upper layers contain <1% of nodes and are kept in memory.
pub struct UpperGraph {
    /// Neighbors at each layer: layers[layer][node_id] = neighbors
    layers: Vec<RwLock<HashMap<NodeId, Vec<NodeId>>>>,

    /// Current entry point node ID
    entry_point: AtomicU64,

    /// Entry point's layer
    entry_point_layer: AtomicU8,

    /// Whether an entry point is set
    has_entry_point: AtomicU8,
}

impl UpperGraph {
    /// Create a new empty upper graph
    pub fn new() -> Self {
        Self {
            layers: (0..MAX_LAYERS)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
            entry_point: AtomicU64::new(0),
            entry_point_layer: AtomicU8::new(0),
            has_entry_point: AtomicU8::new(0),
        }
    }

    /// Get the entry point if set
    ///
    /// Returns (node_id, layer) or None if no entry point exists.
    pub fn get_entry_point(&self) -> Option<(NodeId, LayerId)> {
        if self.has_entry_point.load(Ordering::Acquire) == 0 {
            return None;
        }
        Some((
            self.entry_point.load(Ordering::Acquire),
            self.entry_point_layer.load(Ordering::Acquire),
        ))
    }

    /// Set the entry point
    pub fn set_entry_point(&self, id: NodeId, layer: LayerId) {
        self.entry_point.store(id, Ordering::Release);
        self.entry_point_layer.store(layer, Ordering::Release);
        self.has_entry_point.store(1, Ordering::Release);
    }

    /// Clear the entry point
    pub fn clear_entry_point(&self) {
        self.has_entry_point.store(0, Ordering::Release);
    }

    /// Get neighbors at a specific layer (for layers >= 1)
    ///
    /// Returns None if the node doesn't exist at this layer.
    /// Layer 0 should be queried from the disk-based GraphStore.
    pub fn get_neighbors(&self, id: NodeId, layer: LayerId) -> Option<Vec<NodeId>> {
        if layer == 0 || layer as usize >= MAX_LAYERS {
            return None;
        }
        self.layers[layer as usize].read().get(&id).cloned()
    }

    /// Set neighbors at a specific layer (for layers >= 1)
    ///
    /// Layer 0 should be stored in the disk-based GraphStore.
    pub fn set_neighbors(&self, id: NodeId, layer: LayerId, neighbors: Vec<NodeId>) {
        if layer == 0 || layer as usize >= MAX_LAYERS {
            return;
        }
        self.layers[layer as usize].write().insert(id, neighbors);
    }

    /// Remove a node from all upper layers
    pub fn remove_node(&self, id: NodeId) {
        for layer in &self.layers {
            layer.write().remove(&id);
        }
    }

    /// Check if a node exists at a specific layer
    pub fn contains(&self, id: NodeId, layer: LayerId) -> bool {
        if layer == 0 || layer as usize >= MAX_LAYERS {
            return false;
        }
        self.layers[layer as usize].read().contains_key(&id)
    }

    /// Get the number of nodes at a specific layer
    pub fn layer_size(&self, layer: LayerId) -> usize {
        if layer == 0 || layer as usize >= MAX_LAYERS {
            return 0;
        }
        self.layers[layer as usize].read().len()
    }

    /// Get total number of nodes across all upper layers
    pub fn total_nodes(&self) -> usize {
        self.layers.iter().map(|l| l.read().len()).sum()
    }
}

impl Default for UpperGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_point() {
        let graph = UpperGraph::new();

        // Initially no entry point
        assert!(graph.get_entry_point().is_none());

        // Set entry point
        graph.set_entry_point(42, 3);
        assert_eq!(graph.get_entry_point(), Some((42, 3)));

        // Update entry point
        graph.set_entry_point(100, 5);
        assert_eq!(graph.get_entry_point(), Some((100, 5)));

        // Clear entry point
        graph.clear_entry_point();
        assert!(graph.get_entry_point().is_none());
    }

    #[test]
    fn test_neighbors() {
        let graph = UpperGraph::new();

        // Layer 0 should return None (stored on disk)
        graph.set_neighbors(1, 0, vec![2, 3]);
        assert!(graph.get_neighbors(1, 0).is_none());

        // Layer 1+ should work
        graph.set_neighbors(1, 1, vec![2, 3, 4]);
        assert_eq!(graph.get_neighbors(1, 1), Some(vec![2, 3, 4]));

        // Non-existent node
        assert!(graph.get_neighbors(999, 1).is_none());
    }

    #[test]
    fn test_remove_node() {
        let graph = UpperGraph::new();

        graph.set_neighbors(1, 1, vec![2, 3]);
        graph.set_neighbors(1, 2, vec![4, 5]);
        graph.set_neighbors(1, 3, vec![6]);

        assert!(graph.contains(1, 1));
        assert!(graph.contains(1, 2));
        assert!(graph.contains(1, 3));

        graph.remove_node(1);

        assert!(!graph.contains(1, 1));
        assert!(!graph.contains(1, 2));
        assert!(!graph.contains(1, 3));
    }

    #[test]
    fn test_layer_size() {
        let graph = UpperGraph::new();

        graph.set_neighbors(1, 1, vec![2]);
        graph.set_neighbors(2, 1, vec![3]);
        graph.set_neighbors(3, 2, vec![4]);

        assert_eq!(graph.layer_size(0), 0); // Layer 0 not stored here
        assert_eq!(graph.layer_size(1), 2);
        assert_eq!(graph.layer_size(2), 1);
        assert_eq!(graph.total_nodes(), 3);
    }
}
