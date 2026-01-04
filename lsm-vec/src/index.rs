//! HNSW Index implementation for LSM-VEC
//!
//! This module implements the core HNSW (Hierarchical Navigable Small World) index
//! with the LSM-VEC architecture:
//! - Upper layers (1+) in memory
//! - Layer 0 edges on disk via SlateDB
//! - Vector data in separate contiguous file
//! - SimHash-based probabilistic sampling during search

use ordered_float::OrderedFloat;
use rand::prelude::*;
use rand::rng;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use crate::distance::compute_distance;
use crate::error::Result;
use crate::graph_store::GraphStore;
use crate::simhash::SimHashSampler;
use crate::types::*;
use crate::upper_graph::UpperGraph;
use crate::vector_file::VectorFile;

/// Candidate node with distance for priority queue operations
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
    /// Vector file storage (separate from LSM, O(1) access)
    vectors: Arc<VectorFile>,

    /// Graph edge storage (SlateDB, layer 0 only)
    graph_store: Arc<GraphStore>,

    /// In-memory upper layers (layer >= 1)
    upper_graph: UpperGraph,

    /// SimHash sampler for probabilistic filtering
    sampler: SimHashSampler,

    /// Index configuration
    config: LsmVecConfig,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub async fn new(
        vectors: Arc<VectorFile>,
        graph_store: Arc<GraphStore>,
        config: LsmVecConfig,
    ) -> Result<Self> {
        let sampler = SimHashSampler::new(config.dimension, config.simhash_bits, 42);

        let index = Self {
            vectors,
            graph_store,
            upper_graph: UpperGraph::new(),
            sampler,
            config,
        };

        // Load entry point from persistent storage
        if let Some((id, layer)) = index.graph_store.get_entry_point().await? {
            index.upper_graph.set_entry_point(id, layer);
        }

        // TODO: Load upper layer graph from persistent storage for recovery
        // For now, upper layers are rebuilt on restart

        Ok(index)
    }

    /// Generate a random level using exponential distribution
    ///
    /// Per HNSW paper: L = floor(-ln(uniform(0,1)) * m_level)
    fn random_level(&self) -> LayerId {
        let mut rng = rng();
        let f: f64 = rng.random();
        let level = (-f.ln() * self.config.m_level).floor() as LayerId;
        level.min(15) // Cap at max layers - 1
    }

    /// Get vector by ID from file (O(1))
    fn get_vector(&self, id: NodeId) -> Result<Option<Vec<f32>>> {
        self.vectors.get(id)
    }

    /// Get neighbors at a specific layer
    async fn get_neighbors(&self, id: NodeId, layer: LayerId) -> Result<Vec<NodeId>> {
        if layer == 0 {
            // Layer 0 is on disk
            Ok(self.graph_store.get_neighbors(id).await?.unwrap_or_default())
        } else {
            // Upper layers are in memory
            Ok(self.upper_graph.get_neighbors(id, layer).unwrap_or_default())
        }
    }

    /// Set neighbors at a specific layer
    async fn set_neighbors(
        &self,
        id: NodeId,
        layer: LayerId,
        neighbors: Vec<NodeId>,
    ) -> Result<()> {
        if layer == 0 {
            self.graph_store.put_neighbors(id, &neighbors).await?;
        } else {
            self.upper_graph.set_neighbors(id, layer, neighbors);
        }
        Ok(())
    }

    // ========== Search Algorithm ==========

    /// Search a single layer, returning up to ef closest candidates
    ///
    /// This implements the layer search from the HNSW paper with optional
    /// SimHash filtering for layer 0 (per LSM-VEC paper).
    async fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[NodeId],
        layer: LayerId,
        ef: usize,
        use_simhash: bool,
    ) -> Result<Vec<Candidate>> {
        // Min-heap for candidates to explore (closest first)
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        // Max-heap for results (keeps ef closest)
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        // Visited set to avoid re-processing
        let mut visited: HashSet<NodeId> = HashSet::new();

        // Compute query hash for SimHash filtering (only for layer 0)
        let query_hash = if use_simhash {
            Some(self.sampler.compute_hash(query))
        } else {
            None
        };
        let threshold = self.sampler.threshold(self.config.simhash_epsilon);

        // Initialize with entry points
        for &ep in entry_points {
            if visited.insert(ep) {
                if let Some(vec) = self.get_vector(ep)? {
                    let dist = compute_distance(query, &vec, self.config.metric);
                    let candidate = Candidate {
                        id: ep,
                        distance: OrderedFloat(dist),
                    };
                    candidates.push(Reverse(candidate.clone()));
                    results.push(candidate);
                }
            }
        }

        // Greedy search
        while let Some(Reverse(current)) = candidates.pop() {
            // Early termination: if current is farther than ef-th result
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }

            // Get neighbors
            let mut neighbors = self.get_neighbors(current.id, layer).await?;

            // Apply SimHash filtering for layer 0 (per LSM-VEC paper)
            if layer == 0 {
                if let Some(ref qh) = query_hash {
                    neighbors = self.sampler.filter_candidates(qh, &neighbors, threshold);
                }
            }

            // Evaluate neighbors
            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    if let Some(vec) = self.get_vector(neighbor)? {
                        let dist = compute_distance(query, &vec, self.config.metric);

                        let should_add = results.len() < ef
                            || dist < results.peek().map(|c| c.distance.0).unwrap_or(f32::MAX);

                        if should_add {
                            let candidate = Candidate {
                                id: neighbor,
                                distance: OrderedFloat(dist),
                            };
                            candidates.push(Reverse(candidate.clone()));
                            results.push(candidate);

                            // Keep only ef results
                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Return sorted by distance (ascending)
        let mut result_vec: Vec<_> = results.into_iter().collect();
        result_vec.sort_by_key(|c| c.distance);
        Ok(result_vec)
    }

    /// k-NN search across all layers
    pub async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_ef(query, k, self.config.ef_search).await
    }

    /// k-NN search with custom ef parameter
    pub async fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>> {
        // Get entry point
        let (entry_point, max_layer) = match self.upper_graph.get_entry_point() {
            Some(ep) => ep,
            None => return Ok(vec![]), // Empty index
        };

        let mut current_best = vec![entry_point];

        // Navigate through upper layers (greedy search, ef=1)
        for layer in (1..=max_layer).rev() {
            let results = self
                .search_layer(query, &current_best, layer, 1, false)
                .await?;
            if !results.is_empty() {
                current_best = results.into_iter().map(|c| c.id).collect();
            }
        }

        // Search layer 0 with full ef, using SimHash filtering
        let results = self
            .search_layer(query, &current_best, 0, ef.max(k), true)
            .await?;

        Ok(results
            .into_iter()
            .take(k)
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance.0,
            })
            .collect())
    }

    // ========== Insert Algorithm ==========

    /// Insert a vector into the index
    ///
    /// This implements Algorithm 1 from the HNSW paper adapted for LSM-VEC.
    pub async fn insert(&self, vector: Vec<f32>) -> Result<NodeId> {
        // 1. Append vector to file and get ID
        let id = self.vectors.append(&vector)?;

        // 2. Compute and store SimHash
        self.sampler.compute_and_store(id, &vector);

        // 3. Sample random level
        let node_layer = self.random_level();

        // 4. Store node metadata
        self.graph_store
            .put_metadata(&NodeMetadata {
                id,
                max_layer: node_layer,
            })
            .await?;

        // 5. Get current entry point
        let entry_point = self.upper_graph.get_entry_point();

        if let Some((ep, max_layer)) = entry_point {
            let mut current_best = vec![ep];

            // Navigate to node_layer through upper layers (greedy search)
            for layer in (node_layer.saturating_add(1)..=max_layer).rev() {
                let results = self
                    .search_layer(&vector, &current_best, layer, 1, false)
                    .await?;
                if !results.is_empty() {
                    current_best = results.into_iter().map(|c| c.id).collect();
                }
            }

            // Insert at each layer from node_layer down to 0
            let start_layer = node_layer.min(max_layer);
            for layer in (0..=start_layer).rev() {
                let results = self
                    .search_layer(
                        &vector,
                        &current_best,
                        layer,
                        self.config.ef_construction,
                        layer == 0,
                    )
                    .await?;

                // Select neighbors using heuristic
                let max_neighbors = if layer == 0 {
                    self.config.m_max
                } else {
                    self.config.m
                };
                let neighbors = self.select_neighbors(&vector, results, max_neighbors)?;

                // Set edges for new node
                self.set_neighbors(id, layer, neighbors.clone()).await?;

                // Update neighbors' edge lists (bidirectional)
                for &neighbor in &neighbors {
                    let mut neighbor_edges = self.get_neighbors(neighbor, layer).await?;
                    neighbor_edges.push(id);

                    // Prune if exceeds max_neighbors
                    if neighbor_edges.len() > max_neighbors {
                        if let Some(neighbor_vec) = self.get_vector(neighbor)? {
                            neighbor_edges =
                                self.prune_neighbors(&neighbor_vec, neighbor_edges, max_neighbors)?;
                        }
                    }
                    self.set_neighbors(neighbor, layer, neighbor_edges).await?;
                }

                // Use selected neighbors as entry points for next layer
                current_best = neighbors;
            }

            // Update entry point if new node has higher layer
            if node_layer > max_layer {
                self.upper_graph.set_entry_point(id, node_layer);
                self.graph_store.put_entry_point(id, node_layer).await?;
            }
        } else {
            // First node - set as entry point
            self.upper_graph.set_entry_point(id, node_layer);
            self.graph_store.put_entry_point(id, node_layer).await?;
        }

        Ok(id)
    }

    /// Select neighbors using the heuristic from HNSW paper (Algorithm 4)
    ///
    /// Prefers neighbors that are both close to the query and not too close to each other.
    fn select_neighbors(
        &self,
        _query: &[f32],
        candidates: Vec<Candidate>,
        m: usize,
    ) -> Result<Vec<NodeId>> {
        let mut result = Vec::with_capacity(m);

        for candidate in candidates {
            if result.len() >= m {
                break;
            }

            let candidate_vec = match self.get_vector(candidate.id)? {
                Some(v) => v,
                None => continue,
            };

            // Check if candidate is closer to query than to any already selected neighbor
            let mut is_good = true;
            for &selected_id in &result {
                if let Some(selected_vec) = self.get_vector(selected_id)? {
                    let dist_to_selected =
                        compute_distance(&candidate_vec, &selected_vec, self.config.metric);
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

    /// Prune neighbors to max size, keeping closest
    fn prune_neighbors(
        &self,
        node_vec: &[f32],
        neighbors: Vec<NodeId>,
        max: usize,
    ) -> Result<Vec<NodeId>> {
        let mut scored: Vec<Candidate> = neighbors
            .into_iter()
            .filter_map(|id| {
                self.get_vector(id).ok().flatten().map(|vec| {
                    let dist = compute_distance(node_vec, &vec, self.config.metric);
                    Candidate {
                        id,
                        distance: OrderedFloat(dist),
                    }
                })
            })
            .collect();

        scored.sort_by_key(|c| c.distance);
        Ok(scored.into_iter().take(max).map(|c| c.id).collect())
    }

    // ========== Delete Algorithm ==========

    /// Delete a vector from the index
    ///
    /// This implements Algorithm 2 from the LSM-VEC paper.
    pub async fn delete(&self, id: NodeId) -> Result<()> {
        // Get node metadata
        let meta = match self.graph_store.get_metadata(id).await? {
            Some(m) => m,
            None => return Ok(()), // Node doesn't exist
        };

        // Get layer 0 neighbors first (needed for entry point update)
        let layer0_neighbors = self.get_neighbors(id, 0).await?;

        // For each layer, remove edges and reconnect neighbors
        for layer in 0..=meta.max_layer {
            // Get neighbors of deleted node
            let neighbors = self.get_neighbors(id, layer).await?;

            // For each neighbor, remove edge to deleted node and reconnect
            for &neighbor in &neighbors {
                let mut neighbor_edges = self.get_neighbors(neighbor, layer).await?;
                neighbor_edges.retain(|&n| n != id);

                // Collect candidates for reconnection (neighbors of neighbors)
                let mut candidates: HashSet<NodeId> = HashSet::new();
                candidates.extend(&neighbors);
                candidates.extend(&neighbor_edges);
                candidates.remove(&neighbor);
                candidates.remove(&id);

                // Re-select best neighbors from candidates
                if let Some(neighbor_vec) = self.get_vector(neighbor)? {
                    let max_neighbors = if layer == 0 {
                        self.config.m_max
                    } else {
                        self.config.m
                    };

                    let mut scored: Vec<Candidate> = candidates
                        .into_iter()
                        .filter_map(|c| {
                            self.get_vector(c).ok().flatten().map(|vec| {
                                let dist = compute_distance(&neighbor_vec, &vec, self.config.metric);
                                Candidate {
                                    id: c,
                                    distance: OrderedFloat(dist),
                                }
                            })
                        })
                        .collect();
                    scored.sort_by_key(|c| c.distance);

                    let new_edges: Vec<NodeId> =
                        scored.into_iter().take(max_neighbors).map(|c| c.id).collect();
                    self.set_neighbors(neighbor, layer, new_edges).await?;
                }
            }

            // Delete edges for the deleted node
            if layer == 0 {
                self.graph_store.delete_neighbors(id).await?;
            }
        }

        // Remove from upper graph
        self.upper_graph.remove_node(id);

        // Remove SimHash
        self.sampler.remove_hash(id);

        // Delete metadata
        self.graph_store.delete_metadata(id).await?;

        // Note: Vector data in file is NOT deleted (would create holes)
        // This is a design decision - compaction would need to handle this

        // Update entry point if necessary
        if let Some((ep, _)) = self.upper_graph.get_entry_point() {
            if ep == id {
                // Entry point was deleted - need to find a new one
                // Try to find another node that exists
                let mut new_entry_point = None;

                // Check if any of the deleted node's layer 0 neighbors can be the new entry point
                for &neighbor in &layer0_neighbors {
                    if let Some(neighbor_meta) = self.graph_store.get_metadata(neighbor).await? {
                        if new_entry_point.is_none()
                            || neighbor_meta.max_layer
                                > new_entry_point.as_ref().map(|(_, l)| *l).unwrap_or(0)
                        {
                            new_entry_point = Some((neighbor, neighbor_meta.max_layer));
                        }
                    }
                }

                if let Some((new_ep, new_layer)) = new_entry_point {
                    self.upper_graph.set_entry_point(new_ep, new_layer);
                    self.graph_store.put_entry_point(new_ep, new_layer).await?;
                } else {
                    // No neighbors found, clear entry point
                    self.upper_graph.clear_entry_point();
                    self.graph_store.delete_entry_point().await?;
                }
            }
        }

        Ok(())
    }

    /// Get a vector by ID
    pub fn get(&self, id: NodeId) -> Result<Option<Vec<f32>>> {
        self.get_vector(id)
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            vector_count: self.vectors.count().unwrap_or(0),
            upper_layer_nodes: self.upper_graph.total_nodes(),
            simhash_count: self.sampler.len(),
        }
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Total number of vectors
    pub vector_count: u64,
    /// Number of nodes in upper layers
    pub upper_layer_nodes: usize,
    /// Number of stored SimHash codes
    pub simhash_count: usize,
}
