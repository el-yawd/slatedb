//! SimHash-based probabilistic sampling for neighbor filtering
//!
//! Per the LSM-VEC paper, SimHash is used to filter candidates during search
//! to reduce I/O operations. The hash function uses random projections:
//!
//! Hash(x) = [sgn(x·a_1), sgn(x·a_2), ..., sgn(x·a_m)]
//!
//! where a_i are random vectors sampled from N(0, I_d).

use bitvec::prelude::*;
use parking_lot::RwLock;
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};
use std::collections::HashMap;

use crate::types::{Dimension, NodeId};

/// SimHash sampler for probabilistic neighbor filtering
pub struct SimHashSampler {
    /// Random projection vectors from N(0, I_d)
    /// Shape: [num_bits][dimension]
    projections: Vec<Vec<f32>>,

    /// Number of hash bits (m in paper)
    num_bits: usize,

    /// Vector dimension
    dimension: Dimension,

    /// Cached hash codes for all nodes
    hashes: RwLock<HashMap<NodeId, BitVec>>,
}

impl SimHashSampler {
    /// Create a new SimHash sampler with random projections
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `num_bits` - Number of hash bits (m)
    /// * `seed` - Random seed for reproducibility
    pub fn new(dimension: Dimension, num_bits: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = StandardNormal;

        // Generate m random projection vectors from N(0, I_d)
        let projections: Vec<Vec<f32>> = (0..num_bits)
            .map(|_| {
                (0..dimension)
                    .map(|_| normal.sample(&mut rng))
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

    /// Compute hash for a vector
    ///
    /// Hash(x) = [sgn(x·a_1), sgn(x·a_2), ..., sgn(x·a_m)]
    pub fn compute_hash(&self, vector: &[f32]) -> BitVec {
        debug_assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension mismatch"
        );

        let mut hash = BitVec::with_capacity(self.num_bits);

        for proj in &self.projections {
            let dot: f32 = vector.iter().zip(proj.iter()).map(|(x, p)| x * p).sum();
            hash.push(dot >= 0.0);
        }

        hash
    }

    /// Store hash for a node
    pub fn store_hash(&self, id: NodeId, hash: BitVec) {
        self.hashes.write().insert(id, hash);
    }

    /// Compute and store hash for a node
    pub fn compute_and_store(&self, id: NodeId, vector: &[f32]) {
        let hash = self.compute_hash(vector);
        self.store_hash(id, hash);
    }

    /// Get hash for a node
    pub fn get_hash(&self, id: NodeId) -> Option<BitVec> {
        self.hashes.read().get(&id).cloned()
    }

    /// Remove hash for a node
    pub fn remove_hash(&self, id: NodeId) {
        self.hashes.write().remove(&id);
    }

    /// Count collisions (matching bits) between two hashes
    ///
    /// Higher collision count indicates vectors are more likely to be similar.
    #[inline]
    pub fn collision_count(h1: &BitVec, h2: &BitVec) -> usize {
        debug_assert_eq!(h1.len(), h2.len(), "Hash lengths must match");
        h1.iter().zip(h2.iter()).filter(|(a, b)| *a == *b).count()
    }

    /// Compute the collision threshold for a given epsilon
    ///
    /// Per paper: Pr[||q-u|| <= delta | #Col(q,u) >= T] >= 1 - epsilon
    /// We use a conservative threshold that allows most candidates through
    /// while still filtering out clearly dissimilar vectors.
    pub fn threshold(&self, epsilon: f64) -> usize {
        // Use a relatively low threshold to avoid filtering out good candidates
        // At 50% match rate (random), we'd expect ~32/64 collisions
        // We require slightly above random: 50% + epsilon * 10% of bits
        let threshold = (self.num_bits as f64 * (0.5 + epsilon * 0.1)) as usize;
        // Ensure threshold is at least slightly above pure random
        threshold.max(self.num_bits / 2 + 1)
    }

    /// Filter candidates by SimHash threshold
    ///
    /// Returns candidates where collision count >= threshold.
    /// Candidates without stored hashes are kept (conservative).
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
                match hashes.get(id) {
                    Some(candidate_hash) => {
                        Self::collision_count(query_hash, candidate_hash) >= threshold
                    }
                    // Keep candidates without stored hashes (conservative)
                    None => true,
                }
            })
            .copied()
            .collect()
    }

    /// Get the number of stored hashes
    pub fn len(&self) -> usize {
        self.hashes.read().len()
    }

    /// Check if no hashes are stored
    pub fn is_empty(&self) -> bool {
        self.hashes.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_computation() {
        let sampler = SimHashSampler::new(4, 64, 42);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let h1 = sampler.compute_hash(&v1);
        assert_eq!(h1.len(), 64);

        // Same vector should produce same hash
        let h1_again = sampler.compute_hash(&v1);
        assert_eq!(h1, h1_again);
    }

    #[test]
    fn test_similar_vectors_high_collision() {
        let sampler = SimHashSampler::new(4, 64, 42);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.99, 0.01, 0.0, 0.0]; // Very similar

        let h1 = sampler.compute_hash(&v1);
        let h2 = sampler.compute_hash(&v2);

        let collisions = SimHashSampler::collision_count(&h1, &h2);
        // Similar vectors should have high collision count
        assert!(collisions > 32, "Expected high collision count for similar vectors");
    }

    #[test]
    fn test_orthogonal_vectors_low_collision() {
        let sampler = SimHashSampler::new(4, 64, 42);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0]; // Orthogonal

        let h1 = sampler.compute_hash(&v1);
        let h2 = sampler.compute_hash(&v2);

        let collisions = SimHashSampler::collision_count(&h1, &h2);
        // Orthogonal vectors should have ~50% collision (random)
        // Allow wider range due to randomness in projections
        assert!(
            (20..45).contains(&collisions),
            "Expected ~50% collision for orthogonal vectors, got {}",
            collisions
        );
    }

    #[test]
    fn test_store_and_retrieve() {
        let sampler = SimHashSampler::new(4, 64, 42);

        let v = vec![1.0, 2.0, 3.0, 4.0];
        sampler.compute_and_store(0, &v);

        assert!(sampler.get_hash(0).is_some());
        assert!(sampler.get_hash(1).is_none());

        sampler.remove_hash(0);
        assert!(sampler.get_hash(0).is_none());
    }

    #[test]
    fn test_filter_candidates() {
        let sampler = SimHashSampler::new(4, 64, 42);

        // Store some hashes
        let v_query = vec![1.0, 0.0, 0.0, 0.0];
        let v_similar = vec![0.9, 0.1, 0.0, 0.0];
        let v_different = vec![0.0, 0.0, 0.0, 1.0];

        sampler.compute_and_store(1, &v_similar);
        sampler.compute_and_store(2, &v_different);

        let query_hash = sampler.compute_hash(&v_query);
        let threshold = sampler.threshold(0.1);

        let candidates = vec![1, 2, 3]; // 3 doesn't have a hash
        let filtered = sampler.filter_candidates(&query_hash, &candidates, threshold);

        // Should keep similar (1) and unknown (3), may filter out different (2)
        assert!(filtered.contains(&1));
        assert!(filtered.contains(&3)); // No hash, kept conservatively
    }
}
