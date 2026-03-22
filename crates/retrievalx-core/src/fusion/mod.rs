#![forbid(unsafe_code)]

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FusionMethod {
    ReciprocalRankFusion,
    LinearCombination,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Normalizer {
    MinMax,
    ZScore,
    Cdf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FusionConfig {
    pub method: FusionMethod,
    pub alpha: f32,
    pub normalizer: Option<Normalizer>,
    pub rrf_k: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::ReciprocalRankFusion,
            alpha: 0.5,
            normalizer: Some(Normalizer::Cdf),
            rrf_k: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FusionResult {
    pub doc_id: String,
    pub score: f32,
}

pub fn reciprocal_rank_fusion(
    primary: &[(String, f32)],
    secondary: &[(String, f32)],
    k: usize,
) -> Vec<FusionResult> {
    let mut scores = HashMap::<String, f32>::new();

    for (rank, (doc_id, _)) in primary.iter().enumerate() {
        let value = 1.0 / (k + rank + 1) as f32;
        *scores.entry(doc_id.clone()).or_insert(0.0) += value;
    }

    for (rank, (doc_id, _)) in secondary.iter().enumerate() {
        let value = 1.0 / (k + rank + 1) as f32;
        *scores.entry(doc_id.clone()).or_insert(0.0) += value;
    }

    sort_scores(scores)
}

pub fn linear_combination(
    primary: &[(String, f32)],
    secondary: &[(String, f32)],
    alpha: f32,
) -> Vec<FusionResult> {
    let alpha = alpha.clamp(0.0, 1.0);
    let p = scores_as_map(primary);
    let s = scores_as_map(secondary);

    let mut all_docs = BTreeSet::new();
    all_docs.extend(p.keys().cloned());
    all_docs.extend(s.keys().cloned());

    let mut merged = HashMap::new();
    for doc_id in all_docs {
        let pscore = p.get(&doc_id).copied().unwrap_or(0.0);
        let sscore = s.get(&doc_id).copied().unwrap_or(0.0);
        merged.insert(doc_id, alpha * pscore + (1.0 - alpha) * sscore);
    }

    sort_scores(merged)
}

pub fn min_max_normalize(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let min = scores.iter().copied().min_by(f32::total_cmp).unwrap_or(0.0);
    let max = scores.iter().copied().max_by(f32::total_cmp).unwrap_or(1.0);

    if (max - min).abs() < f32::EPSILON {
        return vec![1.0; scores.len()];
    }

    scores
        .iter()
        .map(|value| (value - min) / (max - min))
        .collect()
}

pub fn z_score_normalize(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let variance = scores
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f32>()
        / scores.len() as f32;

    if variance.abs() < f32::EPSILON {
        return vec![0.0; scores.len()];
    }

    let std = variance.sqrt();
    scores.iter().map(|value| (*value - mean) / std).collect()
}

pub fn cdf_normalize(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let mut sorted = scores
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut out = vec![0.0; scores.len()];
    let denom = scores.len().saturating_sub(1).max(1) as f32;
    for (rank, (idx, _)) in sorted.into_iter().enumerate() {
        out[idx] = rank as f32 / denom;
    }
    out
}

fn scores_as_map(scores: &[(String, f32)]) -> HashMap<String, f32> {
    scores.iter().cloned().collect()
}

fn sort_scores(scores: HashMap<String, f32>) -> Vec<FusionResult> {
    let mut out = scores
        .into_iter()
        .map(|(doc_id, score)| FusionResult { doc_id, score })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}
