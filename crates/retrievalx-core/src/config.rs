#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use crate::query::ExpansionMethod;
use crate::retrieval::RetrievalStrategy;
use crate::scoring::ScoringVariant;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldConfig {
    pub name: String,
    pub weight: f32,
    pub b: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistConfig {
    pub load_mode: String,
    pub enable_wal: bool,
    pub compression: bool,
}

impl Default for PersistConfig {
    fn default() -> Self {
        Self {
            load_mode: "in_memory".to_owned(),
            enable_wal: false,
            compression: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpansionConfig {
    pub method: ExpansionMethod,
    pub num_feedback_docs: usize,
    pub num_expansion_terms: usize,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            method: ExpansionMethod::RM3,
            num_feedback_docs: 10,
            num_expansion_terms: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BM25Config {
    pub scoring: ScoringVariant,
    pub tokenizer: retrievalx_tokenize::TokenizerConfig,
    pub retrieval: RetrievalStrategy,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            scoring: ScoringVariant::Okapi { k1: 1.2, b: 0.75 },
            tokenizer: retrievalx_tokenize::TokenizerConfig::default(),
            retrieval: RetrievalStrategy::BlockMaxWand { top_k_budget: 0 },
        }
    }
}
