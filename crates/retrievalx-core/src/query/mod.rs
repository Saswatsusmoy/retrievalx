#![forbid(unsafe_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryTerm {
    pub term: String,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BagOfWordsQuery {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightedQuery {
    pub terms: Vec<QueryTerm>,
}

impl WeightedQuery {
    pub fn from_map(weights: HashMap<String, f32>) -> Self {
        let terms = weights
            .into_iter()
            .map(|(term, weight)| QueryTerm { term, weight })
            .collect();
        Self { terms }
    }

    pub fn normalize(&mut self) {
        let sum: f32 = self.terms.iter().map(|term| term.weight.abs()).sum();
        if sum > 0.0 {
            for term in &mut self.terms {
                term.weight /= sum;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BooleanClause {
    Must,
    Should,
    MustNot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BooleanQuery {
    pub must: Vec<String>,
    pub should: Vec<String>,
    pub must_not: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhraseQuery {
    pub terms: Vec<String>,
    pub window: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExpansionMethod {
    RM3,
    Rocchio,
    Bo1,
    Synonym,
}

pub type DocTermWeights = HashMap<String, f32>;

pub trait QueryExpander: Send + Sync {
    fn expand(&self, query: &WeightedQuery, feedback_docs: &[DocTermWeights]) -> WeightedQuery;
}

#[derive(Debug, Clone)]
pub struct RM3Expander {
    pub num_expansion_terms: usize,
    pub lambda: f32,
}

impl QueryExpander for RM3Expander {
    fn expand(&self, query: &WeightedQuery, feedback_docs: &[DocTermWeights]) -> WeightedQuery {
        let mut model = HashMap::<String, f32>::new();

        for doc in feedback_docs {
            for (term, value) in doc {
                *model.entry(term.clone()).or_insert(0.0) += *value;
            }
        }

        let mut expanded = query.terms.clone();
        let mut candidates: Vec<(String, f32)> = model.into_iter().collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (term, score) in candidates.into_iter().take(self.num_expansion_terms) {
            let weight = score * self.lambda;
            expanded.push(QueryTerm { term, weight });
        }

        let mut output = WeightedQuery { terms: expanded };
        output.normalize();
        output
    }
}

#[derive(Debug, Clone)]
pub struct RocchioExpander {
    pub alpha: f32,
    pub beta: f32,
    pub num_expansion_terms: usize,
}

impl QueryExpander for RocchioExpander {
    fn expand(&self, query: &WeightedQuery, feedback_docs: &[DocTermWeights]) -> WeightedQuery {
        let mut weights = HashMap::<String, f32>::new();

        for term in &query.terms {
            *weights.entry(term.term.clone()).or_insert(0.0) += self.alpha * term.weight;
        }

        if !feedback_docs.is_empty() {
            let inv = 1.0 / feedback_docs.len() as f32;
            for doc in feedback_docs {
                for (term, score) in doc {
                    *weights.entry(term.clone()).or_insert(0.0) += self.beta * score * inv;
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = weights.into_iter().collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        sorted.truncate(self.num_expansion_terms.max(query.terms.len()));

        let mut expanded = WeightedQuery {
            terms: sorted
                .into_iter()
                .map(|(term, weight)| QueryTerm { term, weight })
                .collect(),
        };
        expanded.normalize();
        expanded
    }
}

#[derive(Debug, Clone)]
pub struct Bo1Expander {
    pub num_expansion_terms: usize,
}

impl QueryExpander for Bo1Expander {
    fn expand(&self, query: &WeightedQuery, feedback_docs: &[DocTermWeights]) -> WeightedQuery {
        let mut scores = HashMap::<String, f32>::new();
        for doc in feedback_docs {
            for (term, tf) in doc {
                let gain = (1.0 + tf).ln_1p();
                *scores.entry(term.clone()).or_insert(0.0) += gain;
            }
        }

        let mut terms = query.terms.clone();
        let mut extra: Vec<(String, f32)> = scores.into_iter().collect();
        extra.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (term, weight) in extra.into_iter().take(self.num_expansion_terms) {
            terms.push(QueryTerm { term, weight });
        }

        let mut expanded = WeightedQuery { terms };
        expanded.normalize();
        expanded
    }
}

#[derive(Debug, Clone)]
pub struct SynonymExpander {
    pub synonyms: HashMap<String, Vec<String>>,
    pub decay: f32,
}

impl QueryExpander for SynonymExpander {
    fn expand(&self, query: &WeightedQuery, _feedback_docs: &[DocTermWeights]) -> WeightedQuery {
        let mut terms = query.terms.clone();

        for term in &query.terms {
            if let Some(synonyms) = self.synonyms.get(&term.term) {
                for synonym in synonyms {
                    terms.push(QueryTerm {
                        term: synonym.clone(),
                        weight: term.weight * self.decay,
                    });
                }
            }
        }

        let mut expanded = WeightedQuery { terms };
        expanded.normalize();
        expanded
    }
}
