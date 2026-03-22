#![forbid(unsafe_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::FieldConfig;

#[derive(Debug, Clone)]
pub struct TermScoreInput<'a> {
    pub term: &'a str,
    pub tf: u32,
    pub doc_len: u32,
    pub avgdl: f32,
    pub doc_freq: u32,
    pub num_docs: u32,
    pub collection_freq: u64,
    pub query_boost: f32,
}

pub trait Scorer: Send + Sync {
    fn score(&self, input: &TermScoreInput<'_>) -> f32;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScoringVariant {
    Okapi {
        k1: f32,
        b: f32,
    },
    Plus {
        k1: f32,
        b: f32,
        delta: f32,
    },
    L {
        k1: f32,
        b: f32,
        c: f32,
    },
    Adpt {
        b: f32,
    },
    F {
        k1: f32,
        b: f32,
        fields: Vec<FieldConfig>,
    },
    T {
        default_k1: f32,
        b: f32,
        term_k1: HashMap<String, f32>,
    },
    Atire {
        k1: f32,
        b: f32,
    },
    TfIdf,
}

impl ScoringVariant {
    pub fn build(&self) -> Box<dyn Scorer> {
        match self {
            Self::Okapi { k1, b } => Box::new(BM25Okapi { k1: *k1, b: *b }),
            Self::Plus { k1, b, delta } => Box::new(BM25Plus {
                k1: *k1,
                b: *b,
                delta: *delta,
            }),
            Self::L { k1, b, c } => Box::new(BM25L {
                k1: *k1,
                b: *b,
                c: *c,
            }),
            Self::Adpt { b } => Box::new(BM25Adpt { b: *b }),
            Self::F { k1, b, fields } => Box::new(BM25F {
                k1: *k1,
                b: *b,
                fields: fields.clone(),
            }),
            Self::T {
                default_k1,
                b,
                term_k1,
            } => Box::new(BM25T {
                default_k1: *default_k1,
                b: *b,
                term_k1: term_k1.clone(),
            }),
            Self::Atire { k1, b } => Box::new(AtireBM25 { k1: *k1, b: *b }),
            Self::TfIdf => Box::new(TfIdf),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BM25Okapi {
    pub k1: f32,
    pub b: f32,
}

impl Scorer for BM25Okapi {
    fn score(&self, input: &TermScoreInput<'_>) -> f32 {
        let idf = robertson_idf(input.num_docs, input.doc_freq);
        let tf = input.tf as f32;
        let norm =
            self.k1 * (1.0 - self.b + self.b * (input.doc_len as f32 / input.avgdl.max(1e-6)));
        let tf_component = (tf * (self.k1 + 1.0)) / (tf + norm);
        idf * tf_component * input.query_boost
    }
}

#[derive(Debug, Clone)]
pub struct BM25Plus {
    pub k1: f32,
    pub b: f32,
    pub delta: f32,
}

impl Scorer for BM25Plus {
    fn score(&self, input: &TermScoreInput<'_>) -> f32 {
        let idf = robertson_idf(input.num_docs, input.doc_freq);
        let tf = input.tf as f32;
        let norm =
            self.k1 * (1.0 - self.b + self.b * (input.doc_len as f32 / input.avgdl.max(1e-6)));
        let tf_component = ((tf * (self.k1 + 1.0)) / (tf + norm)) + self.delta;
        idf * tf_component * input.query_boost
    }
}

#[derive(Debug, Clone)]
pub struct BM25L {
    pub k1: f32,
    pub b: f32,
    pub c: f32,
}

impl Scorer for BM25L {
    fn score(&self, input: &TermScoreInput<'_>) -> f32 {
        let idf = robertson_idf(input.num_docs, input.doc_freq);
        let tf = input.tf as f32;
        let ctd = tf / (1.0 - self.b + self.b * (input.doc_len as f32 / input.avgdl.max(1e-6)));
        let tf_component = ((self.k1 + 1.0) * (ctd + self.c)) / (self.k1 + ctd + self.c);
        idf * tf_component * input.query_boost
    }
}

#[derive(Debug, Clone)]
pub struct BM25Adpt {
    pub b: f32,
}

impl Scorer for BM25Adpt {
    fn score(&self, input: &TermScoreInput<'_>) -> f32 {
        let mean_cf = (input.collection_freq as f32 / input.num_docs.max(1) as f32).max(1e-6);
        let adaptive_k1 = (0.7 + mean_cf.ln_1p()).clamp(0.8, 2.2);
        let idf = robertson_idf(input.num_docs, input.doc_freq);
        let tf = input.tf as f32;
        let norm =
            adaptive_k1 * (1.0 - self.b + self.b * (input.doc_len as f32 / input.avgdl.max(1e-6)));
        let tf_component = (tf * (adaptive_k1 + 1.0)) / (tf + norm);
        idf * tf_component * input.query_boost
    }
}

#[derive(Debug, Clone)]
pub struct BM25F {
    pub k1: f32,
    pub b: f32,
    pub fields: Vec<FieldConfig>,
}

impl Scorer for BM25F {
    fn score(&self, input: &TermScoreInput<'_>) -> f32 {
        let field_factor = if self.fields.is_empty() {
            1.0
        } else {
            self.fields
                .iter()
                .map(|field| field.weight * (1.0 - field.b + self.b * field.b))
                .sum::<f32>()
                / self.fields.len() as f32
        };
        let idf = robertson_idf(input.num_docs, input.doc_freq);
        let tf = (input.tf as f32) * field_factor.max(0.1);
        let norm =
            self.k1 * (1.0 - self.b + self.b * (input.doc_len as f32 / input.avgdl.max(1e-6)));
        let tf_component = (tf * (self.k1 + 1.0)) / (tf + norm);
        idf * tf_component * input.query_boost
    }
}

#[derive(Debug, Clone)]
pub struct BM25T {
    pub default_k1: f32,
    pub b: f32,
    pub term_k1: HashMap<String, f32>,
}

impl Scorer for BM25T {
    fn score(&self, input: &TermScoreInput<'_>) -> f32 {
        let k1 = self
            .term_k1
            .get(input.term)
            .copied()
            .unwrap_or(self.default_k1)
            .clamp(0.1, 4.0);
        let idf = robertson_idf(input.num_docs, input.doc_freq);
        let tf = input.tf as f32;
        let norm = k1 * (1.0 - self.b + self.b * (input.doc_len as f32 / input.avgdl.max(1e-6)));
        let tf_component = (tf * (k1 + 1.0)) / (tf + norm);
        idf * tf_component * input.query_boost
    }
}

#[derive(Debug, Clone)]
pub struct AtireBM25 {
    pub k1: f32,
    pub b: f32,
}

impl Scorer for AtireBM25 {
    fn score(&self, input: &TermScoreInput<'_>) -> f32 {
        let idf = atire_idf(input.num_docs, input.doc_freq);
        let tf = input.tf as f32;
        let norm =
            self.k1 * (1.0 - self.b + self.b * (input.doc_len as f32 / input.avgdl.max(1e-6)));
        let tf_component = (tf * (self.k1 + 1.0)) / (tf + norm);
        idf * tf_component * input.query_boost
    }
}

#[derive(Debug, Clone)]
pub struct TfIdf;

impl Scorer for TfIdf {
    fn score(&self, input: &TermScoreInput<'_>) -> f32 {
        let idf = (input.num_docs as f32 / (1.0 + input.doc_freq as f32)).ln_1p();
        input.tf as f32 * idf * input.query_boost
    }
}

fn robertson_idf(num_docs: u32, doc_freq: u32) -> f32 {
    let n = num_docs as f32;
    let df = doc_freq as f32;
    ((n - df + 0.5) / (df + 0.5)).ln_1p().max(0.0)
}

fn atire_idf(num_docs: u32, doc_freq: u32) -> f32 {
    ((num_docs as f32 + 1.0) / (doc_freq as f32 + 0.5))
        .ln()
        .max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25plus_nonzero_for_matching_doc() {
        let scorer = BM25Plus {
            k1: 1.2,
            b: 0.75,
            delta: 0.5,
        };
        let score = scorer.score(&TermScoreInput {
            term: "rust",
            tf: 1,
            doc_len: 100,
            avgdl: 100.0,
            doc_freq: 10,
            num_docs: 100,
            collection_freq: 100,
            query_boost: 1.0,
        });
        assert!(score > 0.0);
    }
}
