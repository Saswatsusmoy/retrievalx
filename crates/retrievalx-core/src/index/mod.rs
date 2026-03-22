#![forbid(unsafe_code)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use crate::config::BM25Config;
use crate::error::CoreError;
use crate::query::{BooleanQuery, QueryTerm, WeightedQuery};
use crate::retrieval::{PostingScoreInput, RetrievalStrategy, TermPostingList};
use crate::scoring::ScoringVariant;

const BLOCK_SIZE: usize = 64;
const PARALLEL_INGEST_MIN_DOCS: usize = 256;

#[derive(Debug)]
struct IngestDocument {
    external_id: String,
    text: String,
    fields: HashMap<String, String>,
}

#[derive(Debug)]
struct PreparedDocument {
    external_id: String,
    text: String,
    fields: HashMap<String, String>,
    doc_len: u32,
    token_counts: HashMap<String, u32>,
    term_positions: Vec<(String, Vec<u32>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Posting {
    pub doc_id: u32,
    pub term_freq: u32,
    pub positions: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostingBlock {
    pub start_doc_id: u32,
    pub end_doc_id: u32,
    pub max_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TermStats {
    pub term_id: u32,
    pub doc_freq: u32,
    pub collection_freq: u64,
    pub max_tf: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentRecord {
    pub external_id: String,
    pub text: String,
    pub fields: HashMap<String, String>,
    pub token_counts: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    pub doc_id: u32,
    pub external_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexStats {
    pub num_docs: usize,
    pub num_live_docs: usize,
    pub vocabulary_size: usize,
    pub avgdl: f32,
    pub tombstones: usize,
    pub total_terms: usize,
    pub build_timestamp_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexSnapshot {
    pub config: BM25Config,
    pub postings: HashMap<String, Vec<Posting>>,
    pub block_max: HashMap<String, Vec<PostingBlock>>,
    pub doc_lengths: Vec<u32>,
    pub docs: Vec<Option<DocumentRecord>>,
    pub external_to_internal: HashMap<String, u32>,
    pub vocabulary: HashMap<String, TermStats>,
    pub tombstones: Vec<u32>,
    pub total_live_doc_len: u64,
    pub live_doc_count: u32,
    pub next_term_id: u32,
    pub build_timestamp_unix: u64,
}

#[derive(Debug)]
pub struct InvertedIndex {
    config: BM25Config,
    tokenizer: retrievalx_tokenize::TokenizerPipeline,
    postings: HashMap<String, Vec<Posting>>,
    block_max: HashMap<String, Vec<PostingBlock>>,
    vocabulary: HashMap<String, TermStats>,
    doc_lengths: Vec<u32>,
    docs: Vec<Option<DocumentRecord>>,
    external_to_internal: HashMap<String, u32>,
    tombstones: RoaringBitmap,
    total_live_doc_len: u64,
    live_doc_count: u32,
    next_term_id: u32,
    build_timestamp_unix: u64,
}

impl InvertedIndex {
    pub fn new(config: BM25Config) -> Result<Self, CoreError> {
        let tokenizer = retrievalx_tokenize::TokenizerPipeline::new(config.tokenizer.clone())?;
        Ok(Self {
            config,
            tokenizer,
            postings: HashMap::new(),
            block_max: HashMap::new(),
            vocabulary: HashMap::new(),
            doc_lengths: Vec::new(),
            docs: Vec::new(),
            external_to_internal: HashMap::new(),
            tombstones: RoaringBitmap::new(),
            total_live_doc_len: 0,
            live_doc_count: 0,
            next_term_id: 0,
            build_timestamp_unix: now_unix(),
        })
    }

    pub fn from_snapshot(snapshot: IndexSnapshot) -> Result<Self, CoreError> {
        let tokenizer =
            retrievalx_tokenize::TokenizerPipeline::new(snapshot.config.tokenizer.clone())?;
        let tombstones = snapshot.tombstones.iter().copied().collect();

        let mut index = Self {
            config: snapshot.config,
            tokenizer,
            postings: snapshot.postings,
            block_max: snapshot.block_max,
            vocabulary: snapshot.vocabulary,
            doc_lengths: snapshot.doc_lengths,
            docs: snapshot.docs,
            external_to_internal: snapshot.external_to_internal,
            tombstones,
            total_live_doc_len: snapshot.total_live_doc_len,
            live_doc_count: snapshot.live_doc_count,
            next_term_id: snapshot.next_term_id,
            build_timestamp_unix: snapshot.build_timestamp_unix,
        };
        // Rebuild block stats only when the active strategy needs block-level bounds.
        if index.uses_block_max() {
            // Rebuild from postings so persisted indexes remain compatible across
            // block-bound encoding upgrades.
            index.rebuild_block_max();
        }
        Ok(index)
    }

    pub fn snapshot(&self) -> IndexSnapshot {
        IndexSnapshot {
            config: self.config.clone(),
            postings: self.postings.clone(),
            block_max: self.block_max.clone(),
            doc_lengths: self.doc_lengths.clone(),
            docs: self.docs.clone(),
            external_to_internal: self.external_to_internal.clone(),
            vocabulary: self.vocabulary.clone(),
            tombstones: self.tombstones.iter().collect(),
            total_live_doc_len: self.total_live_doc_len,
            live_doc_count: self.live_doc_count,
            next_term_id: self.next_term_id,
            build_timestamp_unix: self.build_timestamp_unix,
        }
    }

    pub fn config(&self) -> &BM25Config {
        &self.config
    }

    pub fn from_documents<I>(documents: I, config: BM25Config) -> Result<Self, CoreError>
    where
        I: IntoIterator<Item = String>,
    {
        let mut index = Self::new(config)?;
        index.insert_batch(documents)?;
        Ok(index)
    }

    pub fn insert_batch<I>(&mut self, documents: I) -> Result<(), CoreError>
    where
        I: IntoIterator<Item = String>,
    {
        let base_doc_id = self.doc_lengths.len() as u32;
        let docs = documents
            .into_iter()
            .enumerate()
            .map(|(offset, text)| {
                let doc_id = doc_id_with_offset(base_doc_id, offset)?;
                Ok(IngestDocument {
                    external_id: format!("doc-{doc_id}"),
                    text,
                    fields: HashMap::new(),
                })
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        self.insert_preprocessed_batch(docs)
    }

    pub fn insert_documents_with_ids<I>(&mut self, documents: I) -> Result<(), CoreError>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let docs = documents
            .into_iter()
            .map(|(external_id, text)| IngestDocument {
                external_id,
                text,
                fields: HashMap::new(),
            })
            .collect::<Vec<_>>();
        self.insert_preprocessed_batch(docs)
    }

    pub fn insert_document(
        &mut self,
        external_id: Option<String>,
        text: String,
        fields: HashMap<String, String>,
    ) -> Result<u32, CoreError> {
        let mut touched_terms = self.uses_block_max().then(HashSet::<String>::new);
        let doc_id =
            self.insert_document_internal(external_id, text, fields, touched_terms.as_mut())?;
        if let Some(touched_terms) = touched_terms {
            self.rebuild_block_max_for_terms(touched_terms);
        }
        Ok(doc_id)
    }

    fn insert_document_internal(
        &mut self,
        external_id: Option<String>,
        text: String,
        fields: HashMap<String, String>,
        mut touched_terms: Option<&mut HashSet<String>>,
    ) -> Result<u32, CoreError> {
        let doc_id = self.doc_lengths.len() as u32;
        let ext_id = external_id.unwrap_or_else(|| format!("doc-{doc_id}"));
        if self.external_to_internal.contains_key(&ext_id) {
            return Err(CoreError::InvalidArgument(format!(
                "duplicate external id: {ext_id}"
            )));
        }

        let mut combined = text.clone();
        for value in fields.values() {
            combined.push(' ');
            combined.push_str(value);
        }

        let tokens = self.tokenizer.tokenize(&combined);
        let doc_len = tokens.len() as u32;
        let mut term_positions: HashMap<String, Vec<u32>> = HashMap::new();
        for (pos, token) in tokens.iter().enumerate() {
            term_positions
                .entry(token.clone())
                .or_default()
                .push(pos as u32);
        }

        let mut token_counts = HashMap::new();
        for (term, positions) in &term_positions {
            token_counts.insert(term.clone(), positions.len() as u32);
            if let Some(tracked_terms) = touched_terms.as_deref_mut() {
                tracked_terms.insert(term.clone());
            }
            self.upsert_posting(term, doc_id, positions.clone());
        }

        let record = DocumentRecord {
            external_id: ext_id.clone(),
            text,
            fields,
            token_counts,
        };

        self.doc_lengths.push(doc_len);
        self.docs.push(Some(record));
        self.external_to_internal.insert(ext_id, doc_id);
        self.total_live_doc_len += u64::from(doc_len);
        self.live_doc_count += 1;

        Ok(doc_id)
    }

    pub fn delete_by_doc_id(&mut self, doc_id: u32) -> Result<(), CoreError> {
        let idx = doc_id as usize;
        if idx >= self.docs.len() {
            return Err(CoreError::DocumentNotFound(doc_id));
        }
        if self.tombstones.contains(doc_id) {
            return Ok(());
        }

        self.tombstones.insert(doc_id);
        if let Some(Some(doc)) = self.docs.get(idx) {
            self.total_live_doc_len = self
                .total_live_doc_len
                .saturating_sub(u64::from(self.doc_lengths[idx]));
            self.live_doc_count = self.live_doc_count.saturating_sub(1);

            let terms = doc.token_counts.clone();
            for (term, tf) in terms {
                if let Some(stats) = self.vocabulary.get_mut(&term) {
                    stats.doc_freq = stats.doc_freq.saturating_sub(1);
                    stats.collection_freq = stats.collection_freq.saturating_sub(u64::from(tf));
                    // Keep max_tf monotonic during tombstoning to avoid O(df) rescans on delete.
                    // `compact()` rebuilds exact stats when users request physical cleanup.
                }
            }
        }

        Ok(())
    }

    pub fn delete_by_external_id(&mut self, external_id: &str) -> Result<(), CoreError> {
        if let Some(&doc_id) = self.external_to_internal.get(external_id) {
            self.delete_by_doc_id(doc_id)
        } else {
            Err(CoreError::ExternalIdNotFound(external_id.to_owned()))
        }
    }

    pub fn compact(&mut self) -> Result<(), CoreError> {
        if self.tombstones.is_empty() {
            return Ok(());
        }

        let live_docs = self
            .docs
            .iter()
            .enumerate()
            .filter_map(|(doc_id, doc)| {
                let id = doc_id as u32;
                if self.tombstones.contains(id) {
                    None
                } else {
                    doc.clone()
                }
            })
            .collect::<Vec<_>>();

        let config = self.config.clone();
        *self = Self::new(config)?;

        let docs = live_docs
            .into_iter()
            .map(|doc| IngestDocument {
                external_id: doc.external_id,
                text: doc.text,
                fields: doc.fields,
            })
            .collect::<Vec<_>>();
        self.insert_preprocessed_batch(docs)
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let terms = self.tokenizer.tokenize(query);
        let weighted = WeightedQuery {
            terms: terms
                .into_iter()
                .map(|term| QueryTerm { term, weight: 1.0 })
                .collect(),
        };
        self.search_weighted_internal(&weighted, top_k, None)
    }

    pub fn search_weighted(&self, query: &WeightedQuery, top_k: usize) -> Vec<SearchResult> {
        self.search_weighted_internal(query, top_k, None)
    }

    fn search_weighted_internal(
        &self,
        query: &WeightedQuery,
        top_k: usize,
        candidate_filter: Option<&HashSet<u32>>,
    ) -> Vec<SearchResult> {
        if top_k == 0 {
            return Vec::new();
        }

        let mut merged_weights = HashMap::<String, f32>::new();
        for term in &query.terms {
            if term.weight > 0.0 {
                *merged_weights.entry(term.term.clone()).or_insert(0.0) += term.weight;
            }
        }

        let mut posting_lists = Vec::<TermPostingList>::new();
        let mut term_contexts = Vec::<TermScoringContext>::new();
        let avgdl = self.avgdl().max(1e-6);
        let num_docs = self.live_doc_count.max(1);
        let field_factor = bm25f_field_factor(&self.config.scoring);
        let effective_retrieval = match (&self.config.retrieval, &self.config.scoring) {
            // MaxScore degenerates on TfIdf in dynamic corpora and can be slower than exhaustive.
            // Preserve correctness while avoiding the pathological overhead.
            (RetrievalStrategy::MaxScore, ScoringVariant::TfIdf) => {
                RetrievalStrategy::ExhaustiveDAAT
            }
            _ => self.config.retrieval.clone(),
        };
        let use_block_max_bounds =
            matches!(&effective_retrieval, RetrievalStrategy::BlockMaxWand { .. });

        for (term, weight) in merged_weights {
            let Some(postings) = self.postings.get(&term) else {
                continue;
            };
            let Some(stats) = self.vocabulary.get(&term) else {
                continue;
            };

            let term_context = TermScoringContext {
                query_weight: weight,
                max_tf: stats.max_tf.max(1) as f32,
                robertson_idf: robertson_idf(num_docs, stats.doc_freq),
                atire_idf: atire_idf(num_docs, stats.doc_freq),
                tfidf_idf: (num_docs as f32 / (1.0 + stats.doc_freq as f32)).ln_1p(),
                adaptive_k1: {
                    let mean_cf = (stats.collection_freq as f32 / num_docs as f32).max(1e-6);
                    (0.7 + mean_cf.ln_1p()).clamp(0.8, 2.2)
                },
                term_k1: match &self.config.scoring {
                    ScoringVariant::T {
                        default_k1,
                        term_k1,
                        ..
                    } => Some(
                        term_k1
                            .get(&term)
                            .copied()
                            .unwrap_or(*default_k1)
                            .clamp(0.1, 4.0),
                    ),
                    _ => None,
                },
            };

            let posting_inputs = postings
                .iter()
                .filter(|posting| !self.tombstones.contains(posting.doc_id))
                .filter(|posting| {
                    candidate_filter.map_or(true, |allowed| allowed.contains(&posting.doc_id))
                })
                .map(|posting| PostingScoreInput {
                    doc_id: posting.doc_id,
                    tf: posting.term_freq,
                })
                .collect::<Vec<_>>();
            if posting_inputs.is_empty() {
                continue;
            }

            let block_upper_bounds = if use_block_max_bounds {
                self.block_max
                    .get(&term)
                    .map(|blocks| {
                        blocks
                            .iter()
                            .map(|block| {
                                score_upper_bound_for_tf(
                                    &self.config.scoring,
                                    &term_context,
                                    block.max_impact.max(1.0),
                                    avgdl,
                                    field_factor,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| {
                        vec![score_upper_bound_for_tf(
                            &self.config.scoring,
                            &term_context,
                            term_context.max_tf,
                            avgdl,
                            field_factor,
                        )]
                    })
            } else {
                vec![score_upper_bound_for_tf(
                    &self.config.scoring,
                    &term_context,
                    term_context.max_tf,
                    avgdl,
                    field_factor,
                )]
            };
            let upper_bound = block_upper_bounds.iter().copied().fold(0.0_f32, f32::max);

            posting_lists.push(TermPostingList {
                term: term.clone(),
                postings: posting_inputs,
                upper_bound,
                block_upper_bounds,
            });
            term_contexts.push(term_context);
        }

        if posting_lists.is_empty() {
            return Vec::new();
        }

        let doc_lengths = &self.doc_lengths;
        let tombstones = &self.tombstones;

        let mut score_fn = |term_idx: usize, doc_id: u32, tf: u32| -> f32 {
            if tombstones.contains(doc_id) {
                return 0.0;
            }
            if candidate_filter.is_some_and(|allowed| !allowed.contains(&doc_id)) {
                return 0.0;
            }

            let Some(context) = term_contexts.get(term_idx) else {
                return 0.0;
            };
            let Some(doc_len) = doc_lengths.get(doc_id as usize).copied() else {
                return 0.0;
            };

            score_term_for_variant(
                &self.config.scoring,
                context,
                tf as f32,
                doc_len as f32,
                avgdl,
                field_factor,
            )
        };

        let retriever = effective_retrieval.build();
        retriever
            .rank(&posting_lists, top_k, &mut score_fn)
            .hits
            .into_iter()
            .filter_map(|(doc_id, score)| {
                self.docs.get(doc_id as usize).and_then(|doc| {
                    doc.as_ref().map(|record| SearchResult {
                        doc_id,
                        external_id: record.external_id.clone(),
                        score,
                    })
                })
            })
            .collect()
    }

    pub fn search_boolean(&self, query: &BooleanQuery, top_k: usize) -> Vec<SearchResult> {
        let must_set = self.docs_for_terms(&query.must, true);
        let should_set = self.docs_for_terms(&query.should, false);
        let must_not = self.docs_for_terms(&query.must_not, false);

        let candidates = if query.must.is_empty() {
            should_set
        } else {
            must_set
        };

        let final_candidates = candidates
            .difference(&must_not)
            .copied()
            .collect::<HashSet<_>>();

        let weighted = WeightedQuery {
            terms: query
                .must
                .iter()
                .chain(query.should.iter())
                .map(|term| QueryTerm {
                    term: term.clone(),
                    weight: 1.0,
                })
                .collect(),
        };

        self.search_weighted_internal(&weighted, top_k, Some(&final_candidates))
    }

    pub fn search_phrase(&self, terms: &[String], window: u32, top_k: usize) -> Vec<SearchResult> {
        if terms.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let matched = self.phrase_match(terms, window);
        if matched.is_empty() {
            return Vec::new();
        }

        let weighted = WeightedQuery {
            terms: terms
                .iter()
                .cloned()
                .map(|term| QueryTerm { term, weight: 1.0 })
                .collect(),
        };

        self.search_weighted_internal(&weighted, top_k, Some(&matched))
    }

    pub fn phrase_match(&self, terms: &[String], window: u32) -> HashSet<u32> {
        if terms.is_empty() {
            return HashSet::new();
        }

        let mut common_docs = self.docs_for_terms(terms, true);
        common_docs.retain(|doc_id| {
            let mut term_positions = Vec::<&[u32]>::with_capacity(terms.len());
            for term in terms {
                let positions = self
                    .postings
                    .get(term)
                    .and_then(|postings| positions_for_doc(postings, *doc_id));
                let Some(positions) = positions else {
                    return false;
                };
                term_positions.push(positions);
            }
            within_window(&term_positions, window)
        });

        common_docs
    }

    pub fn vocabulary(&self) -> &HashMap<String, TermStats> {
        &self.vocabulary
    }

    pub fn postings(&self, term: &str) -> Option<&[Posting]> {
        self.postings.get(term).map(Vec::as_slice)
    }

    pub fn doc_length(&self, doc_id: u32) -> Option<u32> {
        self.doc_lengths.get(doc_id as usize).copied()
    }

    pub fn avgdl(&self) -> f32 {
        if self.live_doc_count == 0 {
            0.0
        } else {
            self.total_live_doc_len as f32 / self.live_doc_count as f32
        }
    }

    pub fn sparse_vector_for_query(&self, query: &str) -> Vec<(u32, f32)> {
        let mut term_counts = HashMap::<String, u32>::new();
        for token in self.tokenizer.tokenize(query) {
            *term_counts.entry(token).or_insert(0) += 1;
        }

        let num_docs = self.live_doc_count.max(1) as f32;
        let mut vector = term_counts
            .into_iter()
            .filter_map(|(term, tf)| {
                self.vocabulary.get(&term).map(|stats| {
                    let doc_freq = stats.doc_freq as f32;
                    let idf = ((num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                        .ln_1p()
                        .max(0.0);
                    (stats.term_id, tf as f32 * idf)
                })
            })
            .collect::<Vec<_>>();

        vector.sort_by_key(|(term_id, _)| *term_id);
        vector
    }

    pub fn sparse_vector_for_document(&self, doc_id: u32) -> Option<Vec<(u32, f32)>> {
        if self.tombstones.contains(doc_id) {
            return None;
        }

        let doc = self.docs.get(doc_id as usize)?.as_ref()?;
        let mut vector = doc
            .token_counts
            .iter()
            .filter_map(|(term, tf)| {
                self.vocabulary
                    .get(term)
                    .map(|stats| (stats.term_id, *tf as f32))
            })
            .collect::<Vec<_>>();
        vector.sort_by_key(|(term_id, _)| *term_id);
        Some(vector)
    }

    pub fn live_documents(&self) -> Vec<(u32, DocumentRecord)> {
        self.docs
            .iter()
            .enumerate()
            .filter_map(|(doc_id, doc)| {
                let id = doc_id as u32;
                if self.tombstones.contains(id) {
                    None
                } else {
                    doc.clone().map(|record| (id, record))
                }
            })
            .collect()
    }

    pub fn live_documents_iter(&self) -> impl Iterator<Item = (u32, &DocumentRecord)> + '_ {
        self.docs.iter().enumerate().filter_map(|(doc_id, doc)| {
            let id = doc_id as u32;
            if self.tombstones.contains(id) {
                None
            } else {
                doc.as_ref().map(|record| (id, record))
            }
        })
    }

    pub fn for_each_live_document<F>(&self, mut f: F)
    where
        F: FnMut(u32, &DocumentRecord),
    {
        for (doc_id, record) in self.live_documents_iter() {
            f(doc_id, record);
        }
    }

    pub fn stats(&self) -> IndexStats {
        let total_terms = self
            .docs
            .iter()
            .flatten()
            .map(|doc| doc.token_counts.len())
            .sum::<usize>();

        IndexStats {
            num_docs: self.docs.len(),
            num_live_docs: self.live_doc_count as usize,
            vocabulary_size: self.vocabulary.len(),
            avgdl: self.avgdl(),
            tombstones: self.tombstones.len() as usize,
            total_terms,
            build_timestamp_unix: self.build_timestamp_unix,
        }
    }

    fn upsert_posting(&mut self, term: &str, doc_id: u32, positions: Vec<u32>) {
        let tf = positions.len() as u32;
        let postings = self.postings.entry(term.to_owned()).or_default();
        postings.push(Posting {
            doc_id,
            term_freq: tf,
            positions,
        });

        let stats = self.vocabulary.entry(term.to_owned()).or_insert_with(|| {
            let term_id = self.next_term_id;
            self.next_term_id += 1;
            TermStats {
                term_id,
                doc_freq: 0,
                collection_freq: 0,
                max_tf: 0,
            }
        });

        stats.doc_freq += 1;
        stats.collection_freq += u64::from(tf);
        stats.max_tf = stats.max_tf.max(tf.min(u32::from(u16::MAX)) as u16);
    }

    fn rebuild_block_max(&mut self) {
        self.block_max.clear();
        let terms = self.postings.keys().cloned().collect::<Vec<_>>();
        for term in terms {
            self.rebuild_block_max_for_term(&term);
        }
    }

    fn rebuild_block_max_for_terms<I>(&mut self, terms: I)
    where
        I: IntoIterator<Item = String>,
    {
        for term in terms {
            self.rebuild_block_max_for_term(&term);
        }
    }

    fn rebuild_block_max_for_term(&mut self, term: &str) {
        let Some(postings) = self.postings.get(term) else {
            self.block_max.remove(term);
            return;
        };

        // Persist TF-only block maxima. Query-time code converts these to scoring
        // upper bounds using current corpus statistics.
        let mut blocks = Vec::with_capacity(postings.len().div_ceil(BLOCK_SIZE));
        for chunk in postings.chunks(BLOCK_SIZE) {
            if chunk.is_empty() {
                continue;
            }
            let start_doc_id = chunk.first().map_or(0, |posting| posting.doc_id);
            let end_doc_id = chunk.last().map_or(0, |posting| posting.doc_id);
            let max_tf = chunk
                .iter()
                .map(|posting| posting.term_freq)
                .max()
                .unwrap_or(0);
            blocks.push(PostingBlock {
                start_doc_id,
                end_doc_id,
                max_impact: max_tf as f32,
            });
        }

        self.block_max.insert(term.to_owned(), blocks);
    }

    fn docs_for_terms(&self, terms: &[String], intersection: bool) -> HashSet<u32> {
        if terms.is_empty() {
            return HashSet::new();
        }

        let doc_sets = terms
            .iter()
            .filter_map(|term| self.postings.get(term))
            .map(|postings| {
                postings
                    .iter()
                    .filter_map(|posting| {
                        if self.tombstones.contains(posting.doc_id) {
                            None
                        } else {
                            Some(posting.doc_id)
                        }
                    })
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();

        if doc_sets.is_empty() {
            return HashSet::new();
        }

        if intersection {
            let mut iter = doc_sets.into_iter();
            let mut acc = iter.next().unwrap_or_default();
            for set in iter {
                acc = acc.intersection(&set).copied().collect();
            }
            acc.into_iter().collect()
        } else {
            let mut out = BTreeSet::new();
            for set in doc_sets {
                out.extend(set);
            }
            out.into_iter().collect()
        }
    }

    fn insert_preprocessed_batch(&mut self, docs: Vec<IngestDocument>) -> Result<(), CoreError> {
        if docs.is_empty() {
            return Ok(());
        }

        self.ensure_external_ids_are_unique(&docs)?;
        let prepared_docs = self.preprocess_documents(docs);
        let base_doc_id = self.doc_lengths.len() as u32;
        let mut touched_terms = self.uses_block_max().then(HashSet::<String>::new);

        for (offset, prepared) in prepared_docs.into_iter().enumerate() {
            let doc_id = doc_id_with_offset(base_doc_id, offset)?;
            self.apply_prepared_document(doc_id, prepared, touched_terms.as_mut());
        }

        if let Some(touched_terms) = touched_terms {
            self.rebuild_block_max_for_terms(touched_terms);
        }

        Ok(())
    }

    fn ensure_external_ids_are_unique(&self, docs: &[IngestDocument]) -> Result<(), CoreError> {
        let mut seen = HashSet::<&str>::with_capacity(docs.len());
        for doc in docs {
            let external_id = doc.external_id.as_str();
            if self.external_to_internal.contains_key(external_id) || !seen.insert(external_id) {
                return Err(CoreError::InvalidArgument(format!(
                    "duplicate external id: {external_id}"
                )));
            }
        }
        Ok(())
    }

    fn preprocess_documents(&self, docs: Vec<IngestDocument>) -> Vec<PreparedDocument> {
        if docs.len() < PARALLEL_INGEST_MIN_DOCS {
            return docs
                .into_iter()
                .map(|doc| preprocess_document_with_tokenizer(&self.tokenizer, doc))
                .collect();
        }

        let tokenizer_config = self.config.tokenizer.clone();
        docs.into_par_iter()
            .map_init(
                move || {
                    retrievalx_tokenize::TokenizerPipeline::new(tokenizer_config.clone())
                        .expect("tokenizer config validated at index construction")
                },
                |tokenizer, doc| preprocess_document_with_tokenizer(tokenizer, doc),
            )
            .collect()
    }

    fn apply_prepared_document(
        &mut self,
        doc_id: u32,
        prepared: PreparedDocument,
        mut touched_terms: Option<&mut HashSet<String>>,
    ) {
        for (term, positions) in prepared.term_positions {
            if let Some(tracked_terms) = touched_terms.as_deref_mut() {
                tracked_terms.insert(term.clone());
            }
            self.upsert_posting(&term, doc_id, positions);
        }

        let record = DocumentRecord {
            external_id: prepared.external_id.clone(),
            text: prepared.text,
            fields: prepared.fields,
            token_counts: prepared.token_counts,
        };

        self.doc_lengths.push(prepared.doc_len);
        self.docs.push(Some(record));
        self.external_to_internal
            .insert(prepared.external_id, doc_id);
        self.total_live_doc_len += u64::from(prepared.doc_len);
        self.live_doc_count += 1;
    }

    fn uses_block_max(&self) -> bool {
        matches!(
            &self.config.retrieval,
            RetrievalStrategy::BlockMaxWand { .. }
        )
    }
}

#[derive(Debug, Clone)]
struct TermScoringContext {
    query_weight: f32,
    max_tf: f32,
    robertson_idf: f32,
    atire_idf: f32,
    tfidf_idf: f32,
    adaptive_k1: f32,
    term_k1: Option<f32>,
}

fn preprocess_document_with_tokenizer(
    tokenizer: &retrievalx_tokenize::TokenizerPipeline,
    doc: IngestDocument,
) -> PreparedDocument {
    let mut combined = doc.text.clone();
    for value in doc.fields.values() {
        combined.push(' ');
        combined.push_str(value);
    }

    let tokens = tokenizer.tokenize(&combined);
    let doc_len = tokens.len() as u32;
    let mut term_positions: HashMap<String, Vec<u32>> = HashMap::new();
    for (pos, token) in tokens.into_iter().enumerate() {
        term_positions.entry(token).or_default().push(pos as u32);
    }

    let mut token_counts = HashMap::with_capacity(term_positions.len());
    for (term, positions) in &term_positions {
        token_counts.insert(term.clone(), positions.len() as u32);
    }

    PreparedDocument {
        external_id: doc.external_id,
        text: doc.text,
        fields: doc.fields,
        doc_len,
        token_counts,
        term_positions: term_positions.into_iter().collect(),
    }
}

fn doc_id_with_offset(base_doc_id: u32, offset: usize) -> Result<u32, CoreError> {
    let offset = u32::try_from(offset).map_err(|_| {
        CoreError::InvalidArgument("document count exceeds u32 doc_id space".to_owned())
    })?;
    base_doc_id.checked_add(offset).ok_or_else(|| {
        CoreError::InvalidArgument("document count exceeds u32 doc_id space".to_owned())
    })
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
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

fn bm25f_field_factor(variant: &ScoringVariant) -> f32 {
    match variant {
        ScoringVariant::F { fields, .. } if fields.is_empty() => 1.0,
        ScoringVariant::F { b, fields, .. } => {
            fields
                .iter()
                .map(|field| field.weight * (1.0 - field.b + *b * field.b))
                .sum::<f32>()
                / fields.len() as f32
        }
        _ => 1.0,
    }
}

fn score_term_for_variant(
    variant: &ScoringVariant,
    context: &TermScoringContext,
    tf: f32,
    doc_len: f32,
    avgdl: f32,
    field_factor: f32,
) -> f32 {
    let avgdl = avgdl.max(1e-6);
    let tf = tf.max(1.0);

    match variant {
        ScoringVariant::Okapi { k1, b } => {
            let norm = *k1 * (1.0 - *b + *b * (doc_len / avgdl));
            let tf_component = (tf * (*k1 + 1.0)) / (tf + norm);
            context.robertson_idf * tf_component * context.query_weight
        }
        ScoringVariant::Plus { k1, b, delta } => {
            let norm = *k1 * (1.0 - *b + *b * (doc_len / avgdl));
            let tf_component = ((tf * (*k1 + 1.0)) / (tf + norm)) + *delta;
            context.robertson_idf * tf_component * context.query_weight
        }
        ScoringVariant::L { k1, b, c } => {
            let ctd = tf / (1.0 - *b + *b * (doc_len / avgdl));
            let tf_component = ((*k1 + 1.0) * (ctd + *c)) / (*k1 + ctd + *c);
            context.robertson_idf * tf_component * context.query_weight
        }
        ScoringVariant::Adpt { b } => {
            let k1 = context.adaptive_k1;
            let norm = k1 * (1.0 - *b + *b * (doc_len / avgdl));
            let tf_component = (tf * (k1 + 1.0)) / (tf + norm);
            context.robertson_idf * tf_component * context.query_weight
        }
        ScoringVariant::F { k1, b, .. } => {
            let tf = tf * field_factor.max(0.1);
            let norm = *k1 * (1.0 - *b + *b * (doc_len / avgdl));
            let tf_component = (tf * (*k1 + 1.0)) / (tf + norm);
            context.robertson_idf * tf_component * context.query_weight
        }
        ScoringVariant::T { default_k1, b, .. } => {
            let k1 = context.term_k1.unwrap_or(*default_k1).clamp(0.1, 4.0);
            let norm = k1 * (1.0 - *b + *b * (doc_len / avgdl));
            let tf_component = (tf * (k1 + 1.0)) / (tf + norm);
            context.robertson_idf * tf_component * context.query_weight
        }
        ScoringVariant::Atire { k1, b } => {
            let norm = *k1 * (1.0 - *b + *b * (doc_len / avgdl));
            let tf_component = (tf * (*k1 + 1.0)) / (tf + norm);
            context.atire_idf * tf_component * context.query_weight
        }
        ScoringVariant::TfIdf => tf * context.tfidf_idf * context.query_weight,
    }
}

fn score_upper_bound_for_tf(
    variant: &ScoringVariant,
    context: &TermScoringContext,
    tf_bound: f32,
    avgdl: f32,
    field_factor: f32,
) -> f32 {
    score_term_for_variant(
        variant,
        context,
        tf_bound.max(1.0),
        0.0,
        avgdl,
        field_factor,
    )
    .max(0.0)
}

fn positions_for_doc(postings: &[Posting], doc_id: u32) -> Option<&[u32]> {
    let idx = postings
        .binary_search_by_key(&doc_id, |posting| posting.doc_id)
        .ok()?;
    Some(postings[idx].positions.as_slice())
}

fn within_window(term_positions: &[&[u32]], window: u32) -> bool {
    if term_positions.is_empty() {
        return false;
    }

    for &start in term_positions[0] {
        let mut prev = start;
        let mut matched = true;

        for positions in term_positions.iter().skip(1) {
            let mut found = false;
            for pos in positions.iter().copied() {
                if pos >= prev && pos.saturating_sub(prev) <= window {
                    prev = pos;
                    found = true;
                    break;
                }
            }
            if !found {
                matched = false;
                break;
            }
        }

        if matched {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserts_and_searches() {
        let mut index = InvertedIndex::new(BM25Config::default()).expect("index init");
        index
            .insert_batch(vec![
                "rust language fast".to_owned(),
                "python language batteries".to_owned(),
                "rust python interoperability".to_owned(),
            ])
            .expect("batch insert");

        let results = index.search("rust", 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn deletes_and_compacts() {
        let mut index = InvertedIndex::new(BM25Config::default()).expect("index init");
        index
            .insert_batch(vec!["doc one".to_owned(), "doc two".to_owned()])
            .expect("batch insert");

        index.delete_by_doc_id(0).expect("delete should work");
        assert_eq!(index.stats().num_live_docs, 1);

        index.compact().expect("compact should work");
        assert_eq!(index.stats().num_docs, 1);
        assert_eq!(index.stats().num_live_docs, 1);
    }

    #[test]
    fn phrase_search_filters_to_exact_window() {
        let mut index = InvertedIndex::new(BM25Config::default()).expect("index init");
        index
            .insert_batch(vec![
                "rust language guide".to_owned(),
                "guide language rust".to_owned(),
                "rust memory safe language".to_owned(),
            ])
            .expect("batch insert");

        let terms = vec!["rust".to_owned(), "language".to_owned()];
        let hits = index.search_phrase(&terms, 1, 10);
        let ids = hits
            .into_iter()
            .map(|hit| hit.doc_id)
            .collect::<HashSet<_>>();

        assert!(ids.contains(&0));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn sparse_vectors_are_available_for_docs_and_queries() {
        let mut index = InvertedIndex::new(BM25Config::default()).expect("index init");
        index
            .insert_batch(vec!["rust retrieval rust".to_owned()])
            .expect("batch insert");

        let query_vec = index.sparse_vector_for_query("rust retrieval");
        assert!(!query_vec.is_empty());

        let doc_vec = index
            .sparse_vector_for_document(0)
            .expect("doc vector should exist");
        assert!(!doc_vec.is_empty());
    }

    #[test]
    fn skips_block_max_build_for_non_blockmax_strategy() {
        let config = BM25Config {
            retrieval: RetrievalStrategy::ExhaustiveDAAT,
            ..BM25Config::default()
        };
        let mut index = InvertedIndex::new(config).expect("index init");
        index
            .insert_batch(vec![
                "rust language guide".to_owned(),
                "python retrieval guide".to_owned(),
            ])
            .expect("batch insert");

        assert!(index.block_max.is_empty());
    }

    #[test]
    fn builds_block_max_for_blockmax_strategy() {
        let config = BM25Config {
            retrieval: RetrievalStrategy::BlockMaxWand { top_k_budget: 1000 },
            ..BM25Config::default()
        };
        let mut index = InvertedIndex::new(config).expect("index init");
        index
            .insert_batch(vec![
                "rust language guide".to_owned(),
                "python retrieval guide".to_owned(),
            ])
            .expect("batch insert");

        assert!(!index.block_max.is_empty());
    }
}
