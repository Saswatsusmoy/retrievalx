#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RetrievalStrategy {
    ExhaustiveDAAT,
    ExhaustiveTAAT,
    Wand { top_k_budget: usize },
    BlockMaxWand { top_k_budget: usize },
    MaxScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PostingScoreInput {
    pub doc_id: u32,
    pub tf: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TermPostingList {
    pub term: String,
    pub postings: Vec<PostingScoreInput>,
    pub upper_bound: f32,
    pub block_upper_bounds: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalDiagnostics {
    pub scored_docs: usize,
    pub scored_postings: usize,
    pub skipped_postings: usize,
    pub pruned_candidates: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RetrievalResult {
    pub hits: Vec<(u32, f32)>,
    pub diagnostics: RetrievalDiagnostics,
}

pub trait Retriever: Send + Sync {
    fn rank(
        &self,
        postings: &[TermPostingList],
        top_k: usize,
        score_fn: &mut dyn FnMut(usize, u32, u32) -> f32,
    ) -> RetrievalResult;
}

impl RetrievalStrategy {
    pub fn build(&self) -> Box<dyn Retriever> {
        match self {
            Self::ExhaustiveDAAT => Box::new(ExhaustiveDaatRetriever),
            Self::ExhaustiveTAAT => Box::new(ExhaustiveTaatRetriever),
            Self::Wand { top_k_budget } => Box::new(WandRetriever {
                use_block_max: false,
                top_k_budget: normalize_budget(*top_k_budget),
            }),
            Self::BlockMaxWand { top_k_budget } => Box::new(WandRetriever {
                use_block_max: true,
                top_k_budget: normalize_budget(*top_k_budget),
            }),
            Self::MaxScore => Box::new(MaxScoreRetriever),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExhaustiveDaatRetriever;

impl Retriever for ExhaustiveDaatRetriever {
    fn rank(
        &self,
        postings: &[TermPostingList],
        top_k: usize,
        score_fn: &mut dyn FnMut(usize, u32, u32) -> f32,
    ) -> RetrievalResult {
        if top_k == 0 || postings.is_empty() {
            return RetrievalResult::default();
        }

        let mut cursors = build_cursors(postings);
        let mut heap = TopKHeap::new(top_k);
        let mut diag = RetrievalDiagnostics::default();

        while let Some(candidate_doc) = min_doc(&cursors) {
            let mut score = 0.0_f32;
            let mut matched = false;

            for (idx, cursor) in cursors.iter_mut().enumerate() {
                if cursor.current_doc() == Some(candidate_doc) {
                    let tf = cursor.current_tf().unwrap_or(0);
                    score += score_fn(idx, candidate_doc, tf);
                    diag.scored_postings += 1;
                    cursor.advance();
                    matched = true;
                }
            }

            if matched {
                diag.scored_docs += 1;
                heap.push(candidate_doc, score);
            }
        }

        RetrievalResult {
            hits: heap.into_sorted_vec(),
            diagnostics: diag,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExhaustiveTaatRetriever;

impl Retriever for ExhaustiveTaatRetriever {
    fn rank(
        &self,
        postings: &[TermPostingList],
        top_k: usize,
        score_fn: &mut dyn FnMut(usize, u32, u32) -> f32,
    ) -> RetrievalResult {
        if top_k == 0 || postings.is_empty() {
            return RetrievalResult::default();
        }

        const DENSE_ACCUM_MAX_DOCS: usize = 5_000_000;

        let mut diag = RetrievalDiagnostics::default();

        let max_doc_id = postings
            .iter()
            .filter_map(|term| term.postings.last().map(|posting| posting.doc_id as usize))
            .max()
            .unwrap_or(0);

        if max_doc_id.saturating_add(1) <= DENSE_ACCUM_MAX_DOCS {
            let mut accum = vec![0.0_f32; max_doc_id.saturating_add(1)];
            let mut seen = vec![false; max_doc_id.saturating_add(1)];
            let mut touched = Vec::<u32>::new();

            for (idx, term) in postings.iter().enumerate() {
                for posting in &term.postings {
                    let score = score_fn(idx, posting.doc_id, posting.tf);
                    if score <= 0.0 {
                        continue;
                    }

                    let doc_idx = posting.doc_id as usize;
                    if !seen[doc_idx] {
                        seen[doc_idx] = true;
                        touched.push(posting.doc_id);
                    }
                    accum[doc_idx] += score;
                    diag.scored_postings += 1;
                }
            }

            diag.scored_docs = touched.len();

            let mut heap = TopKHeap::new(top_k);
            for doc_id in touched {
                heap.push(doc_id, accum[doc_id as usize]);
            }

            return RetrievalResult {
                hits: heap.into_sorted_vec(),
                diagnostics: diag,
            };
        }

        let estimated_capacity = postings
            .iter()
            .map(|term| term.postings.len())
            .sum::<usize>()
            .min(1_000_000);
        let mut accum = HashMap::<u32, f32>::with_capacity(estimated_capacity);

        for (idx, term) in postings.iter().enumerate() {
            for posting in &term.postings {
                let score = score_fn(idx, posting.doc_id, posting.tf);
                if score > 0.0 {
                    *accum.entry(posting.doc_id).or_insert(0.0) += score;
                    diag.scored_postings += 1;
                }
            }
        }

        diag.scored_docs = accum.len();

        let mut heap = TopKHeap::new(top_k);
        for (doc_id, score) in accum {
            heap.push(doc_id, score);
        }

        RetrievalResult {
            hits: heap.into_sorted_vec(),
            diagnostics: diag,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WandRetriever {
    use_block_max: bool,
    top_k_budget: usize,
}

impl Retriever for WandRetriever {
    fn rank(
        &self,
        postings: &[TermPostingList],
        top_k: usize,
        score_fn: &mut dyn FnMut(usize, u32, u32) -> f32,
    ) -> RetrievalResult {
        if top_k == 0 || postings.is_empty() {
            return RetrievalResult::default();
        }

        let mut cursors = build_cursors(postings);
        let mut heap = TopKHeap::new(top_k);
        let mut diag = RetrievalDiagnostics::default();
        let mut active: Vec<usize> = Vec::with_capacity(cursors.len());

        loop {
            active.clear();
            active.extend(
                cursors
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, cursor)| cursor.current_doc().map(|_| idx)),
            );
            if active.is_empty() {
                break;
            }

            active.sort_by_key(|idx| cursors[*idx].current_doc().unwrap_or(u32::MAX));

            let threshold = heap.threshold();
            let mut ub_acc = 0.0_f32;
            let mut pivot_doc: Option<u32> = None;

            for &idx in &active {
                ub_acc += if self.use_block_max {
                    cursors[idx].current_block_upper_bound()
                } else {
                    cursors[idx].upper_bound
                };
                if ub_acc > threshold {
                    pivot_doc = cursors[idx].current_doc();
                    break;
                }
            }

            let Some(pivot_doc) = pivot_doc else {
                break;
            };

            let first_doc = cursors[active[0]].current_doc().unwrap_or(u32::MAX);

            if first_doc == pivot_doc {
                let mut score = 0.0_f32;
                let mut matched = false;

                for &idx in &active {
                    if cursors[idx].current_doc() == Some(pivot_doc) {
                        let tf = cursors[idx].current_tf().unwrap_or(0);
                        score += score_fn(idx, pivot_doc, tf);
                        diag.scored_postings += 1;
                        cursors[idx].advance();
                        matched = true;
                    } else {
                        break;
                    }
                }

                if matched {
                    diag.scored_docs += 1;
                    if score <= threshold {
                        diag.pruned_candidates += 1;
                    }
                    heap.push(pivot_doc, score);
                    if diag.scored_docs >= self.top_k_budget {
                        break;
                    }
                }
            } else {
                for &idx in &active {
                    let Some(doc_id) = cursors[idx].current_doc() else {
                        continue;
                    };
                    if doc_id < pivot_doc {
                        let skipped = cursors[idx].advance_to(pivot_doc);
                        diag.skipped_postings += skipped;
                    } else {
                        break;
                    }
                }
            }
        }

        RetrievalResult {
            hits: heap.into_sorted_vec(),
            diagnostics: diag,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaxScoreRetriever;

impl Retriever for MaxScoreRetriever {
    fn rank(
        &self,
        postings: &[TermPostingList],
        top_k: usize,
        score_fn: &mut dyn FnMut(usize, u32, u32) -> f32,
    ) -> RetrievalResult {
        if top_k == 0 || postings.is_empty() {
            return RetrievalResult::default();
        }

        let mut cursors = build_cursors(postings);
        let mut sorted_terms: Vec<usize> = (0..postings.len()).collect();
        sorted_terms.sort_by(|a, b| {
            postings[*b]
                .upper_bound
                .total_cmp(&postings[*a].upper_bound)
        });

        let total_ub = postings.iter().map(|term| term.upper_bound).sum::<f32>();

        let mut heap = TopKHeap::new(top_k);
        let mut diag = RetrievalDiagnostics::default();

        loop {
            if min_doc(&cursors).is_none() {
                break;
            }

            let threshold = heap.threshold();
            let mut tail_sum = total_ub;
            let mut essential_len = sorted_terms.len();

            for (pos, term_idx) in sorted_terms.iter().enumerate() {
                tail_sum -= postings[*term_idx].upper_bound;
                if tail_sum <= threshold {
                    essential_len = pos + 1;
                    break;
                }
            }

            let essential = &sorted_terms[..essential_len];
            let non_essential = &sorted_terms[essential_len..];

            let candidate_doc = essential
                .iter()
                .filter_map(|idx| cursors[*idx].current_doc())
                .min();
            let Some(candidate_doc) = candidate_doc else {
                break;
            };

            let mut score = 0.0_f32;
            let mut matched = false;

            for &idx in essential {
                let skipped = cursors[idx].advance_to(candidate_doc);
                diag.skipped_postings += skipped;

                if cursors[idx].current_doc() == Some(candidate_doc) {
                    let tf = cursors[idx].current_tf().unwrap_or(0);
                    score += score_fn(idx, candidate_doc, tf);
                    diag.scored_postings += 1;
                    cursors[idx].advance();
                    matched = true;
                }
            }

            if !matched {
                continue;
            }

            let non_essential_ub = non_essential
                .iter()
                .map(|idx| postings[*idx].upper_bound)
                .sum::<f32>();

            if score + non_essential_ub <= threshold {
                diag.pruned_candidates += 1;
                continue;
            }

            for &idx in non_essential {
                let skipped = cursors[idx].advance_to(candidate_doc);
                diag.skipped_postings += skipped;

                if cursors[idx].current_doc() == Some(candidate_doc) {
                    let tf = cursors[idx].current_tf().unwrap_or(0);
                    score += score_fn(idx, candidate_doc, tf);
                    diag.scored_postings += 1;
                    cursors[idx].advance();
                }
            }

            diag.scored_docs += 1;
            heap.push(candidate_doc, score);
        }

        RetrievalResult {
            hits: heap.into_sorted_vec(),
            diagnostics: diag,
        }
    }
}

#[derive(Debug, Clone)]
struct CursorState<'a> {
    postings: &'a [PostingScoreInput],
    upper_bound: f32,
    block_suffix_upper_bounds: Vec<f32>,
    pos: usize,
}

impl CursorState<'_> {
    fn current_doc(&self) -> Option<u32> {
        self.postings.get(self.pos).map(|entry| entry.doc_id)
    }

    fn current_tf(&self) -> Option<u32> {
        self.postings.get(self.pos).map(|entry| entry.tf)
    }

    fn current_block_upper_bound(&self) -> f32 {
        if self.postings.is_empty() || self.pos >= self.postings.len() {
            return 0.0;
        }

        let block_idx = self.pos / 64;
        self.block_suffix_upper_bounds
            .get(block_idx)
            .copied()
            .unwrap_or(self.upper_bound)
    }

    fn advance(&mut self) {
        if self.pos < self.postings.len() {
            self.pos += 1;
        }
    }

    fn advance_to(&mut self, target: u32) -> usize {
        if self.pos >= self.postings.len() {
            return 0;
        }

        let start = self.pos;
        let search = &self.postings[self.pos..];
        let offset = search.partition_point(|entry| entry.doc_id < target);
        self.pos = self.pos.saturating_add(offset);

        self.pos.saturating_sub(start)
    }
}

fn build_cursors(postings: &[TermPostingList]) -> Vec<CursorState<'_>> {
    postings
        .iter()
        .map(|term| CursorState {
            postings: &term.postings,
            upper_bound: term.upper_bound,
            block_suffix_upper_bounds: suffix_maxima(term.block_upper_bounds.as_slice()),
            pos: 0,
        })
        .collect()
}

fn suffix_maxima(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut suffix = vec![0.0_f32; values.len()];
    let mut running = 0.0_f32;
    for (idx, value) in values.iter().enumerate().rev() {
        running = running.max(*value);
        suffix[idx] = running;
    }
    suffix
}

fn min_doc(cursors: &[CursorState<'_>]) -> Option<u32> {
    cursors.iter().filter_map(CursorState::current_doc).min()
}

fn normalize_budget(top_k_budget: usize) -> usize {
    if top_k_budget == 0 {
        usize::MAX
    } else {
        top_k_budget
    }
}

#[derive(Debug, Clone, Copy)]
struct MinHeapItem {
    doc_id: u32,
    score: f32,
}

impl Eq for MinHeapItem {}

impl PartialEq for MinHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Ord for MinHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.score.total_cmp(&self.score) {
            Ordering::Equal => self.doc_id.cmp(&other.doc_id),
            ord => ord,
        }
    }
}

impl PartialOrd for MinHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Default)]
struct TopKHeap {
    top_k: usize,
    heap: BinaryHeap<MinHeapItem>,
}

impl TopKHeap {
    fn new(top_k: usize) -> Self {
        Self {
            top_k,
            heap: BinaryHeap::new(),
        }
    }

    fn threshold(&self) -> f32 {
        if self.heap.len() < self.top_k {
            0.0
        } else {
            self.heap.peek().map_or(0.0, |item| item.score)
        }
    }

    fn push(&mut self, doc_id: u32, score: f32) {
        if score <= 0.0 {
            return;
        }

        if self.heap.len() < self.top_k {
            self.heap.push(MinHeapItem { doc_id, score });
            return;
        }

        if let Some(min_item) = self.heap.peek() {
            if score > min_item.score {
                self.heap.pop();
                self.heap.push(MinHeapItem { doc_id, score });
            }
        }
    }

    fn into_sorted_vec(mut self) -> Vec<(u32, f32)> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(item) = self.heap.pop() {
            out.push((item.doc_id, item.score));
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_postings() -> Vec<TermPostingList> {
        vec![
            TermPostingList {
                term: "rust".to_owned(),
                postings: vec![
                    PostingScoreInput { doc_id: 1, tf: 2 },
                    PostingScoreInput { doc_id: 2, tf: 1 },
                    PostingScoreInput { doc_id: 4, tf: 3 },
                ],
                upper_bound: 3.0,
                block_upper_bounds: vec![3.0],
            },
            TermPostingList {
                term: "python".to_owned(),
                postings: vec![
                    PostingScoreInput { doc_id: 2, tf: 4 },
                    PostingScoreInput { doc_id: 3, tf: 1 },
                    PostingScoreInput { doc_id: 4, tf: 1 },
                ],
                upper_bound: 4.0,
                block_upper_bounds: vec![4.0],
            },
        ]
    }

    fn score(term_idx: usize, _doc_id: u32, tf: u32) -> f32 {
        tf as f32 * (term_idx as f32 + 1.0)
    }

    #[test]
    fn wand_matches_exhaustive_on_sample() {
        let postings = sample_postings();

        let exhaustive = ExhaustiveDaatRetriever.rank(&postings, 3, &mut score);
        let wand = WandRetriever {
            use_block_max: false,
            top_k_budget: usize::MAX,
        }
        .rank(&postings, 3, &mut score);

        assert_eq!(exhaustive.hits, wand.hits);
    }

    #[test]
    fn maxscore_matches_exhaustive_on_sample() {
        let postings = sample_postings();

        let exhaustive = ExhaustiveDaatRetriever.rank(&postings, 3, &mut score);
        let maxscore = MaxScoreRetriever.rank(&postings, 3, &mut score);

        assert_eq!(exhaustive.hits, maxscore.hits);
    }

    #[test]
    fn wand_emits_pruning_diagnostics() {
        let postings = vec![
            TermPostingList {
                term: "t1".to_owned(),
                postings: vec![
                    PostingScoreInput { doc_id: 1, tf: 1 },
                    PostingScoreInput { doc_id: 500, tf: 1 },
                    PostingScoreInput {
                        doc_id: 1000,
                        tf: 1,
                    },
                ],
                upper_bound: 1.0,
                block_upper_bounds: vec![1.0],
            },
            TermPostingList {
                term: "t2".to_owned(),
                postings: vec![
                    PostingScoreInput { doc_id: 2, tf: 1 },
                    PostingScoreInput { doc_id: 600, tf: 1 },
                    PostingScoreInput {
                        doc_id: 1001,
                        tf: 1,
                    },
                ],
                upper_bound: 1.0,
                block_upper_bounds: vec![1.0],
            },
        ];

        let result = WandRetriever {
            use_block_max: false,
            top_k_budget: usize::MAX,
        }
        .rank(&postings, 1, &mut score);

        assert!(
            result.diagnostics.skipped_postings > 0 || result.diagnostics.pruned_candidates > 0
        );
    }

    #[test]
    fn wand_respects_top_k_budget() {
        let postings = sample_postings();
        let result = WandRetriever {
            use_block_max: false,
            top_k_budget: 1,
        }
        .rank(&postings, 3, &mut score);

        assert!(result.diagnostics.scored_docs <= 1);
    }

    #[test]
    fn taat_matches_exhaustive_on_sample() {
        let postings = sample_postings();

        let exhaustive = ExhaustiveDaatRetriever.rank(&postings, 3, &mut score);
        let taat = ExhaustiveTaatRetriever.rank(&postings, 3, &mut score);

        assert_eq!(exhaustive.hits, taat.hits);
    }

    #[test]
    fn block_max_uses_suffix_bounds_for_correctness() {
        let mut t1 = Vec::new();
        for doc_id in 1..=64 {
            t1.push(PostingScoreInput { doc_id, tf: 1 });
        }
        t1.push(PostingScoreInput {
            doc_id: 1000,
            tf: 10,
        });

        let mut t2 = Vec::new();
        for doc_id in 1..=64 {
            t2.push(PostingScoreInput { doc_id, tf: 1 });
        }

        let postings = vec![
            TermPostingList {
                term: "t1".to_owned(),
                postings: t1,
                upper_bound: 10.0,
                block_upper_bounds: vec![1.0, 10.0],
            },
            TermPostingList {
                term: "t2".to_owned(),
                postings: t2,
                upper_bound: 1.0,
                block_upper_bounds: vec![1.0],
            },
        ];

        let exhaustive = ExhaustiveDaatRetriever.rank(&postings, 1, &mut score);
        let block_max = WandRetriever {
            use_block_max: true,
            top_k_budget: usize::MAX,
        }
        .rank(&postings, 1, &mut score);

        assert_eq!(exhaustive.hits, block_max.hits);
        assert_eq!(block_max.hits.first().map(|(doc, _)| *doc), Some(1000));
    }
}
