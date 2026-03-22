#![forbid(unsafe_code)]

use std::collections::HashSet;

pub fn ndcg_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f32 {
    if k == 0 || ranked.is_empty() || relevant.is_empty() {
        return 0.0;
    }

    let dcg = ranked
        .iter()
        .take(k)
        .enumerate()
        .filter_map(|(idx, doc_id)| {
            if relevant.contains(doc_id) {
                Some(1.0 / ((idx + 2) as f32).log2())
            } else {
                None
            }
        })
        .sum::<f32>();

    let ideal_hits = relevant.len().min(k);
    let idcg = (0..ideal_hits)
        .map(|idx| 1.0 / ((idx + 2) as f32).log2())
        .sum::<f32>();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

pub fn average_precision_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f32 {
    if k == 0 || relevant.is_empty() {
        return 0.0;
    }

    let mut hits = 0_usize;
    let mut sum_precision = 0.0_f32;

    for (idx, doc_id) in ranked.iter().take(k).enumerate() {
        if relevant.contains(doc_id) {
            hits += 1;
            sum_precision += hits as f32 / (idx + 1) as f32;
        }
    }

    if hits == 0 {
        0.0
    } else {
        sum_precision / relevant.len().min(k) as f32
    }
}

pub fn recall_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f32 {
    if relevant.is_empty() {
        return 0.0;
    }

    let hits = ranked
        .iter()
        .take(k)
        .filter(|doc_id| relevant.contains(*doc_id))
        .count();

    hits as f32 / relevant.len() as f32
}

pub fn precision_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }

    let hits = ranked
        .iter()
        .take(k)
        .filter(|doc_id| relevant.contains(*doc_id))
        .count();

    hits as f32 / k as f32
}

pub fn mrr(ranked: &[String], relevant: &HashSet<String>) -> f32 {
    for (idx, doc_id) in ranked.iter().enumerate() {
        if relevant.contains(doc_id) {
            return 1.0 / (idx + 1) as f32;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_is_one_for_perfect_retrieval() {
        let ranked = vec!["d1".to_owned(), "d2".to_owned(), "d3".to_owned()];
        let relevant = ["d1".to_owned(), "d2".to_owned()]
            .into_iter()
            .collect::<HashSet<_>>();

        let recall = recall_at_k(&ranked, &relevant, 2);
        assert!((recall - 1.0).abs() < f32::EPSILON);
    }
}
