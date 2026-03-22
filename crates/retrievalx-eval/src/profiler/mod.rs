#![forbid(unsafe_code)]

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LatencyBreakdown {
    pub tokenization: Duration,
    pub postings_fetch: Duration,
    pub scoring: Duration,
    pub top_k_heap: Duration,
    pub total: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct LatencyProfiler {
    samples: Vec<LatencyBreakdown>,
}

impl LatencyProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn profile<F>(&mut self, f: F) -> LatencyBreakdown
    where
        F: FnOnce() -> LatencyBreakdown,
    {
        let start = Instant::now();
        let mut breakdown = f();
        breakdown.total = start.elapsed();
        self.samples.push(breakdown.clone());
        breakdown
    }

    pub fn record(&mut self, breakdown: LatencyBreakdown) {
        self.samples.push(breakdown);
    }

    pub fn p50(&self) -> Option<Duration> {
        percentile(&self.samples, 0.50)
    }

    pub fn p95(&self) -> Option<Duration> {
        percentile(&self.samples, 0.95)
    }

    pub fn p99(&self) -> Option<Duration> {
        percentile(&self.samples, 0.99)
    }

    pub fn p999(&self) -> Option<Duration> {
        percentile(&self.samples, 0.999)
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

fn percentile(samples: &[LatencyBreakdown], p: f32) -> Option<Duration> {
    if samples.is_empty() {
        return None;
    }

    let mut totals = samples
        .iter()
        .map(|sample| sample.total)
        .collect::<Vec<_>>();
    totals.sort();

    let idx = ((totals.len() as f32 - 1.0) * p).round() as usize;
    totals.get(idx).copied()
}
