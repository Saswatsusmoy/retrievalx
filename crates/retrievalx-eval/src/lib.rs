#![forbid(unsafe_code)]

pub mod beir;
pub mod metrics;
pub mod profiler;

pub use beir::{
    dataset_url, known_datasets, run_benchmark, run_benchmark_on_dataset, run_recall_degradation,
    run_recall_degradation_on_dataset, run_variant_comparison, run_variant_comparison_on_dataset,
    BeirAggregateMetrics, BeirCache, BeirDataset, BeirLoader, BeirQrel, BeirQuery,
    BeirQueryMetrics, BeirRunReport, RecallDegradationReport, RecallDegradationRow,
    VariantComparisonReport, VariantComparisonRow,
};
pub use metrics::{average_precision_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k};
pub use profiler::{LatencyBreakdown, LatencyProfiler};
