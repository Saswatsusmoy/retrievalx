#![forbid(unsafe_code)]

pub mod config;
pub mod error;
pub mod fusion;
pub mod index;
pub mod query;
pub mod retrieval;
pub mod scoring;

pub use config::{BM25Config, ExpansionConfig, FieldConfig, PersistConfig};
pub use error::CoreError;
pub use fusion::{
    cdf_normalize, linear_combination, min_max_normalize, reciprocal_rank_fusion,
    z_score_normalize, FusionConfig, FusionMethod, FusionResult, Normalizer,
};
pub use index::{DocumentRecord, IndexSnapshot, IndexStats, InvertedIndex, Posting, TermStats};
pub use query::{
    BagOfWordsQuery, BooleanClause, BooleanQuery, PhraseQuery, QueryExpander, QueryTerm,
    WeightedQuery,
};
pub use retrieval::{
    PostingScoreInput, RetrievalDiagnostics, RetrievalResult, RetrievalStrategy, TermPostingList,
};
pub use scoring::ScoringVariant;
