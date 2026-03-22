#![forbid(unsafe_code)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] retrievalx_tokenize::TokenizeError),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("document id not found: {0}")]
    DocumentNotFound(u32),
    #[error("document external id not found: {0}")]
    ExternalIdNotFound(String),
}
