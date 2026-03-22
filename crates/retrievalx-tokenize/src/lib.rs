#![forbid(unsafe_code)]

pub mod stemmers;
pub mod stopwords;
pub mod tokenizers;

use std::collections::{HashMap, HashSet};

use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use unicode_segmentation::UnicodeSegmentation;

use crate::stemmers::{build_stemmer, Stemmer};
use crate::stopwords::language_stopwords;

#[derive(Debug, Clone, Error)]
pub enum TokenizeError {
    #[error("invalid regex pattern: {0}")]
    InvalidRegex(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NgramMode {
    Character,
    Word,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenizerKind {
    Whitespace,
    Regex {
        pattern: String,
    },
    UnicodeWord,
    Ngram {
        min_n: usize,
        max_n: usize,
        mode: NgramMode,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FilterKind {
    Lowercase,
    Stopwords { lang: String },
    Length { min_len: usize, max_len: usize },
    DuplicateCap { max_per_doc: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StemmerConfig {
    NoOp,
    Porter,
    Snowball { lang: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerConfig {
    pub tokenizer: TokenizerKind,
    pub filters: Vec<FilterKind>,
    pub stemmer: StemmerConfig,
    pub min_token_len: usize,
    pub max_token_len: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            tokenizer: TokenizerKind::Whitespace,
            filters: vec![
                FilterKind::Lowercase,
                FilterKind::Stopwords {
                    lang: "en".to_owned(),
                },
            ],
            stemmer: StemmerConfig::NoOp,
            min_token_len: 1,
            max_token_len: 128,
        }
    }
}

#[derive(Debug)]
pub struct TokenizerPipeline {
    config: TokenizerConfig,
    compiled_regex: Option<Regex>,
    stemmer: Stemmer,
}

impl TokenizerPipeline {
    pub fn new(config: TokenizerConfig) -> Result<Self, TokenizeError> {
        let compiled_regex = match &config.tokenizer {
            TokenizerKind::Regex { pattern } => Some(
                Regex::new(pattern).map_err(|_| TokenizeError::InvalidRegex(pattern.clone()))?,
            ),
            _ => None,
        };

        let stemmer = build_stemmer(&config.stemmer);

        Ok(Self {
            config,
            compiled_regex,
            stemmer,
        })
    }

    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = self.base_tokenize(text);
        for filter in &self.config.filters {
            tokens = self.apply_filter(tokens, filter);
        }

        let min_len = self.config.min_token_len;
        let max_len = self.config.max_token_len;

        tokens
            .into_iter()
            .filter(|token| {
                let len = token.chars().count();
                len >= min_len && len <= max_len
            })
            .map(|token| self.stemmer.stem(&token))
            .collect()
    }

    fn base_tokenize(&self, text: &str) -> Vec<String> {
        match &self.config.tokenizer {
            TokenizerKind::Whitespace => text
                .split_whitespace()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            TokenizerKind::Regex { .. } => {
                if let Some(regex) = &self.compiled_regex {
                    regex
                        .find_iter(text)
                        .map(|m| m.as_str().to_owned())
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                }
            }
            TokenizerKind::UnicodeWord => UnicodeSegmentation::unicode_words(text)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            TokenizerKind::Ngram { min_n, max_n, mode } => match mode {
                NgramMode::Character => char_ngrams(text, *min_n, *max_n),
                NgramMode::Word => {
                    let words: Vec<&str> = UnicodeSegmentation::unicode_words(text).collect();
                    word_ngrams(&words, *min_n, *max_n)
                }
            },
        }
    }

    fn apply_filter(&self, tokens: Vec<String>, filter: &FilterKind) -> Vec<String> {
        match filter {
            FilterKind::Lowercase => tokens
                .into_iter()
                .map(|token| token.to_lowercase())
                .collect::<Vec<_>>(),
            FilterKind::Stopwords { lang } => {
                let stopwords = language_stopwords(lang);
                tokens
                    .into_iter()
                    .filter(|token| !stopwords.contains(token.as_str()))
                    .collect::<Vec<_>>()
            }
            FilterKind::Length { min_len, max_len } => tokens
                .into_iter()
                .filter(|token| {
                    let len = token.chars().count();
                    len >= *min_len && len <= *max_len
                })
                .collect::<Vec<_>>(),
            FilterKind::DuplicateCap { max_per_doc } => {
                let mut counts: HashMap<String, usize> = HashMap::new();
                let mut output = Vec::with_capacity(tokens.len());
                for token in tokens {
                    let count = counts.entry(token.clone()).or_insert(0);
                    if *count < *max_per_doc {
                        output.push(token);
                    }
                    *count += 1;
                }
                output
            }
        }
    }
}

fn char_ngrams(text: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let graphemes = text.graphemes(true).collect::<Vec<_>>();
    if graphemes.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for n in min_n..=max_n {
        if n == 0 || n > graphemes.len() {
            continue;
        }
        for window in graphemes.windows(n) {
            out.push(window.join(""));
        }
    }
    out
}

fn word_ngrams(words: &[&str], min_n: usize, max_n: usize) -> Vec<String> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for n in min_n..=max_n {
        if n == 0 || n > words.len() {
            continue;
        }
        for window in words.windows(n) {
            out.push(window.join(" "));
        }
    }
    out
}

pub fn supported_stopword_languages() -> HashSet<&'static str> {
    stopwords::supported_languages()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_unicode_words() {
        let config = TokenizerConfig {
            tokenizer: TokenizerKind::UnicodeWord,
            filters: vec![FilterKind::Lowercase],
            stemmer: StemmerConfig::NoOp,
            min_token_len: 1,
            max_token_len: 32,
        };

        let pipeline = TokenizerPipeline::new(config).expect("pipeline should build");
        let tokens = pipeline.tokenize("Rust भाषा BM25");
        assert!(tokens.contains(&"rust".to_owned()));
        assert!(tokens.contains(&"भाषा".to_owned()));
    }
}
