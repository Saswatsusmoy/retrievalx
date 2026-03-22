#![forbid(unsafe_code)]

use rust_stemmers::{Algorithm, Stemmer as RustStemmer};

use crate::StemmerConfig;

pub enum Stemmer {
    NoOp,
    Porter(RustStemmer),
    Snowball(RustStemmer),
}

impl std::fmt::Debug for Stemmer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoOp => write!(f, "NoOp"),
            Self::Porter(_) => write!(f, "Porter"),
            Self::Snowball(_) => write!(f, "Snowball"),
        }
    }
}

impl Stemmer {
    pub fn stem(&self, token: &str) -> String {
        match self {
            Self::NoOp => token.to_owned(),
            Self::Porter(stemmer) | Self::Snowball(stemmer) => stemmer.stem(token).into_owned(),
        }
    }
}

pub fn build_stemmer(config: &StemmerConfig) -> Stemmer {
    match config {
        StemmerConfig::NoOp => Stemmer::NoOp,
        StemmerConfig::Porter => Stemmer::Porter(RustStemmer::create(Algorithm::English)),
        StemmerConfig::Snowball { lang } => {
            let alg = language_to_algorithm(lang).unwrap_or(Algorithm::English);
            Stemmer::Snowball(RustStemmer::create(alg))
        }
    }
}

fn language_to_algorithm(lang: &str) -> Option<Algorithm> {
    match lang {
        "ar" => Some(Algorithm::Arabic),
        "da" => Some(Algorithm::Danish),
        "nl" => Some(Algorithm::Dutch),
        "en" => Some(Algorithm::English),
        "fi" => Some(Algorithm::Finnish),
        "fr" => Some(Algorithm::French),
        "de" => Some(Algorithm::German),
        "hu" => Some(Algorithm::Hungarian),
        "it" => Some(Algorithm::Italian),
        "no" => Some(Algorithm::Norwegian),
        "pt" => Some(Algorithm::Portuguese),
        "ro" => Some(Algorithm::Romanian),
        "ru" => Some(Algorithm::Russian),
        "es" => Some(Algorithm::Spanish),
        "sv" => Some(Algorithm::Swedish),
        "ta" => Some(Algorithm::Tamil),
        "tr" => Some(Algorithm::Turkish),
        _ => None,
    }
}
