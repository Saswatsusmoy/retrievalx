#![forbid(unsafe_code)]

use std::fs::File;
use std::io::Write;
use std::path::Path;

use serde::Serialize;

use retrievalx_core::index::InvertedIndex;

use crate::PersistError;

pub fn sparse_vector_for_query(index: &InvertedIndex, query: &str) -> Vec<(u32, f32)> {
    index.sparse_vector_for_query(query)
}

pub fn sparse_vector_for_document(index: &InvertedIndex, doc_id: u32) -> Vec<(u32, f32)> {
    index.sparse_vector_for_document(doc_id).unwrap_or_default()
}

pub fn export_beir_corpus_jsonl<P: AsRef<Path>>(
    index: &InvertedIndex,
    path: P,
) -> Result<(), PersistError> {
    let mut file = File::create(path)?;
    for (_, doc) in index.live_documents_iter() {
        let row = BeirCorpusDocument {
            id: doc.external_id.clone(),
            title: doc.fields.get("title").cloned(),
            text: doc.text.clone(),
        };
        let line = serde_json::to_string(&row)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
    }
    file.flush()?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct BeirCorpusDocument {
    #[serde(rename = "_id")]
    id: String,
    title: Option<String>,
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrievalx_core::{BM25Config, InvertedIndex};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn query_and_document_sparse_vectors_are_term_id_based() {
        let mut index = InvertedIndex::new(BM25Config::default()).expect("index init");
        index
            .insert_batch(vec!["rust retrieval rust".to_owned()])
            .expect("insert should work");

        let qvec = sparse_vector_for_query(&index, "rust retrieval");
        let dvec = sparse_vector_for_document(&index, 0);

        assert!(!qvec.is_empty());
        assert!(!dvec.is_empty());
        assert!(qvec.iter().all(|(term_id, _)| *term_id < 10_000));
        assert!(dvec.iter().all(|(term_id, _)| *term_id < 10_000));
    }

    #[test]
    fn exports_beir_jsonl() {
        let mut index = InvertedIndex::new(BM25Config::default()).expect("index init");
        index
            .insert_document(
                Some("doc-1".to_owned()),
                "retrieval text".to_owned(),
                std::collections::HashMap::from([("title".to_owned(), "sample".to_owned())]),
            )
            .expect("insert should work");

        let file_name = format!(
            "retrievalx-export-{}.jsonl",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0_u128, |duration| duration.as_nanos())
        );
        let out_path = std::env::temp_dir().join(file_name);
        export_beir_corpus_jsonl(&index, &out_path).expect("export should work");

        let bytes = std::fs::read(&out_path).expect("read should work");
        let text = String::from_utf8(bytes).expect("utf8 should be valid");
        assert!(text.contains("\"_id\":\"doc-1\""));
        assert!(text.contains("\"text\":\"retrieval text\""));

        let _ = std::fs::remove_file(out_path);
    }
}
