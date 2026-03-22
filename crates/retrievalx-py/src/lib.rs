#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use retrievalx_core::fusion;
use retrievalx_core::query::{BooleanQuery, QueryTerm, WeightedQuery};
use retrievalx_core::{BM25Config, InvertedIndex};
use retrievalx_eval as eval;
use retrievalx_persist::wal::{WalOperation, WriteAheadLog};

#[pyclass(name = "NativeBM25Index")]
struct PyBM25Index {
    inner: Arc<RwLock<InvertedIndex>>,
    wal: Arc<RwLock<Option<WriteAheadLog>>>,
}

#[pyclass(name = "NativeLatencyProfiler")]
#[derive(Default)]
struct PyLatencyProfiler {
    samples_ms: Vec<f32>,
}

#[pymethods]
impl PyBM25Index {
    #[new]
    #[pyo3(signature = (config_json=None))]
    fn new(config_json: Option<String>) -> PyResult<Self> {
        let config = parse_config(config_json)?;
        let index = InvertedIndex::new(config).map_err(to_pyerr)?;
        Ok(Self::from_index(index))
    }

    #[staticmethod]
    #[pyo3(signature = (documents, config_json=None))]
    fn from_documents(documents: Vec<String>, config_json: Option<String>) -> PyResult<Self> {
        let config = parse_config(config_json)?;
        let index = InvertedIndex::from_documents(documents, config).map_err(to_pyerr)?;
        Ok(Self::from_index(index))
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let index = retrievalx_persist::load_index(path).map_err(to_pyerr)?;
        Ok(Self::from_index(index))
    }

    #[staticmethod]
    fn load_mmap(path: String) -> PyResult<Self> {
        let index = retrievalx_persist::load_index_mmap(path).map_err(to_pyerr)?;
        Ok(Self::from_index(index))
    }

    #[staticmethod]
    #[pyo3(signature = (path, wal_path, mode="in_memory".to_owned()))]
    fn load_with_wal(path: String, wal_path: String, mode: String) -> PyResult<Self> {
        let mut index = match mode.as_str() {
            "in_memory" => retrievalx_persist::load_index(path).map_err(to_pyerr)?,
            "mmap" => retrievalx_persist::load_index_mmap(path).map_err(to_pyerr)?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "mode must be one of: in_memory, mmap (got {other})"
                )));
            }
        };

        let wal = WriteAheadLog::open(wal_path);
        let _ = retrievalx_persist::replay_wal_into_index(&mut index, &wal).map_err(to_pyerr)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(index)),
            wal: Arc::new(RwLock::new(Some(wal))),
        })
    }

    fn save(&self, path: String) -> PyResult<()> {
        let index = self.inner.read();
        retrievalx_persist::save_index(&index, path).map_err(to_pyerr)
    }

    fn save_metadata(&self, path: String) -> PyResult<()> {
        let index = self.inner.read();
        retrievalx_persist::write_metadata_sidecar(&index, path).map_err(to_pyerr)
    }

    fn insert_batch(&self, documents: Vec<String>) -> PyResult<()> {
        if documents.is_empty() {
            return Ok(());
        }
        {
            let wal = self.wal.read();
            if let Some(wal) = wal.as_ref() {
                let ops = documents
                    .iter()
                    .map(|text| WalOperation::Insert {
                        external_id: None,
                        text: text.clone(),
                        fields: HashMap::new(),
                    })
                    .collect::<Vec<_>>();
                wal.append_many(ops.iter()).map_err(to_pyerr)?;
            }
        }

        let mut index = self.inner.write();
        index.insert_batch(documents).map_err(to_pyerr)
    }

    fn insert_documents(&self, documents: Vec<(String, String)>) -> PyResult<()> {
        if documents.is_empty() {
            return Ok(());
        }
        {
            let wal = self.wal.read();
            if let Some(wal) = wal.as_ref() {
                let ops = documents
                    .iter()
                    .map(|(external_id, text)| WalOperation::Insert {
                        external_id: Some(external_id.clone()),
                        text: text.clone(),
                        fields: HashMap::new(),
                    })
                    .collect::<Vec<_>>();
                wal.append_many(ops.iter()).map_err(to_pyerr)?;
            }
        }

        let mut index = self.inner.write();
        index.insert_documents_with_ids(documents).map_err(to_pyerr)
    }

    fn search(&self, query: String, top_k: usize) -> Vec<(String, f32)> {
        let index = self.inner.read();
        hits_to_tuples(index.search(&query, top_k))
    }

    fn search_weighted(&self, terms: Vec<(String, f32)>, top_k: usize) -> Vec<(String, f32)> {
        let weighted = WeightedQuery {
            terms: terms
                .into_iter()
                .map(|(term, weight)| QueryTerm { term, weight })
                .collect(),
        };

        let index = self.inner.read();
        hits_to_tuples(index.search_weighted(&weighted, top_k))
    }

    fn search_boolean(
        &self,
        must: Vec<String>,
        should: Vec<String>,
        must_not: Vec<String>,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let query = BooleanQuery {
            must,
            should,
            must_not,
        };

        let index = self.inner.read();
        hits_to_tuples(index.search_boolean(&query, top_k))
    }

    fn search_phrase(&self, terms: Vec<String>, window: u32, top_k: usize) -> Vec<(String, f32)> {
        let index = self.inner.read();
        hits_to_tuples(index.search_phrase(&terms, window, top_k))
    }

    fn delete_by_external_id(&self, external_id: String) -> PyResult<()> {
        {
            let wal = self.wal.read();
            if let Some(wal) = wal.as_ref() {
                wal.append(&WalOperation::Delete {
                    external_id: external_id.clone(),
                })
                .map_err(to_pyerr)?;
            }
        }
        let mut index = self.inner.write();
        index.delete_by_external_id(&external_id).map_err(to_pyerr)
    }

    fn compact(&self) -> PyResult<()> {
        let mut index = self.inner.write();
        index.compact().map_err(to_pyerr)
    }

    fn compact_and_flush(&self, index_path: String) -> PyResult<()> {
        let wal = self.wal.read().clone();
        let mut index = self.inner.write();
        if let Some(wal) = wal.as_ref() {
            retrievalx_persist::compact_index_with_wal(&mut index, index_path, wal)
                .map_err(to_pyerr)?;
        } else {
            index.compact().map_err(to_pyerr)?;
            retrievalx_persist::save_index(&index, index_path).map_err(to_pyerr)?;
        }
        Ok(())
    }

    fn enable_wal(&self, wal_path: String) {
        *self.wal.write() = Some(WriteAheadLog::open(wal_path));
    }

    fn disable_wal(&self) {
        *self.wal.write() = None;
    }

    fn replay_wal(&self, wal_path: String) -> PyResult<usize> {
        let wal = WriteAheadLog::open(wal_path);
        let mut index = self.inner.write();
        let applied =
            retrievalx_persist::replay_wal_into_index(&mut index, &wal).map_err(to_pyerr)?;
        Ok(applied)
    }

    fn sparse_vector_for_query(&self, query: String) -> Vec<(u32, f32)> {
        let index = self.inner.read();
        index.sparse_vector_for_query(&query)
    }

    fn sparse_vector_for_document(&self, doc_id: u32) -> Vec<(u32, f32)> {
        let index = self.inner.read();
        index.sparse_vector_for_document(doc_id).unwrap_or_default()
    }

    fn stats_json(&self) -> PyResult<String> {
        let index = self.inner.read();
        serde_json::to_string(&index.stats())
            .map_err(|error| PyValueError::new_err(format!("serialize stats failed: {error}")))
    }

    fn vocabulary<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let index = self.inner.read();
        let out = PyList::empty(py);
        for (term, stats) in index.vocabulary() {
            let tuple = (
                term.clone(),
                stats.term_id,
                stats.doc_freq,
                stats.collection_freq,
                stats.max_tf,
            );
            out.append(tuple)
                .expect("append should not fail for valid tuple");
        }
        out
    }

    fn doc_length(&self, doc_id: u32) -> Option<u32> {
        let index = self.inner.read();
        index.doc_length(doc_id)
    }

    fn postings(&self, term: String) -> Vec<(u32, u32)> {
        let index = self.inner.read();
        index
            .postings(&term)
            .unwrap_or(&[])
            .iter()
            .map(|posting| (posting.doc_id, posting.term_freq))
            .collect()
    }

    fn avgdl(&self) -> f32 {
        let index = self.inner.read();
        index.avgdl()
    }

    fn config_json(&self) -> PyResult<String> {
        let index = self.inner.read();
        serde_json::to_string(index.config())
            .map_err(|error| PyValueError::new_err(format!("serialize config failed: {error}")))
    }

    fn stats_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let index = self.inner.read();
        let stats = index.stats();
        let dict = PyDict::new(py);
        dict.set_item("num_docs", stats.num_docs)?;
        dict.set_item("num_live_docs", stats.num_live_docs)?;
        dict.set_item("vocabulary_size", stats.vocabulary_size)?;
        dict.set_item("avgdl", stats.avgdl)?;
        dict.set_item("tombstones", stats.tombstones)?;
        dict.set_item("total_terms", stats.total_terms)?;
        dict.set_item("build_timestamp_unix", stats.build_timestamp_unix)?;
        Ok(dict)
    }
}

#[pymethods]
impl PyLatencyProfiler {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn record(&mut self, total_ms: f32) {
        self.samples_ms.push(total_ms.max(0.0));
    }

    fn p50(&self) -> f32 {
        percentile_ms(&self.samples_ms, 0.50)
    }

    fn p95(&self) -> f32 {
        percentile_ms(&self.samples_ms, 0.95)
    }

    fn p99(&self) -> f32 {
        percentile_ms(&self.samples_ms, 0.99)
    }

    fn p999(&self) -> f32 {
        percentile_ms(&self.samples_ms, 0.999)
    }

    fn len(&self) -> usize {
        self.samples_ms.len()
    }
}

#[pyfunction]
#[pyo3(signature = (primary, secondary, k=60))]
fn native_rrf(
    primary: Vec<(String, f32)>,
    secondary: Vec<(String, f32)>,
    k: usize,
) -> Vec<(String, f32)> {
    fusion_to_tuples(fusion::reciprocal_rank_fusion(&primary, &secondary, k))
}

#[pyfunction]
#[pyo3(signature = (primary, secondary, alpha=0.5))]
fn native_linear_combination(
    primary: Vec<(String, f32)>,
    secondary: Vec<(String, f32)>,
    alpha: f32,
) -> Vec<(String, f32)> {
    fusion_to_tuples(fusion::linear_combination(&primary, &secondary, alpha))
}

#[pyfunction]
fn native_min_max_normalize(values: Vec<f32>) -> Vec<f32> {
    fusion::min_max_normalize(&values)
}

#[pyfunction]
fn native_z_score_normalize(values: Vec<f32>) -> Vec<f32> {
    fusion::z_score_normalize(&values)
}

#[pyfunction]
fn native_cdf_normalize(values: Vec<f32>) -> Vec<f32> {
    fusion::cdf_normalize(&values)
}

#[pyfunction]
fn native_ndcg_at_k(ranked: Vec<String>, relevant: Vec<String>, k: usize) -> f32 {
    eval::ndcg_at_k(&ranked, &vec_to_set(relevant), k)
}

#[pyfunction]
fn native_average_precision_at_k(ranked: Vec<String>, relevant: Vec<String>, k: usize) -> f32 {
    eval::average_precision_at_k(&ranked, &vec_to_set(relevant), k)
}

#[pyfunction]
fn native_recall_at_k(ranked: Vec<String>, relevant: Vec<String>, k: usize) -> f32 {
    eval::recall_at_k(&ranked, &vec_to_set(relevant), k)
}

#[pyfunction]
fn native_precision_at_k(ranked: Vec<String>, relevant: Vec<String>, k: usize) -> f32 {
    eval::precision_at_k(&ranked, &vec_to_set(relevant), k)
}

#[pyfunction]
fn native_mrr(ranked: Vec<String>, relevant: Vec<String>) -> f32 {
    eval::mrr(&ranked, &vec_to_set(relevant))
}

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBM25Index>()?;
    module.add_class::<PyLatencyProfiler>()?;
    module.add_function(wrap_pyfunction!(native_rrf, module)?)?;
    module.add_function(wrap_pyfunction!(native_linear_combination, module)?)?;
    module.add_function(wrap_pyfunction!(native_min_max_normalize, module)?)?;
    module.add_function(wrap_pyfunction!(native_z_score_normalize, module)?)?;
    module.add_function(wrap_pyfunction!(native_cdf_normalize, module)?)?;
    module.add_function(wrap_pyfunction!(native_ndcg_at_k, module)?)?;
    module.add_function(wrap_pyfunction!(native_average_precision_at_k, module)?)?;
    module.add_function(wrap_pyfunction!(native_recall_at_k, module)?)?;
    module.add_function(wrap_pyfunction!(native_precision_at_k, module)?)?;
    module.add_function(wrap_pyfunction!(native_mrr, module)?)?;
    Ok(())
}

fn parse_config(config_json: Option<String>) -> PyResult<BM25Config> {
    match config_json {
        Some(json) => serde_json::from_str::<BM25Config>(&json)
            .map_err(|error| PyValueError::new_err(format!("invalid config json: {error}"))),
        None => Ok(BM25Config::default()),
    }
}

fn to_pyerr(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

impl PyBM25Index {
    fn from_index(index: InvertedIndex) -> Self {
        Self {
            inner: Arc::new(RwLock::new(index)),
            wal: Arc::new(RwLock::new(None)),
        }
    }
}

fn hits_to_tuples(results: Vec<retrievalx_core::index::SearchResult>) -> Vec<(String, f32)> {
    results
        .into_iter()
        .map(|r| (r.external_id, r.score))
        .collect()
}

fn fusion_to_tuples(results: Vec<fusion::FusionResult>) -> Vec<(String, f32)> {
    results.into_iter().map(|r| (r.doc_id, r.score)).collect()
}

fn vec_to_set(v: Vec<String>) -> HashSet<String> {
    v.into_iter().collect()
}

fn percentile_ms(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let idx = ((sorted.len() as f32 - 1.0) * p.clamp(0.0, 1.0)).round() as usize;
    sorted.get(idx).copied().unwrap_or(0.0)
}
