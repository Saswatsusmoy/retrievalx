#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use zip::ZipArchive;

use retrievalx_core::index::InvertedIndex;
use retrievalx_core::{BM25Config, RetrievalStrategy, ScoringVariant};

use crate::metrics::{average_precision_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k};

const DEFAULT_BEIR_BASE_URL: &str =
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets";
const DEFAULT_MAX_ARCHIVE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

const KNOWN_DATASETS: &[&str] = &[
    "msmarco",
    "trec-covid",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
    "cqadupstack",
    "quora",
    "robust04",
    "signal1m",
    "trec-news",
];

#[derive(Debug, Error)]
pub enum BeirError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("unsupported dataset: {0}")]
    UnsupportedDataset(String),
    #[error("invalid archive entry path: {0}")]
    InvalidArchiveEntry(String),
    #[error("archive exceeded max allowed size: {bytes} > {limit}")]
    ArchiveTooLarge { bytes: u64, limit: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeirDocument {
    #[serde(rename = "_id")]
    pub id: String,
    pub title: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeirQuery {
    #[serde(rename = "_id")]
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeirQrel {
    pub query_id: String,
    pub corpus_id: String,
    pub score: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeirDataset {
    pub corpus: Vec<BeirDocument>,
    pub queries: Vec<BeirQuery>,
    pub qrels: HashMap<String, HashMap<String, i32>>,
}

#[derive(Debug, Clone)]
pub struct BeirLoader {
    root: PathBuf,
}

impl BeirLoader {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn load(&self) -> Result<BeirDataset, BeirError> {
        let corpus = read_jsonl::<BeirDocument, _>(self.root.join("corpus.jsonl"))?;
        let queries = read_jsonl::<BeirQuery, _>(self.root.join("queries.jsonl"))?;

        let qrels_path = self.root.join("qrels.tsv");
        let qrels = if qrels_path.exists() {
            read_qrels_tsv(qrels_path)?
        } else {
            HashMap::new()
        };

        Ok(BeirDataset {
            corpus,
            queries,
            qrels,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BeirCache {
    root: PathBuf,
    max_archive_bytes: u64,
}

impl BeirCache {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            max_archive_bytes: DEFAULT_MAX_ARCHIVE_BYTES,
        }
    }

    pub fn with_max_archive_bytes(mut self, max_archive_bytes: u64) -> Self {
        self.max_archive_bytes = max_archive_bytes;
        self
    }

    pub fn ensure_dataset(&self, dataset: &str) -> Result<PathBuf, BeirError> {
        if !known_datasets().contains(&dataset) {
            return Err(BeirError::UnsupportedDataset(dataset.to_owned()));
        }

        fs::create_dir_all(&self.root)?;

        let dataset_dir = self.root.join(dataset);
        if dataset_is_ready(&dataset_dir) {
            return Ok(dataset_dir);
        }

        let archive_path = self.root.join(format!("{dataset}.zip"));
        if !archive_path.exists() {
            let url = dataset_url(dataset)?;
            download_archive(&url, &archive_path, self.max_archive_bytes)?;
        }

        extract_archive(&archive_path, &self.root)?;

        if dataset_is_ready(&dataset_dir) {
            Ok(dataset_dir)
        } else {
            find_extracted_dataset_dir(&self.root, dataset)
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BeirAggregateMetrics {
    pub ndcg_at_10: f32,
    pub map_at_10: f32,
    pub recall_at_10: f32,
    pub precision_at_10: f32,
    pub mrr: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeirQueryMetrics {
    pub query_id: String,
    pub latency_ms: f32,
    pub ndcg_at_10: f32,
    pub map_at_10: f32,
    pub recall_at_10: f32,
    pub precision_at_10: f32,
    pub mrr: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeirRunReport {
    pub dataset: String,
    pub scoring_variant: String,
    pub retrieval_strategy: String,
    pub top_k: usize,
    pub num_docs: usize,
    pub num_queries: usize,
    pub qps: f32,
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub aggregate: BeirAggregateMetrics,
    pub per_query: Vec<BeirQueryMetrics>,
}

impl BeirRunReport {
    pub fn to_csv(&self) -> String {
        let mut csv = String::from(
            "query_id,latency_ms,ndcg_at_10,map_at_10,recall_at_10,precision_at_10,mrr\n",
        );

        for row in &self.per_query {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                row.query_id,
                row.latency_ms,
                row.ndcg_at_10,
                row.map_at_10,
                row.recall_at_10,
                row.precision_at_10,
                row.mrr
            ));
        }

        csv
    }

    pub fn summary_ascii_table(&self) -> String {
        format!(
            "dataset={} strategy={} scoring={} queries={} docs={} qps={:.2} p50={:.2}ms p95={:.2}ms p99={:.2}ms ndcg@10={:.4} map@10={:.4} recall@10={:.4} precision@10={:.4} mrr={:.4}",
            self.dataset,
            self.retrieval_strategy,
            self.scoring_variant,
            self.num_queries,
            self.num_docs,
            self.qps,
            self.p50_ms,
            self.p95_ms,
            self.p99_ms,
            self.aggregate.ndcg_at_10,
            self.aggregate.map_at_10,
            self.aggregate.recall_at_10,
            self.aggregate.precision_at_10,
            self.aggregate.mrr
        )
    }

    pub fn write_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), BeirError> {
        fs::write(path, self.to_csv())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecallDegradationRow {
    pub strategy: String,
    pub k: usize,
    pub average_recall_vs_exhaustive: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecallDegradationReport {
    pub dataset: String,
    pub rows: Vec<RecallDegradationRow>,
}

impl RecallDegradationReport {
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("strategy,k,average_recall_vs_exhaustive\n");
        for row in &self.rows {
            csv.push_str(&format!(
                "{},{},{}\n",
                row.strategy, row.k, row.average_recall_vs_exhaustive
            ));
        }
        csv
    }

    pub fn summary_ascii_table(&self) -> String {
        let body = self
            .rows
            .iter()
            .map(|row| {
                format!(
                    "{}@{}={:.4}",
                    row.strategy, row.k, row.average_recall_vs_exhaustive
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        format!("dataset={} {}", self.dataset, body)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VariantComparisonRow {
    pub scoring_variant: String,
    pub ndcg_at_10: f32,
    pub map_at_10: f32,
    pub recall_at_10: f32,
    pub precision_at_10: f32,
    pub mrr: f32,
    pub qps: f32,
    pub p95_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VariantComparisonReport {
    pub dataset: String,
    pub retrieval_strategy: String,
    pub top_k: usize,
    pub rows: Vec<VariantComparisonRow>,
}

impl VariantComparisonReport {
    pub fn to_csv(&self) -> String {
        let mut csv = String::from(
            "scoring_variant,ndcg_at_10,map_at_10,recall_at_10,precision_at_10,mrr,qps,p95_ms\n",
        );
        for row in &self.rows {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                row.scoring_variant,
                row.ndcg_at_10,
                row.map_at_10,
                row.recall_at_10,
                row.precision_at_10,
                row.mrr,
                row.qps,
                row.p95_ms
            ));
        }
        csv
    }

    pub fn summary_ascii_table(&self) -> String {
        let rows = self
            .rows
            .iter()
            .map(|row| {
                format!(
                    "{} ndcg@10={:.4} map@10={:.4} qps={:.2} p95={:.2}ms",
                    row.scoring_variant, row.ndcg_at_10, row.map_at_10, row.qps, row.p95_ms
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!(
            "dataset={} retrieval={} top_k={} {}",
            self.dataset, self.retrieval_strategy, self.top_k, rows
        )
    }
}

pub fn known_datasets() -> &'static [&'static str] {
    KNOWN_DATASETS
}

pub fn dataset_url(dataset: &str) -> Result<String, BeirError> {
    if !known_datasets().contains(&dataset) {
        return Err(BeirError::UnsupportedDataset(dataset.to_owned()));
    }

    Ok(format!("{DEFAULT_BEIR_BASE_URL}/{dataset}.zip"))
}

pub fn run_benchmark<P: AsRef<Path>>(
    dataset_name: &str,
    cache_dir: P,
    config: BM25Config,
    top_k: usize,
) -> Result<BeirRunReport, BeirError> {
    let cache = BeirCache::new(cache_dir);
    let dataset_path = cache.ensure_dataset(dataset_name)?;
    let dataset = BeirLoader::new(dataset_path).load()?;
    run_benchmark_on_dataset(dataset_name, &dataset, config, top_k)
}

pub fn run_benchmark_on_dataset(
    dataset_name: &str,
    dataset: &BeirDataset,
    config: BM25Config,
    top_k: usize,
) -> Result<BeirRunReport, BeirError> {
    let index = build_index_from_dataset(dataset, &config)?;
    Ok(run_benchmark_with_index(
        dataset_name,
        dataset,
        &index,
        &config,
        top_k,
    ))
}

pub fn run_recall_degradation<P: AsRef<Path>>(
    dataset_name: &str,
    cache_dir: P,
    config: BM25Config,
    ks: &[usize],
) -> Result<RecallDegradationReport, BeirError> {
    let cache = BeirCache::new(cache_dir);
    let dataset_path = cache.ensure_dataset(dataset_name)?;
    let dataset = BeirLoader::new(dataset_path).load()?;
    run_recall_degradation_on_dataset(dataset_name, &dataset, config, ks)
}

pub fn run_recall_degradation_on_dataset(
    dataset_name: &str,
    dataset: &BeirDataset,
    config: BM25Config,
    ks: &[usize],
) -> Result<RecallDegradationReport, BeirError> {
    let ks = normalize_ks(ks);
    if ks.is_empty() {
        return Ok(RecallDegradationReport {
            dataset: dataset_name.to_owned(),
            rows: Vec::new(),
        });
    }

    let max_k = *ks.last().unwrap_or(&10);
    let mut exhaustive_cfg = config.clone();
    exhaustive_cfg.retrieval = RetrievalStrategy::ExhaustiveDAAT;
    let exhaustive_index = build_index_from_dataset(dataset, &exhaustive_cfg)?;

    let strategy_configs = vec![
        (
            "Wand".to_owned(),
            RetrievalStrategy::Wand { top_k_budget: 1000 },
        ),
        (
            "BlockMaxWand".to_owned(),
            RetrievalStrategy::BlockMaxWand { top_k_budget: 1000 },
        ),
        ("MaxScore".to_owned(), RetrievalStrategy::MaxScore),
    ];

    let mut strategy_indexes = Vec::<(String, InvertedIndex)>::new();
    for (name, retrieval) in strategy_configs {
        let mut cfg = config.clone();
        cfg.retrieval = retrieval;
        strategy_indexes.push((name, build_index_from_dataset(dataset, &cfg)?));
    }

    let mut sums = HashMap::<(String, usize), (f32, usize)>::new();
    for query in &dataset.queries {
        let ground_truth = exhaustive_index
            .search(&query.text, max_k)
            .into_iter()
            .map(|result| result.external_id)
            .collect::<Vec<_>>();

        for (strategy, index) in &strategy_indexes {
            let candidate = index
                .search(&query.text, max_k)
                .into_iter()
                .map(|result| result.external_id)
                .collect::<Vec<_>>();

            for &k in &ks {
                let recall = recall_against_ground_truth(&ground_truth, &candidate, k);
                let entry = sums.entry((strategy.clone(), k)).or_insert((0.0, 0));
                entry.0 += recall;
                entry.1 += 1;
            }
        }
    }

    let mut rows = sums
        .into_iter()
        .map(|((strategy, k), (sum, count))| RecallDegradationRow {
            strategy,
            k,
            average_recall_vs_exhaustive: if count == 0 { 0.0 } else { sum / count as f32 },
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| a.strategy.cmp(&b.strategy).then_with(|| a.k.cmp(&b.k)));

    Ok(RecallDegradationReport {
        dataset: dataset_name.to_owned(),
        rows,
    })
}

pub fn run_variant_comparison<P: AsRef<Path>>(
    dataset_name: &str,
    cache_dir: P,
    base_config: BM25Config,
    top_k: usize,
) -> Result<VariantComparisonReport, BeirError> {
    let cache = BeirCache::new(cache_dir);
    let dataset_path = cache.ensure_dataset(dataset_name)?;
    let dataset = BeirLoader::new(dataset_path).load()?;
    run_variant_comparison_on_dataset(dataset_name, &dataset, base_config, top_k)
}

pub fn run_variant_comparison_on_dataset(
    dataset_name: &str,
    dataset: &BeirDataset,
    base_config: BM25Config,
    top_k: usize,
) -> Result<VariantComparisonReport, BeirError> {
    let mut rows = Vec::new();

    for scoring in default_scoring_variants() {
        let mut config = base_config.clone();
        config.scoring = scoring.clone();
        let run = run_benchmark_on_dataset(dataset_name, dataset, config, top_k)?;

        rows.push(VariantComparisonRow {
            scoring_variant: format!("{scoring:?}"),
            ndcg_at_10: run.aggregate.ndcg_at_10,
            map_at_10: run.aggregate.map_at_10,
            recall_at_10: run.aggregate.recall_at_10,
            precision_at_10: run.aggregate.precision_at_10,
            mrr: run.aggregate.mrr,
            qps: run.qps,
            p95_ms: run.p95_ms,
        });
    }

    rows.sort_by(|a, b| b.ndcg_at_10.total_cmp(&a.ndcg_at_10));

    Ok(VariantComparisonReport {
        dataset: dataset_name.to_owned(),
        retrieval_strategy: format!("{:?}", base_config.retrieval),
        top_k,
        rows,
    })
}

fn run_benchmark_with_index(
    dataset_name: &str,
    dataset: &BeirDataset,
    index: &InvertedIndex,
    config: &BM25Config,
    top_k: usize,
) -> BeirRunReport {
    let eval_start = Instant::now();
    let mut per_query = Vec::with_capacity(dataset.queries.len());

    for query in &dataset.queries {
        let start = Instant::now();
        let ranked = index.search(&query.text, top_k);
        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;

        let ranked_ids = ranked
            .iter()
            .map(|result| result.external_id.clone())
            .collect::<Vec<_>>();

        let relevant = dataset
            .qrels
            .get(&query.id)
            .map_or_else(HashSet::new, |entries| {
                entries
                    .iter()
                    .filter_map(|(doc_id, score)| {
                        if *score > 0 {
                            Some(doc_id.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<HashSet<_>>()
            });

        let ndcg = ndcg_at_k(&ranked_ids, &relevant, 10);
        let map = average_precision_at_k(&ranked_ids, &relevant, 10);
        let recall = recall_at_k(&ranked_ids, &relevant, 10);
        let precision = precision_at_k(&ranked_ids, &relevant, 10);
        let mrr_score = mrr(&ranked_ids, &relevant);

        per_query.push(BeirQueryMetrics {
            query_id: query.id.clone(),
            latency_ms,
            ndcg_at_10: ndcg,
            map_at_10: map,
            recall_at_10: recall,
            precision_at_10: precision,
            mrr: mrr_score,
        });
    }

    let total_elapsed = eval_start.elapsed().as_secs_f32();
    let num_queries = per_query.len();

    let aggregate = aggregate_metrics(&per_query);
    let p50_ms = percentile_latency(&per_query, 0.50);
    let p95_ms = percentile_latency(&per_query, 0.95);
    let p99_ms = percentile_latency(&per_query, 0.99);

    let qps = if total_elapsed > 0.0 {
        num_queries as f32 / total_elapsed
    } else {
        0.0
    };

    BeirRunReport {
        dataset: dataset_name.to_owned(),
        scoring_variant: format!("{:?}", config.scoring),
        retrieval_strategy: format!("{:?}", config.retrieval),
        top_k,
        num_docs: dataset.corpus.len(),
        num_queries,
        qps,
        p50_ms,
        p95_ms,
        p99_ms,
        aggregate,
        per_query,
    }
}

fn build_index_from_dataset(
    dataset: &BeirDataset,
    config: &BM25Config,
) -> Result<InvertedIndex, BeirError> {
    let mut index = InvertedIndex::new(config.clone())
        .map_err(|error| BeirError::Io(std::io::Error::other(error.to_string())))?;

    for doc in &dataset.corpus {
        let combined_text = if let Some(title) = &doc.title {
            format!("{title}\n{}", doc.text)
        } else {
            doc.text.clone()
        };

        index
            .insert_document(Some(doc.id.clone()), combined_text, HashMap::new())
            .map_err(|error| BeirError::Io(std::io::Error::other(error.to_string())))?;
    }

    Ok(index)
}

fn default_scoring_variants() -> Vec<ScoringVariant> {
    vec![
        ScoringVariant::Okapi { k1: 1.2, b: 0.75 },
        ScoringVariant::Plus {
            k1: 1.2,
            b: 0.75,
            delta: 0.5,
        },
        ScoringVariant::L {
            k1: 1.2,
            b: 0.75,
            c: 1.0,
        },
        ScoringVariant::Adpt { b: 0.75 },
        ScoringVariant::F {
            k1: 1.2,
            b: 0.75,
            fields: Vec::new(),
        },
        ScoringVariant::T {
            default_k1: 1.2,
            b: 0.75,
            term_k1: HashMap::new(),
        },
        ScoringVariant::Atire { k1: 1.2, b: 0.75 },
        ScoringVariant::TfIdf,
    ]
}

fn normalize_ks(ks: &[usize]) -> Vec<usize> {
    let mut out = ks.iter().copied().filter(|k| *k > 0).collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn recall_against_ground_truth(ground_truth: &[String], candidate: &[String], k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }

    let gt = ground_truth
        .iter()
        .take(k)
        .map(String::as_str)
        .collect::<HashSet<_>>();
    if gt.is_empty() {
        return 1.0;
    }

    let got = candidate
        .iter()
        .take(k)
        .map(String::as_str)
        .collect::<HashSet<_>>();

    let hits = gt.intersection(&got).count();
    hits as f32 / gt.len() as f32
}

fn aggregate_metrics(per_query: &[BeirQueryMetrics]) -> BeirAggregateMetrics {
    if per_query.is_empty() {
        return BeirAggregateMetrics::default();
    }

    let count = per_query.len() as f32;

    BeirAggregateMetrics {
        ndcg_at_10: per_query.iter().map(|row| row.ndcg_at_10).sum::<f32>() / count,
        map_at_10: per_query.iter().map(|row| row.map_at_10).sum::<f32>() / count,
        recall_at_10: per_query.iter().map(|row| row.recall_at_10).sum::<f32>() / count,
        precision_at_10: per_query.iter().map(|row| row.precision_at_10).sum::<f32>() / count,
        mrr: per_query.iter().map(|row| row.mrr).sum::<f32>() / count,
    }
}

fn percentile_latency(per_query: &[BeirQueryMetrics], percentile: f32) -> f32 {
    if per_query.is_empty() {
        return 0.0;
    }

    let mut values = per_query
        .iter()
        .map(|row| row.latency_ms)
        .collect::<Vec<_>>();
    values.sort_by(f32::total_cmp);

    let idx = ((values.len() as f32 - 1.0) * percentile).round() as usize;
    values.get(idx).copied().unwrap_or(0.0)
}

fn dataset_is_ready(path: &Path) -> bool {
    path.join("corpus.jsonl").exists()
        && path.join("queries.jsonl").exists()
        && path.join("qrels.tsv").exists()
}

fn find_extracted_dataset_dir(root: &Path, dataset: &str) -> Result<PathBuf, BeirError> {
    let mut candidates = fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(dataset))
        })
        .collect::<Vec<_>>();

    if let Some(path) = candidates.pop() {
        if dataset_is_ready(&path) {
            return Ok(path);
        }
    }

    Err(BeirError::UnsupportedDataset(dataset.to_owned()))
}

fn download_archive(url: &str, path: &Path, max_archive_bytes: u64) -> Result<(), BeirError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let mut response = client.get(url).send()?.error_for_status()?;
    let mut file = File::create(path)?;

    let mut total_written = 0_u64;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }

        total_written += read as u64;
        if total_written > max_archive_bytes {
            return Err(BeirError::ArchiveTooLarge {
                bytes: total_written,
                limit: max_archive_bytes,
            });
        }

        file.write_all(&buffer[..read])?;
    }

    file.flush()?;
    Ok(())
}

fn extract_archive(archive_path: &Path, output_root: &Path) -> Result<(), BeirError> {
    let file = File::open(archive_path)?;
    let mut archive = ZipArchive::new(file)?;

    fs::create_dir_all(output_root)?;

    for idx in 0..archive.len() {
        let mut entry = archive.by_index(idx)?;
        let Some(name) = entry.enclosed_name() else {
            return Err(BeirError::InvalidArchiveEntry(entry.name().to_owned()));
        };

        let out_path = output_root.join(name);
        if entry.name().ends_with('/') {
            fs::create_dir_all(&out_path)?;
            continue;
        }

        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut out_file = File::create(&out_path)?;
        std::io::copy(&mut entry, &mut out_file)?;
    }

    Ok(())
}

fn read_jsonl<T, P>(path: P) -> Result<Vec<T>, BeirError>
where
    T: for<'de> Deserialize<'de>,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str::<T>(&line)?;
        out.push(value);
    }
    Ok(out)
}

fn read_qrels_tsv<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, HashMap<String, i32>>, BeirError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut qrels = HashMap::<String, HashMap<String, i32>>::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if idx == 0 && line.to_ascii_lowercase().contains("query-id") {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            continue;
        }

        let query_id = parts[0].to_owned();
        let corpus_id = parts[1].to_owned();
        let score = parts[2].parse::<i32>().unwrap_or(0);

        qrels.entry(query_id).or_default().insert(corpus_id, score);
    }

    Ok(qrels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_metrics_for_synthetic_dataset() {
        let dataset = BeirDataset {
            corpus: vec![
                BeirDocument {
                    id: "d1".to_owned(),
                    title: None,
                    text: "rust retrieval".to_owned(),
                },
                BeirDocument {
                    id: "d2".to_owned(),
                    title: None,
                    text: "python retrieval".to_owned(),
                },
            ],
            queries: vec![BeirQuery {
                id: "q1".to_owned(),
                text: "rust".to_owned(),
            }],
            qrels: HashMap::from([("q1".to_owned(), HashMap::from([("d1".to_owned(), 1)]))]),
        };

        let report = run_benchmark_on_dataset("synthetic", &dataset, BM25Config::default(), 5)
            .expect("benchmark should run");

        assert_eq!(report.dataset, "synthetic");
        assert_eq!(report.num_docs, 2);
        assert_eq!(report.num_queries, 1);
        assert!(report.aggregate.recall_at_10 >= 1.0);
        assert!(report.summary_ascii_table().contains("dataset=synthetic"));
        assert!(report.to_csv().contains("query_id"));
    }

    #[test]
    fn known_dataset_has_url() {
        let url = dataset_url("msmarco").expect("msmarco should be supported");
        assert!(url.ends_with("/msmarco.zip"));
    }

    #[test]
    fn recall_degradation_and_variant_comparison_reports_work() {
        let dataset = BeirDataset {
            corpus: vec![
                BeirDocument {
                    id: "d1".to_owned(),
                    title: None,
                    text: "rust retrieval".to_owned(),
                },
                BeirDocument {
                    id: "d2".to_owned(),
                    title: None,
                    text: "python retrieval".to_owned(),
                },
                BeirDocument {
                    id: "d3".to_owned(),
                    title: None,
                    text: "hybrid sparse dense".to_owned(),
                },
            ],
            queries: vec![
                BeirQuery {
                    id: "q1".to_owned(),
                    text: "rust".to_owned(),
                },
                BeirQuery {
                    id: "q2".to_owned(),
                    text: "retrieval".to_owned(),
                },
            ],
            qrels: HashMap::from([
                ("q1".to_owned(), HashMap::from([("d1".to_owned(), 1)])),
                (
                    "q2".to_owned(),
                    HashMap::from([("d1".to_owned(), 1), ("d2".to_owned(), 1)]),
                ),
            ]),
        };

        let recall = run_recall_degradation_on_dataset(
            "synthetic",
            &dataset,
            BM25Config::default(),
            &[1, 3],
        )
        .expect("recall report should run");
        assert!(!recall.rows.is_empty());
        assert!(recall
            .rows
            .iter()
            .all(|row| (0.0..=1.0).contains(&row.average_recall_vs_exhaustive)));
        assert!(recall.to_csv().contains("strategy,k"));

        let comparison =
            run_variant_comparison_on_dataset("synthetic", &dataset, BM25Config::default(), 3)
                .expect("variant comparison should run");
        assert!(!comparison.rows.is_empty());
        assert!(comparison.to_csv().contains("scoring_variant"));
        assert!(comparison
            .rows
            .windows(2)
            .all(|rows| rows[0].ndcg_at_10 >= rows[1].ndcg_at_10));
    }
}
