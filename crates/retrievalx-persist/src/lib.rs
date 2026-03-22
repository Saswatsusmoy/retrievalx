#![deny(unsafe_code)]

pub mod codec;
pub mod export;
pub mod mmap;
pub mod wal;

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use retrievalx_core::index::{IndexSnapshot, InvertedIndex};
use retrievalx_core::CoreError;
use wal::{WalOperation, WriteAheadLog};

const MAGIC: &[u8; 8] = b"RTRVLX01";
const FORMAT_VERSION_JSON_V1: u32 = 1;
const FORMAT_VERSION_MSGPACK_V2: u32 = 2;
const FORMAT_VERSION: u32 = FORMAT_VERSION_MSGPACK_V2;

#[derive(Debug, Error)]
pub enum PersistError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("core error: {0}")]
    Core(#[from] retrievalx_core::CoreError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("msgpack encode error: {0}")]
    MsgpackEncode(#[from] rmp_serde::encode::Error),
    #[error("msgpack decode error: {0}")]
    MsgpackDecode(#[from] rmp_serde::decode::Error),
    #[error("invalid index file magic")]
    InvalidMagic,
    #[error("unsupported index format version: {0}")]
    UnsupportedFormatVersion(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetadataSidecar {
    pub vocabulary_size: usize,
    pub num_documents: usize,
    pub avgdl: f32,
    pub scoring_variant: retrievalx_core::ScoringVariant,
    pub tokenizer_config: retrievalx_tokenize::TokenizerConfig,
    pub build_timestamp_unix: u64,
    pub format_version: u32,
}

pub fn save_index<P: AsRef<Path>>(index: &InvertedIndex, path: P) -> Result<(), PersistError> {
    let snapshot = index.snapshot();
    let bytes = rmp_serde::to_vec_named(&snapshot)?;
    let mut payload = Vec::with_capacity(MAGIC.len() + 4 + bytes.len());
    payload.extend_from_slice(MAGIC);
    payload.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    payload.extend_from_slice(&bytes);
    atomic_write(path.as_ref(), &payload)?;
    Ok(())
}

pub fn load_index<P: AsRef<Path>>(path: P) -> Result<InvertedIndex, PersistError> {
    let bytes = std::fs::read(path)?;
    parse_index_bytes(&bytes)
}

pub fn load_index_mmap<P: AsRef<Path>>(path: P) -> Result<InvertedIndex, PersistError> {
    let mapped = mmap::memory_map_file(path)?;
    parse_index_bytes(mapped.as_slice())
}

pub fn write_metadata_sidecar<P: AsRef<Path>>(
    index: &InvertedIndex,
    path: P,
) -> Result<(), PersistError> {
    let stats = index.stats();
    let metadata = MetadataSidecar {
        vocabulary_size: stats.vocabulary_size,
        num_documents: stats.num_live_docs,
        avgdl: stats.avgdl,
        scoring_variant: index.config().scoring.clone(),
        tokenizer_config: index.config().tokenizer.clone(),
        build_timestamp_unix: stats.build_timestamp_unix,
        format_version: FORMAT_VERSION,
    };

    let json = serde_json::to_vec_pretty(&metadata)?;
    atomic_write(path.as_ref(), &json)?;
    Ok(())
}

pub fn load_index_with_wal<P: AsRef<Path>, Q: AsRef<Path>>(
    index_path: P,
    wal_path: Q,
) -> Result<InvertedIndex, PersistError> {
    let mut index = load_index(index_path)?;
    let wal = WriteAheadLog::open(wal_path);
    let _ = replay_wal_into_index(&mut index, &wal)?;
    Ok(index)
}

pub fn replay_wal_into_index(
    index: &mut InvertedIndex,
    wal: &WriteAheadLog,
) -> Result<usize, PersistError> {
    let mut applied = 0_usize;
    for op in wal.replay()? {
        if apply_wal_operation(index, &op)? {
            applied += 1;
        }
    }
    Ok(applied)
}

pub fn compact_index_with_wal<P: AsRef<Path>>(
    index: &mut InvertedIndex,
    index_path: P,
    wal: &WriteAheadLog,
) -> Result<(), PersistError> {
    index.compact()?;
    save_index(index, index_path)?;
    wal.truncate()?;
    Ok(())
}

fn parse_index_bytes(bytes: &[u8]) -> Result<InvertedIndex, PersistError> {
    if bytes.len() < MAGIC.len() + 4 {
        return Err(PersistError::InvalidMagic);
    }

    let magic = &bytes[..MAGIC.len()];
    if magic != MAGIC {
        return Err(PersistError::InvalidMagic);
    }

    let version_offset = MAGIC.len();
    let version = u32::from_le_bytes([
        bytes[version_offset],
        bytes[version_offset + 1],
        bytes[version_offset + 2],
        bytes[version_offset + 3],
    ]);
    let payload_start = MAGIC.len() + 4;
    let payload = &bytes[payload_start..];
    let snapshot: IndexSnapshot = match version {
        FORMAT_VERSION_JSON_V1 => serde_json::from_slice(payload)?,
        FORMAT_VERSION_MSGPACK_V2 => rmp_serde::from_slice(payload)?,
        other => return Err(PersistError::UnsupportedFormatVersion(other)),
    };
    InvertedIndex::from_snapshot(snapshot).map_err(PersistError::from)
}

fn apply_wal_operation(index: &mut InvertedIndex, op: &WalOperation) -> Result<bool, PersistError> {
    match op {
        WalOperation::Insert {
            external_id,
            text,
            fields,
        } => match index.insert_document(external_id.clone(), text.clone(), fields.clone()) {
            Ok(_) => Ok(true),
            Err(CoreError::InvalidArgument(message))
                if message.contains("duplicate external id") =>
            {
                Ok(false)
            }
            Err(error) => Err(PersistError::Core(error)),
        },
        WalOperation::Delete { external_id } => match index.delete_by_external_id(external_id) {
            Ok(_) => Ok(true),
            Err(CoreError::ExternalIdNotFound(_)) => Ok(false),
            Err(error) => Err(PersistError::Core(error)),
        },
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), PersistError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0_u128, |duration| duration.as_nanos());
    let tmp = path.with_extension(format!("tmp-{nonce}"));

    let mut file = File::create(&tmp)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(&tmp, path)?;
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use retrievalx_core::{BM25Config, InvertedIndex};

    use super::{load_index, load_index_mmap, replay_wal_into_index, save_index};
    use crate::wal::{WalOperation, WriteAheadLog};

    #[test]
    fn mmap_load_matches_regular_load() {
        let mut index = InvertedIndex::new(BM25Config::default()).expect("index init should pass");
        index
            .insert_batch(vec![
                "rust retrieval engine".to_owned(),
                "python search toolkit".to_owned(),
                "hybrid sparse dense retrieval".to_owned(),
            ])
            .expect("insert should pass");

        let path = temp_index_path("retrievalx-mmap-test");
        save_index(&index, &path).expect("save index should pass");

        let regular = load_index(&path).expect("regular load should pass");
        let mapped = load_index_mmap(&path).expect("mmap load should pass");

        let r = regular.search("retrieval", 2);
        let m = mapped.search("retrieval", 2);

        assert_eq!(r, m);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn replays_wal_operations() {
        let mut index = InvertedIndex::new(BM25Config::default()).expect("index init should pass");
        index
            .insert_document(
                Some("base".to_owned()),
                "base document".to_owned(),
                HashMap::new(),
            )
            .expect("insert should pass");

        let wal_path = temp_index_path("retrievalx-wal-test");
        let wal = WriteAheadLog::open(&wal_path);
        wal.append(&WalOperation::Insert {
            external_id: Some("new-doc".to_owned()),
            text: "new wal doc".to_owned(),
            fields: HashMap::new(),
        })
        .expect("wal append should pass");
        wal.append(&WalOperation::Delete {
            external_id: "base".to_owned(),
        })
        .expect("wal append should pass");

        let applied = replay_wal_into_index(&mut index, &wal).expect("wal replay should pass");
        assert_eq!(applied, 2);

        let result = index.search("wal", 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].external_id, "new-doc");
        assert_eq!(index.stats().num_live_docs, 1);

        let _ = std::fs::remove_file(wal_path);
    }

    fn temp_index_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0_u128, |duration| duration.as_nanos());
        std::env::temp_dir().join(format!("{prefix}-{nanos}.rtx"))
    }
}
