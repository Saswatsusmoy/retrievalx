#![forbid(unsafe_code)]

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::PersistError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalOperation {
    Insert {
        external_id: Option<String>,
        text: String,
        fields: std::collections::HashMap<String, String>,
    },
    Delete {
        external_id: String,
    },
}

#[derive(Debug, Clone)]
pub struct WriteAheadLog {
    path: PathBuf,
    sync_on_append: bool,
}

impl WriteAheadLog {
    pub fn open<P: AsRef<Path>>(path: P) -> Self {
        Self::open_with_options(path, true)
    }

    pub fn open_with_options<P: AsRef<Path>>(path: P, sync_on_append: bool) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            sync_on_append,
        }
    }

    pub fn append(&self, op: &WalOperation) -> Result<(), PersistError> {
        self.append_many(std::iter::once(op))
    }

    pub fn append_many<'a, I>(&self, ops: I) -> Result<(), PersistError>
    where
        I: IntoIterator<Item = &'a WalOperation>,
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        for op in ops {
            let line = serde_json::to_string(op).map_err(PersistError::Json)?;
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
        }

        file.flush()?;
        if self.sync_on_append {
            file.sync_data()?;
        }
        Ok(())
    }

    pub fn replay(&self) -> Result<Vec<WalOperation>, PersistError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let bytes = std::fs::read(&self.path)?;
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        let mut operations = Vec::new();
        let mut lines = bytes.split(|b| *b == b'\n').peekable();
        while let Some(line) = lines.next() {
            if line.iter().all(|byte| byte.is_ascii_whitespace()) {
                continue;
            }
            match serde_json::from_slice::<WalOperation>(line) {
                Ok(op) => operations.push(op),
                Err(error) => {
                    // Tolerate a truncated last line caused by abrupt shutdown.
                    if lines.peek().is_none() {
                        break;
                    }
                    return Err(PersistError::Json(error));
                }
            }
        }
        Ok(operations)
    }

    pub fn truncate(&self) -> Result<(), PersistError> {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        if self.sync_on_append {
            OpenOptions::new()
                .read(true)
                .open(&self.path)?
                .sync_data()?;
        }
        Ok(())
    }
}
