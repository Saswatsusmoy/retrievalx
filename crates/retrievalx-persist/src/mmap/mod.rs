#![allow(unsafe_code)]

use std::fs::File;
use std::path::Path;

use memmap2::MmapOptions;

use crate::PersistError;

#[derive(Debug)]
pub struct MemoryMappedFile {
    mmap: memmap2::Mmap,
}

impl MemoryMappedFile {
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
}

pub fn memory_map_file<P: AsRef<Path>>(path: P) -> Result<MemoryMappedFile, PersistError> {
    let file = File::open(path)?;
    let mmap = {
        // SAFETY: The mapped file descriptor remains valid for the duration of mapping creation.
        // The returned `Mmap` owns the mapping and enforces read-only access from safe Rust.
        unsafe { MmapOptions::new().map(&file) }
    }
    .map_err(PersistError::Io)?;

    Ok(MemoryMappedFile { mmap })
}
