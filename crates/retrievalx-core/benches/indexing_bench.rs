use criterion::{criterion_group, criterion_main, Criterion};
use retrievalx_core::{BM25Config, InvertedIndex};

fn bench_bulk_insert(c: &mut Criterion) {
    c.bench_function("bulk_insert_10k", |b| {
        b.iter(|| {
            let mut index = InvertedIndex::new(BM25Config::default()).expect("init index");
            let docs = (0..10_000)
                .map(|i| format!("document {i} rust retrieval engine"))
                .collect::<Vec<_>>();
            index.insert_batch(docs).expect("insert batch");
        });
    });
}

criterion_group!(benches, bench_bulk_insert);
criterion_main!(benches);
