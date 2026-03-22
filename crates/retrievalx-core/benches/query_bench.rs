use criterion::{criterion_group, criterion_main, Criterion};
use retrievalx_core::{BM25Config, InvertedIndex};

fn bench_query_qps(c: &mut Criterion) {
    let mut index = InvertedIndex::new(BM25Config::default()).expect("init index");
    let docs = (0..50_000)
        .map(|i| format!("document {i} about rust python bm25 ranking"))
        .collect::<Vec<_>>();
    index.insert_batch(docs).expect("insert docs");

    c.bench_function("query_top10", |b| {
        b.iter(|| {
            let _ = index.search("rust retrieval", 10);
        });
    });
}

criterion_group!(benches, bench_query_qps);
criterion_main!(benches);
