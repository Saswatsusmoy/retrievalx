use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use retrievalx_core::{BM25Config, InvertedIndex, RetrievalStrategy};

fn benchmark_strategies(c: &mut Criterion) {
    let docs = (0..20_000)
        .map(|i| format!("doc {i} rust python retrieval and bm25 ranking"))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("topk_strategies");

    for (name, strategy) in [
        ("exhaustive", RetrievalStrategy::ExhaustiveDAAT),
        ("wand", RetrievalStrategy::Wand { top_k_budget: 1000 }),
        (
            "blockmax",
            RetrievalStrategy::BlockMaxWand { top_k_budget: 1000 },
        ),
    ] {
        let config = BM25Config {
            retrieval: strategy,
            ..BM25Config::default()
        };

        let mut index = InvertedIndex::new(config).expect("init index");
        index.insert_batch(docs.clone()).expect("insert docs");

        group.bench_with_input(BenchmarkId::new("query", name), &name, |b, _| {
            b.iter(|| {
                let _ = index.search("rust retrieval", 10);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_strategies);
criterion_main!(benches);
