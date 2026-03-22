use retrievalx_core::BM25Config;
use retrievalx_eval::run_benchmark;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_benchmark("scifact", ".cache/beir", BM25Config::default(), 100)?;
    println!("{}", report.summary_ascii_table());
    report.write_csv("scifact-report.csv")?;
    Ok(())
}
