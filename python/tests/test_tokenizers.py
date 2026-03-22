from retrievalx import BM25Config, BM25Index, Filter, Tokenizer, TokenizerConfig


def test_regex_tokenizer_pipeline() -> None:
    tok = TokenizerConfig(
        tokenizer=Tokenizer.regex(r"[A-Za-z]+"),
        filters=[Filter.LOWERCASE],
    )
    index = BM25Index(config=BM25Config(tokenizer=tok))
    index.insert_batch(["Rust-2026 retrievalx"])
    results = index.search("rust", top_k=1)
    assert results
