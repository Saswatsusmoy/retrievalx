from retrievalx import cdf_normalize, linear_combination, rrf, z_score_normalize


def test_rrf_and_linear_combination() -> None:
    a = [("d1", 2.0), ("d2", 1.0)]
    b = [("d2", 3.0), ("d3", 2.0)]

    rrf_out = rrf(a, b, k=60)
    lin_out = linear_combination(a, b, alpha=0.6)

    assert rrf_out
    assert lin_out
    assert rrf_out[0].doc_id in {"d1", "d2"}


def test_additional_normalizers() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    z = z_score_normalize(values)
    cdf = cdf_normalize(values)
    assert len(z) == len(values)
    assert len(cdf) == len(values)
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
