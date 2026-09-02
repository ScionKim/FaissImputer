import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.utils import get_tags

from faiss_imputer import FaissImputer


def test_transformer_tag():
    tags = get_tags(FaissImputer())
    assert tags.transformer_tags is not None


def test_nan_support_tag():
    tags = get_tags(FaissImputer())
    assert tags.input_tags.allow_nan


def test_dtype_preservation_tag():
    tags = get_tags(FaissImputer())
    assert tags.transformer_tags is not None
    assert tags.transformer_tags.preserves_dtype == ["float32"]


def test_pipeline_fit_and_transform():
    pipeline = make_pipeline(FaissImputer(n_neighbors=1))
    train = np.array(
        [[0.0, 10.0], [2.0, 20.0]],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        pipeline.fit_transform(train),
        train,
    )

    result = pipeline.transform([[1.9, np.nan]])

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[1.9, 20.0]])
