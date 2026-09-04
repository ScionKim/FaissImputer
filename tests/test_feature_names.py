import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from faiss_imputer import FaissImputer


TRAIN = np.array(
    [[0, 10, 100], [1, 20, 200], [2, 30, 300]],
    dtype=np.float32,
)
NAMES = ["age", "income", "score"]


@pytest.fixture(params=["complete", "available"])
def imputer(request):
    return FaissImputer(n_neighbors=1, donor_policy=request.param)


def test_generated_feature_names(imputer):
    imputer.fit(TRAIN)
    np.testing.assert_array_equal(
        imputer.get_feature_names_out(), ["x0", "x1", "x2"]
    )


def test_explicit_feature_names(imputer):
    imputer.fit(TRAIN)
    np.testing.assert_array_equal(
        imputer.get_feature_names_out(NAMES), NAMES
    )


def test_feature_name_count_is_checked(imputer):
    imputer.fit(TRAIN)
    with pytest.raises(ValueError):
        imputer.get_feature_names_out(["age"])


def test_feature_names_require_fit(imputer):
    with pytest.raises(NotFittedError):
        imputer.get_feature_names_out()


def test_failed_refit_clears_feature_names(imputer):
    imputer.fit(TRAIN)
    imputer.set_params(n_neighbors=0)
    with pytest.raises(ValueError):
        imputer.fit(TRAIN)
    with pytest.raises(NotFittedError):
        imputer.get_feature_names_out()


def test_pipeline_feature_names(imputer):
    pipeline = make_pipeline(imputer, StandardScaler()).fit(TRAIN)
    np.testing.assert_array_equal(
        pipeline.get_feature_names_out(), ["x0", "x1", "x2"]
    )


def test_column_transformer_preserves_feature_order(imputer):
    transformer = ColumnTransformer(
        [("impute", imputer, [2, 0])],
        remainder="passthrough",
    ).fit(TRAIN)
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(),
        ["impute__x2", "impute__x0", "remainder__x1"],
    )


def test_dataframe_feature_names(imputer):
    pd = pytest.importorskip("pandas")
    imputer.fit(pd.DataFrame(TRAIN, columns=NAMES))
    np.testing.assert_array_equal(imputer.get_feature_names_out(), NAMES)
    with pytest.raises(ValueError):
        imputer.get_feature_names_out(NAMES[::-1])


def test_output_container_selection(imputer):
    pd = pytest.importorskip("pandas")
    imputer.fit(pd.DataFrame(TRAIN, columns=NAMES))
    query = pd.DataFrame([[0, np.nan, 100]], columns=NAMES, index=["row"])
    original = query.copy(deep=True)

    default_result = imputer.transform(query)
    assert isinstance(default_result, np.ndarray)
    assert default_result.dtype == np.float32

    frame_result = imputer.set_output(transform="pandas").transform(query)
    assert isinstance(frame_result, pd.DataFrame)
    assert frame_result.columns.tolist() == NAMES
    assert frame_result.index.tolist() == ["row"]
    np.testing.assert_array_equal(frame_result.to_numpy(), default_result)

    array_result = imputer.set_output(transform="default").transform(query)
    assert isinstance(array_result, np.ndarray)
    assert array_result.dtype == np.float32
    np.testing.assert_array_equal(array_result, default_result)
    pd.testing.assert_frame_equal(query, original)
