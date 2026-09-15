# Migration notes

[Project README](../README.md) · [API reference](api.md)

## Upgrading from 0.1.x

FaissImputer 0.1.x has a known neighbor-mapping bug and is incompatible
with scikit-learn 1.8 and newer. Do not use those releases for new work.

Upgrade to the latest compatible release:

    python -m pip install --upgrade faiss-imputer

If earlier results were produced with 0.1.x, fit a new estimator and
rerun the imputation. Check the resulting values before reusing those
results in downstream analyses.

## Reproducing historical results

Historical benchmark reports retain the package versions, source commits,
dependencies, and conditions used for their measurements. They do not
automatically describe the current package.

Use the versions and reproduction instructions recorded in the relevant
report when reproducing an older measurement.

- [Release history](https://github.com/ScionKim/FaissImputer/releases)
- [Benchmark reports](benchmarks/README.md)