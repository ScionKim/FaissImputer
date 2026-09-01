# FaissImputer
[!WARNING]
Maintenance notice: FaissImputer 0.1.x has a known neighbor-mapping bug that can produce incorrect imputations, and it is incompatible with scikit-learn 1.8+. Please do not use it in production. A corrected v0.2.0 is in progress.

[![PyPI Version](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/ScionKim/FaissImputer/blob/main/LICENSE)

Impute missing values using Meta's [faiss](https://github.com/facebookresearch/faiss) - A Python library for missing data imputation with k nearest neighbors.

## Installation

You can install `faiss-imputer` using `pip`:

```bash
pip install faiss-imputer
```

## Usage

```python
import pandas as pd
from faiss_imputer import FaissImputer

# Create your DataFrame and introduce missing values
# ...

# Create an instance of FaissImputer
imputer = FaissImputer(n_neighbors=3)

# Fit the imputer on the data frame with missing values
imputer.fit(df_missing)

# Transform the data frame with missing values
df_imputed = imputer.transform(df_missing)
```

## Parameters

**n_neighbors:** Number of nearest neighbors to consider for imputation.

**metric:** Distance metric to use for nearest neighbor search ('l2' or 'ip').

**strategy:** Imputation strategy ('mean' or 'median').

**index_factory:** Faiss index type ('Flat' or others).

## Example

For a detailed example, refer to the example.py file.

## Contributing

Contributions are welcome! If you find a bug or have an enhancement suggestion, please open an issue or create a pull request.

## Author

![Profile Photo](https://avatars.githubusercontent.com/u/93073728?v=4&size=100)

- **Name:** Hakkil Kim
- **GitHub Profile:** [GitHub Profile Link](https://github.com/ScionKim/)

## License

This project is licensed under the [MIT License](LICENSE).

### Third-Party Licenses

FaissImputer depends on Meta's [Faiss](https://github.com/facebookresearch/faiss), which is distributed under the [MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).

FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.

For detailed licensing information of the Faiss library, please refer to the [Faiss repository](https://github.com/facebookresearch/faiss).
