# FaissImputer

[![PyPI Version](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/your-username/FaissImputer/blob/main/LICENSE)

Impute missing values using faiss - A Python library for missing data imputation with k nearest neighbors.

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

## License

This project is licensed under the [MIT License](LICENSE).

### Third-Party Licenses

This project utilizes code from Meta's Faiss library, which is distributed under the [Apache License 2.0](https://github.com/facebookresearch/faiss/blob/master/LICENSE).

Please note that while this project includes code from the Faiss library, it is not officially associated with or endorsed by the Faiss maintainers or Meta.

For detailed licensing information of the Faiss library, please refer to the [Faiss repository](https://github.com/facebookresearch/faiss).
