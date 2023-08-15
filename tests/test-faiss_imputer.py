import numpy as np
import pandas as pd
from faiss_imputer import FaissImputer

# Set the random seed for reproducibility
np.random.seed(42)

# Generate a random data frame with 10 rows and 5 columns
df = pd.DataFrame(np.random.randint(0, 100, size=(10, 5)), columns=list('ABCDE'))

# Print the original data frame
print("Original data frame:")
print(df)

# Introduce some missing values randomly
df_missing = df.copy()
df_missing.iloc[np.random.randint(0, 10, size=3), np.random.randint(0, 5, size=3)] = np.nan

# Print the data frame with missing values
print("Data frame with missing values:")
print(df_missing)

# Create an instance of FaissImputer with default parameters
imputer = FaissImputer(5, strategy='median')

# Fit the imputer on the data frame with missing values
imputer.fit(df_missing)

# Transform the data frame with missing values
df_imputed = imputer.transform(df_missing)

# Print the imputed data frame
print("Imputed data frame:")
print(df_imputed)

# Compare the imputed data frame with the original data frame
print("Comparison:")
print(np.where(df_imputed == df, 'O', 'X'))