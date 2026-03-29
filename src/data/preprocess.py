"""
Data preprocessing for model training.

Handles scaling, encoding, train/test split, and dimensionality reduction.
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import get_feature_columns


class DataPreprocessor:
    """Handles all preprocessing for pitcher strikeout prediction."""

    def __init__(
        self,
        n_pca_components: int | None = 50,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize preprocessor.

        Args:
            n_pca_components: Number of PCA components (None to skip PCA)
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining after test)
            random_state: Random seed for reproducibility
        """
        self.n_pca_components = n_pca_components
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []

        self.preprocessor: ColumnTransformer | None = None
        self.pca: PCA | None = None
        self.is_fitted: bool = False

    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Identify numeric and categorical columns."""
        self.feature_columns = get_feature_columns(df)

        self.numeric_columns = [
            c for c in self.feature_columns
            if df[c].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        # Currently we don't use categorical features directly
        # Team info is captured in opponent batting stats
        self.categorical_columns = []

    def _create_preprocessor(self) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline."""
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

        transformers = [
            ('numeric', numeric_pipeline, self.numeric_columns),
        ]

        if self.categorical_columns:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ])
            transformers.append(
                ('categorical', categorical_pipeline, self.categorical_columns)
            )

        return ColumnTransformer(transformers=transformers)

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.

        Args:
            df: Training DataFrame with features

        Returns:
            self
        """
        self._identify_columns(df)
        self.preprocessor = self._create_preprocessor()

        # Fit preprocessing pipeline
        X = df[self.feature_columns]
        X_transformed = self.preprocessor.fit_transform(X)

        # Fit PCA if requested
        if self.n_pca_components is not None:
            n_components = min(self.n_pca_components, X_transformed.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X_transformed)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            df: DataFrame with features

        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        X = df[self.feature_columns]
        X_transformed = self.preprocessor.transform(X)

        if self.pca is not None:
            X_transformed = self.pca.transform(X_transformed)

        return X_transformed

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'K_actual',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/val/test splits with preprocessing.

        Args:
            df: Full DataFrame with features and target
            target_column: Name of target column

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Drop rows with missing target
        df = df.dropna(subset=[target_column])

        # Extract target
        y = df[target_column].values

        # Train/test split
        df_train, df_test, y_train, y_test = train_test_split(
            df, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Train/val split
        df_train, df_val, y_train, y_val = train_test_split(
            df_train, y_train,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
        )

        # Fit on training data only
        self.fit(df_train)

        # Transform all sets
        X_train = self.transform(df_train)
        X_val = self.transform(df_val)
        X_test = self.transform(df_test)

        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save(self, path: str) -> None:
        """Save fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Nothing to save.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'pca': self.pca,
                'feature_columns': self.feature_columns,
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'n_pca_components': self.n_pca_components,
            }, f)

        print(f"Preprocessor saved to {save_path}")

    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load fitted preprocessor from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(n_pca_components=data['n_pca_components'])
        instance.preprocessor = data['preprocessor']
        instance.pca = data['pca']
        instance.feature_columns = data['feature_columns']
        instance.numeric_columns = data['numeric_columns']
        instance.categorical_columns = data['categorical_columns']
        instance.is_fitted = True

        return instance


def preprocess_data(
    features_df: pd.DataFrame,
    n_pca_components: int = 50,
    test_size: float = 0.2,
    val_size: float = 0.1,
    save_path: str | None = None,
) -> Tuple[DataPreprocessor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to preprocess data for training.

    Args:
        features_df: DataFrame with features and target
        n_pca_components: Number of PCA components
        test_size: Test set fraction
        val_size: Validation set fraction
        save_path: Path to save preprocessor (optional)

    Returns:
        Tuple of (preprocessor, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    preprocessor = DataPreprocessor(
        n_pca_components=n_pca_components,
        test_size=test_size,
        val_size=val_size,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(features_df)

    if save_path:
        preprocessor.save(save_path)

    return preprocessor, X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Load sample data
    df = pd.read_csv("data/processed/features.csv")

    preprocessor, X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        n_pca_components=50,
        save_path="models/preprocessor.pkl",
    )
