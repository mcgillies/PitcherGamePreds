"""
Binary classifier ensemble for pitcher-batter matchup prediction.

Trains separate binary classifiers for each outcome:
- K vs not-K
- BB vs not-BB
- HR vs not-HR
- 1B vs not-1B
- 2B vs not-2B
- 3B vs not-3B
- OUT vs not-OUT

Each model can use optimized features for that specific outcome.
Outputs are normalized to produce valid probability distributions.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from flaml import AutoML
import shap


# Outcome classes (alphabetical to match sklearn LabelEncoder)
OUTCOME_CLASSES = ["1B", "2B", "3B", "BB", "HR", "K", "OUT"]

# Default estimators that handle nulls
NULL_TOLERANT_ESTIMATORS = ["lgbm", "xgboost", "catboost"]


def prepare_features(
    df: pd.DataFrame,
    preprocessor_path: str = "models/matchup_preprocessor.pkl",
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """
    Prepare features for binary model training/prediction.

    Args:
        df: Raw matchup DataFrame with outcome column
        preprocessor_path: Path to fitted MatchupPreprocessor

    Returns:
        Tuple of (X DataFrame, y array, feature_names, outcome_classes)
    """
    from src.data.preprocess import MatchupPreprocessor

    # Load preprocessor
    preprocessor = MatchupPreprocessor.load(preprocessor_path)

    feature_names = preprocessor.get_feature_names()
    outcome_classes = list(preprocessor.label_encoder.classes_)

    # Get features as DataFrame (tree models handle nulls natively)
    X = df[feature_names].copy()

    # Encode target
    y = preprocessor.label_encoder.transform(df["outcome"])

    return X, y, feature_names, outcome_classes


class BinaryModelEnsemble:
    """
    Ensemble of binary classifiers for matchup outcome prediction.

    Each outcome gets its own model with potentially different features.
    Probabilities are normalized to sum to 1.
    """

    def __init__(
        self,
        time_budget_per_model: int = 120,
        metric: str = "log_loss",
        estimator_list: list[str] | None = None,
        seed: int = 42,
        min_num_leaves: int = 16,
        feature_selection: bool = True,
        feature_selection_threshold: float = 0.2,
    ):
        """
        Initialize ensemble.

        Args:
            time_budget_per_model: Time budget in seconds for each binary model
            metric: Optimization metric for FLAML
            estimator_list: List of estimators to try
            seed: Random seed
            min_num_leaves: Minimum leaves for tree models
            feature_selection: Whether to run feature selection per model
            feature_selection_threshold: Importance threshold for feature selection
        """
        self.time_budget_per_model = time_budget_per_model
        self.metric = metric
        self.estimator_list = estimator_list or NULL_TOLERANT_ESTIMATORS
        self.seed = seed
        self.min_num_leaves = min_num_leaves
        self.feature_selection = feature_selection
        self.feature_selection_threshold = feature_selection_threshold

        self.models: dict[str, AutoML] = {}
        self.feature_names: list[str] = []
        self.selected_features: dict[str, list[str]] = {}
        self.metrics: dict[str, dict] = {}
        self.shap_explainers: dict[str, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        verbose: int = 1,
        save_dir: str | Path | None = None,
        memory_efficient: bool = True,
    ) -> "BinaryModelEnsemble":
        """
        Train binary classifiers for each outcome.

        Args:
            X_train: Training features (DataFrame)
            y_train: Training labels (integer-encoded outcomes)
            X_val: Validation features
            y_val: Validation labels
            verbose: Verbosity level
            save_dir: Directory to save models (required if memory_efficient=True)
            memory_efficient: If True, save each model to disk and clear from memory

        Returns:
            Self for chaining
        """
        import gc

        if memory_efficient and save_dir is None:
            raise ValueError("save_dir required when memory_efficient=True")

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_dir = save_dir

        self.feature_names = list(X_train.columns)

        print(f"Training {len(OUTCOME_CLASSES)} binary classifiers")
        print(f"  Time budget per model: {self.time_budget_per_model}s")
        print(f"  Feature selection: {self.feature_selection}")
        print(f"  Memory efficient: {memory_efficient}")
        print(f"  Train shape: {X_train.shape}")
        print()

        for i, outcome in enumerate(OUTCOME_CLASSES):
            print(f"[{i+1}/{len(OUTCOME_CLASSES)}] Training {outcome} vs not-{outcome}...")

            # Create binary target
            y_binary = (y_train == i).astype(int)
            y_val_binary = (y_val == i).astype(int) if y_val is not None else None

            # Class balance info
            pos_rate = y_binary.mean()
            print(f"  Positive rate: {pos_rate:.1%} ({y_binary.sum():,} / {len(y_binary):,})")

            # Feature selection (optional)
            X_train_model = X_train
            X_val_model = X_val

            if self.feature_selection:
                selected = self._select_features(X_train, y_binary, outcome)
                self.selected_features[outcome] = selected
                X_train_model = X_train[selected]
                X_val_model = X_val[selected] if X_val is not None else None
                print(f"  Selected features: {len(selected)} / {len(self.feature_names)}")
            else:
                self.selected_features[outcome] = self.feature_names

            # Train model
            automl = AutoML()

            automl_settings = {
                "time_budget": self.time_budget_per_model,
                "metric": self.metric,
                "task": "classification",
                "estimator_list": self.estimator_list,
                "seed": self.seed,
                "verbose": max(0, verbose - 1),
                "ensemble": False,
                "early_stop": True,
                "n_jobs": -1,
            }

            # Set minimum leaves
            if self.min_num_leaves > 4:
                from flaml import tune
                automl_settings["custom_hp"] = {
                    "lgbm": {
                        "num_leaves": {
                            "domain": tune.randint(self.min_num_leaves, 256),
                        },
                    },
                    "xgboost": {
                        "max_leaves": {
                            "domain": tune.randint(self.min_num_leaves, 256),
                        },
                    },
                }

            if X_val_model is not None and y_val_binary is not None:
                automl_settings["X_val"] = X_val_model
                automl_settings["y_val"] = y_val_binary

            automl.fit(X_train_model, y_binary, **automl_settings)

            print(f"  Best model: {automl.best_estimator}")
            print(f"  Best {self.metric}: {automl.best_loss:.4f}")

            # Evaluate
            if X_val is not None and y_val is not None:
                metrics = self._evaluate_binary(
                    automl, X_val_model, y_val_binary, outcome
                )
                self.metrics[outcome] = metrics
                print(f"  Val AUC: {metrics['auc']:.3f}, Brier: {metrics['brier']:.4f}")

            # Save and optionally clear from memory
            if save_dir is not None:
                model_path = save_dir / f"model_{outcome}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(automl, f)
                print(f"  Saved to {model_path}")

            if memory_efficient:
                # Don't keep model in memory
                del automl
                gc.collect()
            else:
                self.models[outcome] = automl

            print()

        # Save metadata
        if save_dir is not None:
            self._save_metadata(save_dir)

        return self

    def _save_metadata(self, save_dir: Path):
        """Save feature names and selected features metadata."""
        metadata = {
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'metrics': self.metrics,
            'time_budget_per_model': self.time_budget_per_model,
            'metric': self.metric,
            'estimator_list': self.estimator_list,
            'seed': self.seed,
            'min_num_leaves': self.min_num_leaves,
            'feature_selection': self.feature_selection,
        }
        with open(save_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

    def load_model(self, outcome: str, save_dir: str | Path | None = None) -> AutoML:
        """Load a single model from disk."""
        if save_dir is None:
            save_dir = getattr(self, '_save_dir', None)
        if save_dir is None:
            raise ValueError("No save_dir specified")

        save_dir = Path(save_dir)
        model_path = save_dir / f"model_{outcome}.pkl"
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def _select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        outcome: str,
    ) -> list[str]:
        """
        Select features for a specific outcome using quick LightGBM.

        Args:
            X: Features
            y: Binary target
            outcome: Outcome name (for logging)

        Returns:
            List of selected feature names
        """
        from lightgbm import LGBMClassifier

        # Quick model for feature selection
        model = LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            random_state=self.seed,
            verbose=-1,
        )
        model.fit(X, y)

        # Get feature importances
        importances = model.feature_importances_

        # Select features above threshold (relative importance)
        threshold = self.feature_selection_threshold * importances.max()
        selected_mask = importances >= threshold
        selected_features = [f for f, m in zip(X.columns, selected_mask) if m]

        # Ensure minimum features
        if len(selected_features) < 20:
            # Take top 20 by importance
            top_idx = np.argsort(importances)[-20:]
            selected_features = [X.columns[i] for i in top_idx]

        return selected_features

    def _evaluate_binary(
        self,
        model: AutoML,
        X: pd.DataFrame,
        y: np.ndarray,
        outcome: str,
    ) -> dict:
        """Evaluate a binary model."""
        proba = model.predict_proba(X)[:, 1]
        pred = model.predict(X)

        return {
            "outcome": outcome,
            "auc": roc_auc_score(y, proba),
            "avg_precision": average_precision_score(y, proba),
            "brier": brier_score_loss(y, proba),
            "log_loss": log_loss(y, proba),
            "pos_rate_actual": y.mean(),
            "pos_rate_predicted": proba.mean(),
        }

    def predict_proba(self, X: pd.DataFrame, save_dir: str | Path | None = None) -> np.ndarray:
        """
        Predict outcome probabilities.

        Runs each binary model and normalizes outputs to sum to 1.
        Loads models from disk if not in memory.

        Args:
            X: Features DataFrame
            save_dir: Directory where models are saved (if not in memory)

        Returns:
            Array of shape (n_samples, n_outcomes) with probabilities
        """
        import gc

        if save_dir is None:
            save_dir = getattr(self, '_save_dir', None)

        n_samples = len(X)
        raw_probs = np.zeros((n_samples, len(OUTCOME_CLASSES)))

        for i, outcome in enumerate(OUTCOME_CLASSES):
            # Get model - from memory or disk
            if outcome in self.models:
                model = self.models[outcome]
            elif save_dir is not None:
                model = self.load_model(outcome, save_dir)
            else:
                raise ValueError(f"Model for {outcome} not found in memory or on disk")

            features = self.selected_features[outcome]
            X_model = X[features]

            # Get probability of positive class
            proba = model.predict_proba(X_model)[:, 1]
            raw_probs[:, i] = proba

        # Normalize to sum to 1
        row_sums = raw_probs.sum(axis=1, keepdims=True)
        normalized_probs = raw_probs / row_sums

        return normalized_probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict most likely outcome."""
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def get_shap_explainer(self, outcome: str) -> shap.TreeExplainer:
        """Get or create SHAP explainer for an outcome model."""
        if outcome not in self.shap_explainers:
            model = self.models[outcome].model
            self.shap_explainers[outcome] = shap.TreeExplainer(model)
        return self.shap_explainers[outcome]

    def explain_prediction(
        self,
        X: pd.DataFrame,
        outcome: str,
        idx: int = 0,
    ) -> shap.Explanation:
        """
        Get SHAP explanation for a prediction.

        Args:
            X: Features DataFrame
            outcome: Which outcome model to explain
            idx: Index of sample to explain

        Returns:
            SHAP Explanation object
        """
        features = self.selected_features[outcome]
        X_model = X[features]

        explainer = self.get_shap_explainer(outcome)
        shap_values = explainer(X_model.iloc[[idx]])

        return shap_values

    def plot_feature_importance(
        self,
        outcome: str,
        top_n: int = 20,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot feature importance for an outcome model."""
        model = self.models[outcome].model
        features = self.selected_features[outcome]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError(f"Model for {outcome} doesn't have feature_importances_")

        # Sort by importance
        sorted_idx = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            [features[i] for i in sorted_idx],
            [importances[i] for i in sorted_idx],
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance: {outcome} Model")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_shap_summary(
        self,
        X: pd.DataFrame,
        outcome: str,
        max_display: int = 20,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot SHAP summary for an outcome model."""
        features = self.selected_features[outcome]
        X_model = X[features]

        explainer = self.get_shap_explainer(outcome)
        shap_values = explainer(X_model)

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_model,
            max_display=max_display,
            show=False,
        )
        plt.title(f"SHAP Summary: {outcome} Model")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def get_feature_importance_df(self, outcome: str, save_dir: str | Path | None = None) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if save_dir is None:
            save_dir = getattr(self, '_save_dir', None)

        # Get model from memory or disk
        if outcome in self.models:
            model = self.models[outcome].model
        elif save_dir is not None:
            automl = self.load_model(outcome, save_dir)
            model = automl.model
        else:
            return pd.DataFrame()

        features = self.selected_features[outcome]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': features,
            'importance': importances,
        }).sort_values('importance', ascending=False)

        return df

    @classmethod
    def load(cls, path: str | Path) -> "BinaryModelEnsemble":
        """
        Load ensemble from disk.

        Args:
            path: Either a .pkl file (old format) or a directory (memory-efficient format)
        """
        path = Path(path)

        # Check if it's a directory (memory-efficient format)
        if path.is_dir():
            metadata_path = path / 'metadata.pkl'
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            ensemble = cls(
                time_budget_per_model=metadata['time_budget_per_model'],
                metric=metadata['metric'],
                estimator_list=metadata['estimator_list'],
                seed=metadata['seed'],
                min_num_leaves=metadata['min_num_leaves'],
                feature_selection=metadata['feature_selection'],
            )

            ensemble.feature_names = metadata['feature_names']
            ensemble.selected_features = metadata['selected_features']
            ensemble.metrics = metadata['metrics']
            ensemble._save_dir = path
            # Models loaded on demand via load_model()

            return ensemble

        # Old format: single pickle file
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        ensemble = cls(
            time_budget_per_model=save_dict['time_budget_per_model'],
            metric=save_dict['metric'],
            estimator_list=save_dict['estimator_list'],
            seed=save_dict['seed'],
            min_num_leaves=save_dict['min_num_leaves'],
            feature_selection=save_dict['feature_selection'],
        )

        ensemble.models = save_dict['models']
        ensemble.feature_names = save_dict['feature_names']
        ensemble.selected_features = save_dict['selected_features']
        ensemble.metrics = save_dict['metrics']

        return ensemble

    def summary(self) -> pd.DataFrame:
        """Get summary of all models."""
        rows = []
        for outcome in OUTCOME_CLASSES:
            if outcome in self.models:
                model = self.models[outcome]
                n_features = len(self.selected_features.get(outcome, []))
                metrics = self.metrics.get(outcome, {})

                rows.append({
                    'outcome': outcome,
                    'best_estimator': model.best_estimator,
                    'n_features': n_features,
                    'best_loss': model.best_loss,
                    'val_auc': metrics.get('auc'),
                    'val_brier': metrics.get('brier'),
                })

        return pd.DataFrame(rows)


def compare_with_multiclass(
    binary_ensemble: BinaryModelEnsemble,
    multiclass_trainer,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Compare binary ensemble vs multiclass model predictions.

    Args:
        binary_ensemble: Trained BinaryModelEnsemble
        multiclass_trainer: Trained multiclass model (dict or MatchupModelTrainer)
        X_test: Test features
        y_test: Test labels

    Returns:
        DataFrame comparing metrics
    """
    # Binary ensemble predictions
    binary_proba = binary_ensemble.predict_proba(X_test)

    # Multiclass predictions - handle dict format from pickle
    if isinstance(multiclass_trainer, dict):
        automl = multiclass_trainer['automl']
        multi_proba = automl.predict_proba(X_test)
    else:
        multi_proba = multiclass_trainer.predict_proba(X_test)

    results = []
    for i, outcome in enumerate(OUTCOME_CLASSES):
        y_binary = (y_test == i).astype(int)

        binary_auc = roc_auc_score(y_binary, binary_proba[:, i])
        multi_auc = roc_auc_score(y_binary, multi_proba[:, i])

        binary_brier = brier_score_loss(y_binary, binary_proba[:, i])
        multi_brier = brier_score_loss(y_binary, multi_proba[:, i])

        results.append({
            'outcome': outcome,
            'binary_auc': binary_auc,
            'multi_auc': multi_auc,
            'auc_diff': binary_auc - multi_auc,
            'binary_brier': binary_brier,
            'multi_brier': multi_brier,
            'brier_diff': multi_brier - binary_brier,  # Lower is better
        })

    return pd.DataFrame(results)
