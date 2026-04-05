"""
FLAML AutoML training for pitcher-batter matchup prediction.

Uses tree-based models that handle nulls natively:
- LightGBM
- XGBoost
- CatBoost

Provides:
- Automated hyperparameter tuning
- Feature importance analysis
- SHAP explanations for individual predictions
- Comprehensive evaluation metrics
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from flaml import AutoML
import shap


# Estimators that handle nulls natively
NULL_TOLERANT_ESTIMATORS = ["lgbm", "xgboost", "catboost"]

# Outcome classes (alphabetical order - matches sklearn LabelEncoder)
OUTCOME_CLASSES = ["1B", "2B", "3B", "BB", "HR", "K", "OUT"]


class MatchupModelTrainer:
    """
    FLAML-based trainer for pitcher-batter matchup prediction.

    Attributes:
        automl: FLAML AutoML instance
        feature_names: List of feature column names
        outcome_classes: List of outcome class labels
        best_model: Best model from AutoML search
        metrics: Dictionary of evaluation metrics
    """

    def __init__(
        self,
        time_budget: int = 600,
        metric: str = "log_loss",
        estimator_list: list[str] | None = None,
        seed: int = 42,
        min_num_leaves: int = 4,
    ):
        """
        Initialize trainer.

        Args:
            time_budget: Time budget in seconds for AutoML search
            metric: Optimization metric (log_loss, accuracy, f1, etc.)
            estimator_list: List of estimators to try (defaults to null-tolerant)
            seed: Random seed for reproducibility
            min_num_leaves: Minimum num_leaves for tree models (higher = more complex, default 4)
                           Set to 32+ to avoid regression-to-mean for extreme values
        """
        self.time_budget = time_budget
        self.metric = metric
        self.estimator_list = estimator_list or NULL_TOLERANT_ESTIMATORS
        self.seed = seed
        self.min_num_leaves = min_num_leaves

        self.automl = None
        self.feature_names = []
        self.outcome_classes = OUTCOME_CLASSES
        self.best_model = None
        self.metrics = {}
        self.shap_explainer = None

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        verbose: int = 1,
    ) -> "MatchupModelTrainer":
        """
        Train model using FLAML AutoML.

        Args:
            X_train: Training features
            y_train: Training labels (encoded integers)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of feature columns
            verbose: Verbosity level

        Returns:
            Self for chaining
        """
        if feature_names:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Convert to DataFrame for FLAML (preserves NaN handling)
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=self.feature_names)
        if X_val is not None and isinstance(X_val, np.ndarray):
            X_val = pd.DataFrame(X_val, columns=self.feature_names)

        print(f"Training with FLAML AutoML")
        print(f"  Time budget: {self.time_budget}s")
        print(f"  Metric: {self.metric}")
        print(f"  Estimators: {self.estimator_list}")
        print(f"  Train shape: {X_train.shape}")
        if X_val is not None:
            print(f"  Val shape: {X_val.shape}")
        print(f"  Null values in train: {X_train.isnull().sum().sum():,}")

        self.automl = AutoML()

        automl_settings = {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "task": "classification",
            "estimator_list": self.estimator_list,
            "log_file_name": "flaml_training.log",
            "seed": self.seed,
            "verbose": verbose,
            "ensemble": False,  # Keep single best model for interpretability
            "early_stop": True,
            "n_jobs": -1,
        }

        # Set minimum num_leaves to avoid regression-to-mean
        if self.min_num_leaves > 4:
            from flaml import tune
            print(f"  Min num_leaves: {self.min_num_leaves}")
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

        # Add validation set if provided
        if X_val is not None and y_val is not None:
            automl_settings["X_val"] = X_val
            automl_settings["y_val"] = y_val

        self.automl.fit(X_train, y_train, **automl_settings)

        self.best_model = self.automl.model

        print(f"\nBest model: {self.automl.best_estimator}")
        print(f"Best {self.metric}: {self.automl.best_loss:.4f}")

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.automl.predict(X)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.automl.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        set_name: str = "test",
    ) -> dict[str, Any]:
        """
        Comprehensive evaluation with multiple metrics.

        Args:
            X: Features
            y: True labels
            set_name: Name for this evaluation set

        Returns:
            Dictionary of metrics
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            "set_name": set_name,
            "n_samples": len(y),
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "log_loss": log_loss(y, y_proba),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        }

        # Per-class metrics
        report = classification_report(
            y, y_pred,
            target_names=self.outcome_classes,
            output_dict=True,
            zero_division=0,
        )
        metrics["classification_report"] = report

        # ROC AUC (one-vs-rest)
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y, y_proba, multi_class="ovr", average="weighted"
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y, y_pred)

        self.metrics[set_name] = metrics
        return metrics

    def print_evaluation(self, metrics: dict[str, Any]) -> None:
        """Pretty print evaluation metrics."""
        print(f"\n{'='*60}")
        print(f"Evaluation: {metrics['set_name']} ({metrics['n_samples']:,} samples)")
        print(f"{'='*60}")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"Log Loss:           {metrics['log_loss']:.4f}")
        print(f"F1 (macro):         {metrics['f1_macro']:.4f}")
        print(f"F1 (weighted):      {metrics['f1_weighted']:.4f}")
        if metrics.get("roc_auc_ovr"):
            print(f"ROC AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")

        print(f"\nPer-class F1 scores:")
        for cls in self.outcome_classes:
            if cls in metrics["classification_report"]:
                f1 = metrics["classification_report"][cls]["f1-score"]
                support = metrics["classification_report"][cls]["support"]
                print(f"  {cls:4s}: {f1:.4f} (n={support:,})")

    def plot_confusion_matrix(
        self,
        metrics: dict[str, Any],
        normalize: bool = True,
        figsize: tuple = (10, 8),
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot confusion matrix heatmap."""
        cm = metrics["confusion_matrix"]

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            fmt = ".2%"
        else:
            fmt = "d"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=self.outcome_classes,
            yticklabels=self.outcome_classes,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {metrics['set_name']}")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved confusion matrix to {save_path}")

        return fig

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Get feature importance from the best model.

        Args:
            importance_type: Type of importance (gain, split, weight)

        Returns:
            DataFrame with feature names and importance scores
        """
        # Get underlying model (FLAML wraps models)
        model = self._get_underlying_model()
        n_features = len(self.feature_names)

        # Handle different model types
        if hasattr(model, "booster_"):
            # LightGBM
            importances = model.booster_.feature_importance(importance_type=importance_type)
        elif hasattr(model, "get_booster"):
            # XGBoost
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)
            importances = np.array([
                importance_dict.get(f"f{i}", 0) for i in range(n_features)
            ])
        elif hasattr(model, "feature_importances_"):
            # Sklearn-style (CatBoost, etc.)
            importances = model.feature_importances_
        else:
            raise ValueError(f"Cannot extract feature importance from {type(model)}")

        # Ensure length matches
        if len(importances) != n_features:
            # Some models may have different feature indexing
            importances = np.zeros(n_features)
            if hasattr(model, "feature_importances_"):
                importances[:len(model.feature_importances_)] = model.feature_importances_

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
        df["cumulative_pct"] = df["importance_pct"].cumsum()

        return df

    def plot_feature_importance(
        self,
        top_n: int = 30,
        figsize: tuple = (12, 10),
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot top feature importances."""
        importance_df = self.get_feature_importance()
        top_df = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_df)))
        ax.barh(range(len(top_df)), top_df["importance"].values, color=colors)
        ax.set_yticks(range(len(top_df)))
        ax.set_yticklabels(top_df["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (Gain)")
        ax.set_title(f"Top {top_n} Feature Importances")

        # Add percentage annotations
        for i, (imp, pct) in enumerate(zip(top_df["importance"], top_df["importance_pct"])):
            ax.text(imp + 0.01 * top_df["importance"].max(), i, f"{pct:.1f}%", va="center")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved feature importance plot to {save_path}")

        return fig

    def _get_underlying_model(self):
        """Extract the underlying sklearn/native model from FLAML wrapper."""
        model = self.best_model

        # FLAML wraps models - extract the underlying estimator
        if hasattr(model, "model"):
            # FLAML estimator wrapper
            return model.model
        elif hasattr(model, "_model"):
            return model._model
        elif hasattr(model, "estimator"):
            return model.estimator
        else:
            return model

    def init_shap_explainer(
        self,
        X_background: np.ndarray | pd.DataFrame,
        max_samples: int = 1000,
    ) -> None:
        """
        Initialize SHAP explainer with background data.

        Args:
            X_background: Background dataset for SHAP
            max_samples: Max samples for background (for speed)
        """
        if isinstance(X_background, np.ndarray):
            X_background = pd.DataFrame(X_background, columns=self.feature_names)

        # Subsample for speed
        if len(X_background) > max_samples:
            X_background = X_background.sample(max_samples, random_state=self.seed)

        print(f"Initializing SHAP explainer with {len(X_background)} background samples...")

        # Get underlying model (FLAML wraps models)
        underlying_model = self._get_underlying_model()

        # Use TreeExplainer for tree models
        self.shap_explainer = shap.TreeExplainer(underlying_model, X_background)
        print("SHAP explainer ready.")

    def explain_prediction(
        self,
        X: np.ndarray | pd.DataFrame,
        index: int = 0,
        class_idx: int | None = None,
        plot_type: str = "waterfall",
        figsize: tuple = (12, 8),
        save_path: str | None = None,
    ) -> tuple[shap.Explanation, plt.Figure | None]:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            X: Features (single row or batch)
            index: Index of sample to explain
            class_idx: Class index for waterfall (defaults to predicted class)
            plot_type: Type of plot (waterfall, force, bar)
            figsize: Figure size
            save_path: Path to save plot

        Returns:
            Tuple of (SHAP values, figure)
        """
        if self.shap_explainer is None:
            raise ValueError("Call init_shap_explainer() first")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Get single sample
        if len(X) > 1:
            X_single = X.iloc[[index]]
        else:
            X_single = X

        # Compute SHAP values
        shap_values = self.shap_explainer(X_single)

        # Get predicted class if not specified
        if class_idx is None:
            pred = self.predict(X_single)[0]
            class_idx = pred

        fig = None
        if plot_type == "waterfall":
            fig = plt.figure(figsize=figsize)
            # For multiclass, select the class
            shap.waterfall_plot(
                shap_values[0, :, class_idx],
                max_display=15,
                show=False,
            )
            plt.title(f"SHAP Waterfall - Predicted: {self.outcome_classes[class_idx]}")

        elif plot_type == "force":
            shap.initjs()
            fig = shap.force_plot(
                shap_values[0, :, class_idx],
                matplotlib=True,
                figsize=figsize,
                show=False,
            )

        elif plot_type == "bar":
            fig = plt.figure(figsize=figsize)
            shap.bar_plot(shap_values[0, :, class_idx], max_display=15, show=False)

        if save_path and fig is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved SHAP plot to {save_path}")

        return shap_values, fig

    def plot_shap_summary(
        self,
        X: np.ndarray | pd.DataFrame,
        max_samples: int = 1000,
        class_idx: int | None = None,
        figsize: tuple = (12, 10),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot SHAP summary (beeswarm) for feature impact.

        Args:
            X: Features
            max_samples: Max samples for SHAP computation
            class_idx: Class index for summary (None for global)
            figsize: Figure size
            save_path: Path to save plot
        """
        if self.shap_explainer is None:
            raise ValueError("Call init_shap_explainer() first")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Subsample for speed
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=self.seed)

        print(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.shap_explainer(X)

        fig = plt.figure(figsize=figsize)

        if class_idx is not None:
            shap.summary_plot(
                shap_values[:, :, class_idx],
                X,
                max_display=20,
                show=False,
            )
            plt.title(f"SHAP Summary - {self.outcome_classes[class_idx]}")
        else:
            # Average absolute SHAP across classes
            shap_abs_mean = np.abs(shap_values.values).mean(axis=(0, 2))
            feature_order = np.argsort(shap_abs_mean)[::-1][:20]

            shap.summary_plot(
                shap_values[:, feature_order, :].values.mean(axis=2),
                X.iloc[:, feature_order],
                max_display=20,
                show=False,
            )
            plt.title("SHAP Summary - All Classes (mean)")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved SHAP summary to {save_path}")

        return fig

    def save(self, path: str) -> None:
        """Save trainer state to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "automl": self.automl,
            "feature_names": self.feature_names,
            "outcome_classes": self.outcome_classes,
            "metrics": self.metrics,
            "time_budget": self.time_budget,
            "metric": self.metric,
            "estimator_list": self.estimator_list,
            "seed": self.seed,
            "min_num_leaves": self.min_num_leaves,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        print(f"Saved trainer to {save_path}")

    @classmethod
    def load(cls, path: str) -> "MatchupModelTrainer":
        """Load trainer from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        trainer = cls(
            time_budget=state["time_budget"],
            metric=state["metric"],
            estimator_list=state["estimator_list"],
            seed=state["seed"],
            min_num_leaves=state.get("min_num_leaves", 4),  # Default for old models
        )
        trainer.automl = state["automl"]
        trainer.best_model = trainer.automl.model
        trainer.feature_names = state["feature_names"]
        trainer.outcome_classes = state["outcome_classes"]
        trainer.metrics = state["metrics"]

        return trainer

    def save_metrics_report(self, path: str) -> None:
        """Save detailed metrics report to JSON."""
        report = {
            "model_type": self.automl.best_estimator,
            "training_time": self.automl.time_to_best if hasattr(self.automl, "time_to_best") else None,
            "best_config": self.automl.best_config,
            "metrics": {},
        }

        for set_name, metrics in self.metrics.items():
            # Convert numpy arrays to lists for JSON
            metrics_clean = {}
            for k, v in metrics.items():
                if isinstance(v, np.ndarray):
                    metrics_clean[k] = v.tolist()
                elif isinstance(v, (np.float64, np.int64)):
                    metrics_clean[k] = float(v)
                else:
                    metrics_clean[k] = v
            report["metrics"][set_name] = metrics_clean

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Saved metrics report to {path}")


def load_processed_data(
    data_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load processed train/val/test data.

    Returns:
        Tuple of (train_df, val_df, test_df, feature_names)
    """
    data_path = Path(data_dir)

    train = pd.read_parquet(data_path / "train.parquet")
    val = pd.read_parquet(data_path / "val.parquet")
    test = pd.read_parquet(data_path / "test.parquet")

    print(f"Loaded data:")
    print(f"  Train: {len(train):,} rows, {train['game_date'].min()} to {train['game_date'].max()}")
    print(f"  Val:   {len(val):,} rows, {val['game_date'].min()} to {val['game_date'].max()}")
    print(f"  Test:  {len(test):,} rows, {test['game_date'].min()} to {test['game_date'].max()}")

    return train, val, test


def prepare_features(
    df: pd.DataFrame,
    preprocessor_path: str = "models/matchup_preprocessor.pkl",
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """
    Prepare features for training/prediction.

    Args:
        df: Raw matchup DataFrame
        preprocessor_path: Path to fitted preprocessor

    Returns:
        Tuple of (X DataFrame, y array, feature_names, outcome_classes)
    """
    from src.data.preprocess import MatchupPreprocessor

    # Load preprocessor
    preprocessor = MatchupPreprocessor.load(preprocessor_path)

    feature_names = preprocessor.get_feature_names()
    outcome_classes = list(preprocessor.label_encoder.classes_)

    # Get features (no scaling for tree models - they don't need it)
    X = df[feature_names].copy()

    # Encode target
    y = preprocessor.label_encoder.transform(df["outcome"])

    return X, y, feature_names, outcome_classes


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train FLAML matchup model")
    parser.add_argument("--time-budget", type=int, default=600, help="Time budget in seconds")
    parser.add_argument("--metric", type=str, default="log_loss", help="Optimization metric")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()

    # Load data
    train, val, test = load_processed_data()

    # Prepare features
    X_train, y_train, feature_names = prepare_features(train)
    X_val, y_val, _ = prepare_features(val)
    X_test, y_test, _ = prepare_features(test)

    print(f"\nFeatures: {len(feature_names)}")
    print(f"Null values: {X_train.isnull().sum().sum():,} in train")

    # Train
    trainer = MatchupModelTrainer(
        time_budget=args.time_budget,
        metric=args.metric,
    )
    trainer.fit(X_train, y_train, X_val, y_val, feature_names)

    # Evaluate
    train_metrics = trainer.evaluate(X_train, y_train, "train")
    val_metrics = trainer.evaluate(X_val, y_val, "val")
    test_metrics = trainer.evaluate(X_test, y_test, "test")

    trainer.print_evaluation(train_metrics)
    trainer.print_evaluation(val_metrics)
    trainer.print_evaluation(test_metrics)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save(output_dir / "flaml_trainer.pkl")
    trainer.save_metrics_report(output_dir / "metrics_report.json")

    # Feature importance
    importance_df = trainer.get_feature_importance()
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    trainer.plot_feature_importance(save_path=str(output_dir / "feature_importance.png"))

    # Confusion matrix
    trainer.plot_confusion_matrix(test_metrics, save_path=str(output_dir / "confusion_matrix.png"))

    print("\nTraining complete!")
