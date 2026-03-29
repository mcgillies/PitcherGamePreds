"""
Configuration for pitcher strikeout prediction.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data collection and processing configuration."""

    # Directories
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Data collection
    season: int = 2024
    api_delay_seconds: float = 5.0

    # Feature engineering
    rolling_windows: list[int] = field(default_factory=lambda: [3, 5])

    # Preprocessing
    n_pca_components: int = 50
    test_size: float = 0.2
    val_size: float = 0.1


@dataclass
class ModelConfig:
    """Model training configuration."""

    # Architecture
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    learning_rate: float = 0.001

    # Training
    epochs: int = 200
    batch_size: int = 32
    patience: int = 20

    # Paths
    model_dir: str = "models"
    model_name: str = "strikeout_model.keras"
    preprocessor_name: str = "preprocessor.pkl"

    @property
    def model_path(self) -> str:
        return str(Path(self.model_dir) / self.model_name)

    @property
    def preprocessor_path(self) -> str:
        return str(Path(self.model_dir) / self.preprocessor_name)


@dataclass
class Config:
    """Main configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    random_seed: int = 42


# Default configuration
DEFAULT_CONFIG = Config()
