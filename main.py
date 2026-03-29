"""
Main entry point for pitcher strikeout prediction.

Usage:
    # Collect data and train model
    python main.py train --season 2024

    # Make predictions
    python main.py predict --pitcher "Gerrit Cole" --opponent "Boston"

    # Evaluate existing model
    python main.py evaluate
"""

import argparse
from datetime import datetime
from pathlib import Path

from config import DEFAULT_CONFIG, Config


def collect_data(config: Config) -> None:
    """Collect raw data from pybaseball."""
    from src.data.collect import collect_all_data

    print(f"\n{'='*60}")
    print(f"Collecting data for {config.data.season} season")
    print(f"{'='*60}\n")

    collect_all_data(
        season=config.data.season,
        output_dir=config.data.raw_data_dir,
    )


def build_features(config: Config) -> None:
    """Build features from raw data."""
    import pandas as pd

    from src.data.features import build_features as _build_features

    print(f"\n{'='*60}")
    print("Building features")
    print(f"{'='*60}\n")

    raw_dir = Path(config.data.raw_data_dir)
    processed_dir = Path(config.data.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    pitcher_games = pd.read_csv(raw_dir / "pitcher_games.csv")
    team_batting = pd.read_csv(raw_dir / "team_batting.csv")

    # Build features
    features_df = _build_features(
        pitcher_games=pitcher_games,
        team_batting=team_batting,
        rolling_windows=config.data.rolling_windows,
    )

    # Save
    output_path = processed_dir / "features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"\nSaved features to {output_path}")
    print(f"Shape: {features_df.shape}")


def train_model(config: Config) -> None:
    """Train the strikeout prediction model."""
    import pandas as pd

    from src.data.preprocess import preprocess_data
    from src.model.evaluate import evaluate_model
    from src.model.train import save_model
    from src.model.train import train_model as _train_model

    print(f"\n{'='*60}")
    print("Training model")
    print(f"{'='*60}\n")

    # Load features
    features_path = Path(config.data.processed_data_dir) / "features.csv"
    if not features_path.exists():
        print("Features not found. Building features first...")
        build_features(config)

    features_df = pd.read_csv(features_path)

    # Preprocess
    preprocessor, X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        features_df,
        n_pca_components=config.data.n_pca_components,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        save_path=config.model.preprocessor_path,
    )

    # Train
    model, history = _train_model(
        X_train, y_train,
        X_val, y_val,
        hidden_layers=config.model.hidden_layers,
        dropout_rate=config.model.dropout_rate,
        learning_rate=config.model.learning_rate,
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        patience=config.model.patience,
    )

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save
    save_model(model, config.model.model_path)


def evaluate(config: Config) -> None:
    """Evaluate the trained model."""
    import pandas as pd
    from tensorflow import keras

    from src.data.preprocess import DataPreprocessor
    from src.model.evaluate import evaluate_by_strikeout_range, evaluate_model

    print(f"\n{'='*60}")
    print("Evaluating model")
    print(f"{'='*60}\n")

    # Load model and preprocessor
    model = keras.models.load_model(config.model.model_path)
    preprocessor = DataPreprocessor.load(config.model.preprocessor_path)

    # Load features
    features_df = pd.read_csv(Path(config.data.processed_data_dir) / "features.csv")

    # Get test set
    from sklearn.model_selection import train_test_split

    _, test_df = train_test_split(
        features_df,
        test_size=config.data.test_size,
        random_state=config.random_seed,
    )

    X_test = preprocessor.transform(test_df)
    y_test = test_df['K_actual'].values

    # Evaluate
    evaluate_model(model, X_test, y_test)

    print("\nPerformance by strikeout range:")
    range_results = evaluate_by_strikeout_range(model, X_test, y_test)
    print(range_results.to_string(index=False))


def predict(config: Config, pitcher: str, opponent: str, is_home: bool = True) -> None:
    """Make a prediction for a single game."""
    from src.predict import StrikeoutPredictor

    print(f"\n{'='*60}")
    print(f"Predicting strikeouts")
    print(f"{'='*60}\n")

    predictor = StrikeoutPredictor(
        model_path=config.model.model_path,
        preprocessor_path=config.model.preprocessor_path,
    )

    predictor.load_current_data(season=config.data.season)

    prediction = predictor.predict(pitcher, opponent, is_home)

    print(f"\nPitcher:    {pitcher}")
    print(f"Opponent:   {opponent}")
    print(f"Home:       {'Yes' if is_home else 'No'}")
    print(f"\nPredicted strikeouts: {prediction:.2f}")
    print(f"Rounded (0.5):        {round(prediction * 2) / 2}")


def main():
    parser = argparse.ArgumentParser(description="Pitcher Strikeout Prediction")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--season", type=int, default=2024, help="MLB season")
    train_parser.add_argument("--skip-collect", action="store_true", help="Skip data collection")
    train_parser.add_argument("--skip-features", action="store_true", help="Skip feature building")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--pitcher", required=True, help="Pitcher name")
    predict_parser.add_argument("--opponent", required=True, help="Opponent team abbreviation (e.g., BOS, NYY, LAD)")
    predict_parser.add_argument("--away", action="store_true", help="Pitcher is away")
    predict_parser.add_argument("--season", type=int, default=2024, help="MLB season")

    # Evaluate command
    subparsers.add_parser("evaluate", help="Evaluate the model")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect data only")
    collect_parser.add_argument("--season", type=int, default=2024, help="MLB season")

    # Features command
    subparsers.add_parser("features", help="Build features only")

    # Teams command
    subparsers.add_parser("teams", help="List all team abbreviations")

    args = parser.parse_args()

    # Load config
    config = DEFAULT_CONFIG

    if args.command == "train":
        config.data.season = args.season
        if not args.skip_collect:
            collect_data(config)
        if not args.skip_features:
            build_features(config)
        train_model(config)

    elif args.command == "predict":
        config.data.season = args.season
        predict(config, args.pitcher, args.opponent, not args.away)

    elif args.command == "evaluate":
        evaluate(config)

    elif args.command == "collect":
        config.data.season = args.season
        collect_data(config)

    elif args.command == "features":
        build_features(config)

    elif args.command == "teams":
        from src.data.collect import list_teams
        list_teams()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
