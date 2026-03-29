"""
Model evaluation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import Sequential


def evaluate_model(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    print_results: bool = True,
) -> dict:
    """
    Evaluate model performance on test set.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test targets
        print_results: Whether to print results

    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.predict(X_test, verbose=0).flatten()

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
        'mean_actual': np.mean(y_test),
        'mean_predicted': np.mean(predictions),
        'std_actual': np.std(y_test),
        'std_predicted': np.std(predictions),
    }

    if print_results:
        print("\n" + "=" * 50)
        print("Model Evaluation Results")
        print("=" * 50)
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"MAE:  {metrics['mae']:.3f}")
        print(f"R2:   {metrics['r2']:.3f}")
        print(f"\nMean Actual:    {metrics['mean_actual']:.2f}")
        print(f"Mean Predicted: {metrics['mean_predicted']:.2f}")
        print(f"Std Actual:     {metrics['std_actual']:.2f}")
        print(f"Std Predicted:  {metrics['std_predicted']:.2f}")
        print("=" * 50)

    return metrics


def analyze_predictions(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Create detailed prediction analysis.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        test_df: Optional original DataFrame for adding context

    Returns:
        DataFrame with predictions and errors
    """
    predictions = model.predict(X_test, verbose=0).flatten()

    results = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'error': y_test - predictions,
        'abs_error': np.abs(y_test - predictions),
    })

    # Round predictions for betting context
    results['predicted_rounded'] = np.round(predictions * 2) / 2  # Round to 0.5

    # Classification accuracy (over/under the line)
    # Assuming line is the actual value
    results['direction_correct'] = (
        (results['predicted'] > results['actual'].mean()) ==
        (results['actual'] > results['actual'].mean())
    )

    if test_df is not None:
        # Add context columns if available
        for col in ['Name', 'Opp', 'game_date', 'Team']:
            if col in test_df.columns:
                results[col] = test_df[col].values[:len(results)]

    return results


def evaluate_by_strikeout_range(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate model performance by strikeout range.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        DataFrame with metrics by K range
    """
    predictions = model.predict(X_test, verbose=0).flatten()

    results = []

    ranges = [(0, 3), (4, 5), (6, 7), (8, 10), (11, 20)]

    for low, high in ranges:
        mask = (y_test >= low) & (y_test <= high)
        if mask.sum() == 0:
            continue

        y_sub = y_test[mask]
        pred_sub = predictions[mask]

        results.append({
            'range': f"{low}-{high}",
            'count': mask.sum(),
            'rmse': np.sqrt(mean_squared_error(y_sub, pred_sub)),
            'mae': mean_absolute_error(y_sub, pred_sub),
            'mean_actual': np.mean(y_sub),
            'mean_predicted': np.mean(pred_sub),
        })

    return pd.DataFrame(results)


def betting_simulation(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lines: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Simulate betting performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets (actual strikeouts)
        lines: Betting lines (if None, uses actual values as proxy)
        threshold: Minimum difference from line to place bet

    Returns:
        Dictionary with betting metrics
    """
    predictions = model.predict(X_test, verbose=0).flatten()

    if lines is None:
        # Use mean as proxy for lines
        lines = np.full_like(y_test, np.mean(y_test))

    # Betting decisions
    bet_over = predictions > (lines + threshold)
    bet_under = predictions < (lines - threshold)

    # Results
    over_wins = bet_over & (y_test > lines)
    over_losses = bet_over & (y_test <= lines)
    under_wins = bet_under & (y_test < lines)
    under_losses = bet_under & (y_test >= lines)

    total_bets = bet_over.sum() + bet_under.sum()
    total_wins = over_wins.sum() + under_wins.sum()
    total_losses = over_losses.sum() + under_losses.sum()

    results = {
        'total_bets': int(total_bets),
        'total_wins': int(total_wins),
        'total_losses': int(total_losses),
        'win_rate': total_wins / total_bets if total_bets > 0 else 0,
        'over_bets': int(bet_over.sum()),
        'over_wins': int(over_wins.sum()),
        'under_bets': int(bet_under.sum()),
        'under_wins': int(under_wins.sum()),
        'no_bet': int((~bet_over & ~bet_under).sum()),
    }

    print("\nBetting Simulation Results")
    print("=" * 40)
    print(f"Total bets: {results['total_bets']}")
    print(f"Win rate:   {results['win_rate']:.1%}")
    print(f"Over bets:  {results['over_bets']} ({results['over_wins']} wins)")
    print(f"Under bets: {results['under_bets']} ({results['under_wins']} wins)")
    print(f"No bet:     {results['no_bet']}")
    print("=" * 40)

    return results
