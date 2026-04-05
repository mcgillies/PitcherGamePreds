# Model Metrics Guide: Interpreting Log Loss for 7-Class Plate Appearance Prediction

## Overview

This document explains how to interpret log loss (cross-entropy loss) for our 7-class pitcher-batter matchup model predicting plate appearance outcomes.

**Outcome Classes:** 1B, 2B, 3B, BB, HR, K, OUT

## Understanding Log Loss

Log loss measures how well predicted probabilities match actual outcomes. Lower is better.

**Formula:** `log_loss = -1/N * Σ ln(p_predicted for true class)`

- Perfect predictions (100% confidence on correct class): **0.0**
- Completely wrong (0% on correct class): **∞**

## Baseline References

### 1. Random Uniform Guessing
Predicting equal probability (1/7 = 14.3%) for all classes:

```
log_loss = -ln(1/7) = ln(7) ≈ 1.946
```

**Any model scoring above ~1.95 is worse than random guessing.**

### 2. Naive Class Distribution Baseline
Predicting the training set's marginal class distribution for every sample:

| Class | Distribution | -p*ln(p) |
|-------|-------------|----------|
| OUT   | 46.3%       | 0.357    |
| K     | 23.1%       | 0.338    |
| 1B    | 14.3%       | 0.278    |
| BB    | 8.3%        | 0.207    |
| 2B    | 4.4%        | 0.137    |
| HR    | 3.1%        | 0.108    |
| 3B    | 0.4%        | 0.022    |

```
Naive baseline log_loss ≈ 1.45
```

**This is the minimum bar - any useful model must beat ~1.45.**

## Interpreting Your Model's Log Loss

| Log Loss | Interpretation |
|----------|----------------|
| > 1.95   | Worse than random - something is wrong |
| 1.45-1.95 | Worse than naive baseline - model isn't learning |
| 1.35-1.45 | Slightly better than baseline - minimal signal |
| 1.25-1.35 | **Decent** - model has learned useful patterns |
| 1.15-1.25 | **Good** - meaningful improvement over baseline |
| 1.05-1.15 | **Very good** - strong predictive signal |
| < 1.05   | **Excellent** - approaching practical limits |

## Why This Problem is Hard

Plate appearance outcomes have inherent randomness:
- Even the best pitchers give up hits
- Even the worst hitters get lucky
- Single at-bat outcomes are highly stochastic

**Theoretical limits:** Unlike image classification where perfect accuracy is possible, baseball outcomes have irreducible uncertainty. A log loss of ~1.0-1.1 may represent the practical ceiling.

## Converting to Intuitive Metrics

### Average Probability on Correct Class

| Log Loss | Avg P(correct) | Interpretation |
|----------|---------------|----------------|
| 1.95     | 14.3%         | Random (1/7) |
| 1.45     | 23.5%         | Naive baseline |
| 1.30     | 27.3%         | Decent |
| 1.20     | 30.1%         | Good |
| 1.10     | 33.3%         | Very good |
| 1.00     | 36.8%         | Excellent |

### Improvement Over Baseline

Calculate relative improvement:
```
improvement = (baseline_loss - model_loss) / baseline_loss * 100%

Example: (1.45 - 1.25) / 1.45 = 13.8% improvement
```

## Complementary Metrics

Log loss alone doesn't tell the whole story. Also consider:

| Metric | What it measures | Target | Notes |
|--------|------------------|--------|-------|
| **ROC AUC (OvR)** | Ranking quality per class | > 0.60 | Best for probability quality |
| **Calibration** | Do 30% predictions happen 30% of time? | Visual check | Use calibration curves |
| **Brier Score** | Probability accuracy | Lower is better | Similar to log loss |

### Metrics to IGNORE for this problem

| Metric | Why it's misleading |
|--------|---------------------|
| **Accuracy** | Model will just predict OUT/K (70% of data) and look "good" |
| **F1 Macro** | Penalizes rare class predictions even when probabilities are correct |
| **Balanced Accuracy** | Same issue - we care about probability quality, not hard predictions |

**Key insight:** We sum probabilities across plate appearances to get expected stats. A model that outputs P(K)=0.25, P(OUT)=0.45, P(1B)=0.15... is useful even if its "prediction" (argmax) is always OUT.

## Per-Class Expectations

Some classes are harder to predict than others:

| Class | Base Rate | Difficulty | Notes |
|-------|-----------|------------|-------|
| OUT   | 46.3%     | Easiest    | Most common, model will favor this |
| K     | 23.1%     | Medium     | Good signal from pitcher/batter K rates |
| 1B    | 14.3%     | Medium     | Contact-based, harder to predict |
| BB    | 8.3%      | Medium     | Good signal from BB rates |
| 2B    | 4.4%      | Hard       | Rare, requires lucky placement |
| HR    | 3.1%      | Hard       | Rare but has good predictive features |
| 3B    | 0.4%      | Very Hard  | Extremely rare, expect poor recall |

## Practical Recommendations

1. **Start with baselines:** Before training, compute naive baseline log loss (~1.45)

2. **Set realistic targets:**
   - Initial goal: Beat 1.40
   - Good model: Below 1.30
   - Excellent: Below 1.15

3. **Watch for overfitting:**
   - Train log loss << Val/Test log loss = overfitting
   - Gap > 0.05 is concerning

4. **Don't over-optimize log loss:**
   - Sometimes accuracy or F1 matters more for downstream use
   - Calibrated probabilities (good log loss) are valuable for betting/DFS

## Example Interpretation

```
Model Results:
  Train log_loss: 1.18
  Val log_loss:   1.24  
  Test log_loss:  1.25

Interpretation:
- Model beats naive baseline (1.45) by 14%
- Slight overfitting (train-val gap = 0.06)
- Test performance is "Good" tier
- Average ~29% confidence on correct class
```

## References

- [Scikit-learn log_loss documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
- Cross-entropy is equivalent to log loss for classification
- Also called "negative log likelihood" in some contexts
