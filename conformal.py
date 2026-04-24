import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

ALPHA = 0.10
NOMINAL_COVERAGE = 1 - ALPHA
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95
RANDOM_STATE = 42


def load_models_and_data():
    lower_model = joblib.load(f"lgbm_q{int(LOWER_QUANTILE*100)}.pkl")
    upper_model = joblib.load(f"lgbm_q{int(UPPER_QUANTILE*100)}.pkl")
    median_model = joblib.load("lgbm_q50.pkl")

    X_calib = pd.read_parquet("X_calib.parquet")
    y_calib = pd.read_parquet("y_calib.parquet").squeeze()
    X_test = pd.read_parquet("X_test.parquet")
    y_test = pd.read_parquet("y_test.parquet").squeeze()

    return lower_model, upper_model, median_model, X_calib, y_calib, X_test, y_test


def compute_nonconformity_scores(lower_model, upper_model, X_calib, y_calib):
    q_low_calib = lower_model.predict(X_calib)
    q_high_calib = upper_model.predict(X_calib)

    scores = np.maximum(q_low_calib - y_calib.values, y_calib.values - q_high_calib)
    return scores, q_low_calib, q_high_calib


def compute_conformal_quantile(scores, alpha=ALPHA):
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q_hat = np.quantile(scores, level)
    return q_hat


def apply_conformal_correction(lower_model, upper_model, X_new, q_hat):
    q_low_raw = lower_model.predict(X_new)
    q_high_raw = upper_model.predict(X_new)

    q_low_conf = q_low_raw - q_hat
    q_high_conf = q_high_raw + q_hat

    q_low_conf = np.clip(q_low_conf, a_min=0, a_max=None)

    return q_low_conf, q_high_conf


def evaluate_conformal_coverage(y_true, lower_bounds, upper_bounds):
    covered = (y_true.values >= lower_bounds) & (y_true.values <= upper_bounds)
    empirical_coverage = covered.mean()
    interval_widths = upper_bounds - lower_bounds
    mean_width = interval_widths.mean()
    median_width = np.median(interval_widths)

    print(f"Nominal Coverage Target : {NOMINAL_COVERAGE:.2%}")
    print(f"Empirical Coverage      : {empirical_coverage:.4%}")
    print(f"Mean Interval Width     : {mean_width:.3f} days")
    print(f"Median Interval Width   : {median_width:.3f} days")
    print(f"Coverage Gap            : {(empirical_coverage - NOMINAL_COVERAGE):+.4%}")

    return {
        "nominal_coverage": NOMINAL_COVERAGE,
        "empirical_coverage": empirical_coverage,
        "mean_interval_width": mean_width,
        "median_interval_width": median_width,
        "coverage_gap": empirical_coverage - NOMINAL_COVERAGE
    }


def compute_conditional_coverage(y_true, lower_bounds, upper_bounds, los_bins=None):
    if los_bins is None:
        los_bins = [0, 1, 3, 7, 14, np.inf]
        los_labels = ["<1d", "1-3d", "3-7d", "7-14d", ">14d"]

    y_true_arr = y_true.values
    covered = (y_true_arr >= lower_bounds) & (y_true_arr <= upper_bounds)
    widths = upper_bounds - lower_bounds
    bin_indices = np.digitize(y_true_arr, los_bins) - 1

    print("\nConditional Coverage by LOS Stratum:")
    print(f"{'Stratum':<12} {'N':>6} {'Coverage':>12} {'Mean Width':>12}")
    print("-" * 45)
    for i, label in enumerate(los_labels):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        cov = covered[mask].mean()
        w = widths[mask].mean()
        print(f"{label:<12} {mask.sum():>6} {cov:>12.4%} {w:>12.3f}")


def compute_winkler_score(y_true, lower_bounds, upper_bounds, alpha=ALPHA):
    widths = upper_bounds - lower_bounds
    penalties = np.where(
        y_true.values < lower_bounds,
        (2 / alpha) * (lower_bounds - y_true.values),
        np.where(
            y_true.values > upper_bounds,
            (2 / alpha) * (y_true.values - upper_bounds),
            0.0
        )
    )
    winkler = widths + penalties
    print(f"\nMean Winkler Score: {winkler.mean():.4f}")
    return winkler.mean()


def plot_conformal_intervals(y_test, lower_conf, upper_conf, median_preds,
                             n_samples=200, save_path="conformal_intervals.png"):
    idx = np.argsort(y_test.values)[:n_samples]
    y_sorted = y_test.values[idx]
    lower_sorted = lower_conf[idx]
    upper_sorted = upper_conf[idx]
    median_sorted = median_preds[idx]
    x = np.arange(n_samples)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x, lower_sorted, upper_sorted, alpha=0.3, color="steelblue",
                    label=f"{NOMINAL_COVERAGE:.0%} Conformal Interval")
    ax.plot(x, y_sorted, "k.", markersize=3, label="Observed LOS", zorder=5)
    ax.plot(x, median_sorted, "r-", linewidth=1.5, label="Median Prediction (q=0.5)", zorder=4)
    ax.set_xlabel("Patient (sorted by observed LOS)")
    ax.set_ylabel("ICU Length of Stay (days)")
    ax.set_title(f"Conformal Quantile Regression — {NOMINAL_COVERAGE:.0%} Prediction Intervals")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_calibration_scores(scores, q_hat, save_path="calibration_scores.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(q_hat, color="crimson", linewidth=2,
               label=f"q_hat = {q_hat:.3f} (1-α={NOMINAL_COVERAGE:.0%})")
    ax.set_xlabel("Nonconformity Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Calibration Nonconformity Scores")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration score plot saved to {save_path}")


def predict_single_patient(lower_model, upper_model, median_model, x_new, q_hat):
    q_low_raw = lower_model.predict(x_new)[0]
    q_high_raw = upper_model.predict(x_new)[0]
    median_pred = median_model.predict(x_new)[0]

    lower_conf = max(0.0, q_low_raw - q_hat)
    upper_conf = q_high_raw + q_hat

    print("\nSingle Patient Prediction:")
    print(f"  Median Predicted LOS  : {median_pred:.2f} days")
    print(f"  Conformal Lower Bound : {lower_conf:.2f} days")
    print(f"  Conformal Upper Bound : {upper_conf:.2f} days")
    print(f"  Interval Width        : {upper_conf - lower_conf:.2f} days")
    print(f"  Coverage Guarantee    : >= {NOMINAL_COVERAGE:.0%} marginal")

    return {
        "median_prediction": median_pred,
        "lower_bound": lower_conf,
        "upper_bound": upper_conf,
        "interval_width": upper_conf - lower_conf
    }


def save_conformal_outputs(X_test, y_test, lower_conf, upper_conf, median_preds,
                           q_hat, metrics, save_path="conformal_predictions.parquet"):
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["median_pred"] = median_preds
    out["lower_bound"] = lower_conf
    out["upper_bound"] = upper_conf
    out["interval_width"] = upper_conf - lower_conf
    out["covered"] = (out["y_true"] >= out["lower_bound"]) & (out["y_true"] <= out["upper_bound"])
    out.to_parquet(save_path, index=False)
    print(f"\nConformal predictions saved to {save_path}")

    metrics_df = pd.DataFrame([{**metrics, "q_hat": q_hat}])
    metrics_df.to_csv("conformal_metrics.csv", index=False)
    print("Metrics saved to conformal_metrics.csv")


def run_conformal_pipeline():
    print("Loading models and data...")
    lower_model, upper_model, median_model, X_calib, y_calib, X_test, y_test = load_models_and_data()

    print("\nComputing nonconformity scores on calibration set...")
    scores, q_low_calib, q_high_calib = compute_nonconformity_scores(lower_model, upper_model, X_calib, y_calib)

    print(f"Score statistics: mean={scores.mean():.3f}, "
          f"std={scores.std():.3f}, "
          f"p95={np.percentile(scores, 95):.3f}")

    print(f"\nComputing conformal correction at alpha={ALPHA}...")
    q_hat = compute_conformal_quantile(scores, alpha=ALPHA)
    print(f"q_hat = {q_hat:.4f}")

    print("\nApplying conformal correction to test set...")
    lower_conf, upper_conf = apply_conformal_correction(lower_model, upper_model, X_test, q_hat)
    median_preds = median_model.predict(X_test)

    print("\nEvaluating coverage and efficiency...")
    metrics = evaluate_conformal_coverage(y_test, lower_conf, upper_conf)

    compute_conditional_coverage(y_test, lower_conf, upper_conf)
    winkler = compute_winkler_score(y_test, lower_conf, upper_conf)

    mae_median = mean_absolute_error(y_test, median_preds)
    print(f"\nMedian Prediction MAE: {mae_median:.4f} days")

    plot_conformal_intervals(y_test, lower_conf, upper_conf, median_preds)
    plot_calibration_scores(scores, q_hat)

    save_conformal_outputs(X_test, y_test, lower_conf, upper_conf,
                           median_preds, q_hat, metrics)

    print("\nConformal prediction pipeline complete.")
    return {
        "q_hat": q_hat,
        "metrics": metrics,
        "winkler_score": winkler
    }


if __name__ == "__main__":
    run_conformal_pipeline()
