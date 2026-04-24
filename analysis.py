import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

QUANTILES = [0.1, 0.5, 0.9]
TARGET = "los_days"
RANDOM_STATE = 42
N_SPLITS = 5


def load_cohort(path="cohort_preprocessed.parquet"):
    df = pd.read_parquet(path)
    return df


def prepare_features(df):
    drop_cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
                 "admittime", "los_hours", TARGET]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y, feature_cols


def build_quantile_model(quantile, X_train, y_train, X_val, y_val):
    model = LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[
            __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False),
            __import__("lightgbm").log_evaluation(period=-1)
        ]
    )
    return model


def cross_validate_quantile_models(X, y, quantiles=QUANTILES, n_splits=N_SPLITS):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {q: [] for q in quantiles}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for q in quantiles:
            model = build_quantile_model(q, X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)
            pinball = mean_pinball_loss(y_val, preds, alpha=q)
            cv_results[q].append(pinball)
            print(f"Fold {fold+1} | q={q:.1f} | Pinball Loss: {pinball:.4f}")

    print("\nCross-Validation Summary:")
    for q in quantiles:
        scores = cv_results[q]
        print(f"  q={q:.1f}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")

    return cv_results


def train_final_models(X_train, y_train, X_val, y_val, quantiles=QUANTILES):
    models = {}
    for q in quantiles:
        print(f"Training final model for quantile q={q}...")
        model = build_quantile_model(q, X_train, y_train, X_val, y_val)
        models[q] = model
        joblib.dump(model, f"lgbm_q{int(q*100)}.pkl")
        print(f"  Saved to lgbm_q{int(q*100)}.pkl")
    return models


def evaluate_models(models, X_test, y_test):
    results = {}
    for q, model in models.items():
        preds = model.predict(X_test)
        pinball = mean_pinball_loss(y_test, preds, alpha=q)
        mae = mean_absolute_error(y_test, preds)
        results[q] = {"pinball_loss": pinball, "mae": mae}
        print(f"q={q:.1f} | Pinball: {pinball:.4f} | MAE: {mae:.4f}")

    coverage_lower = models[QUANTILES[0]].predict(X_test)
    coverage_upper = models[QUANTILES[-1]].predict(X_test)
    empirical_coverage = np.mean((y_test >= coverage_lower) & (y_test <= coverage_upper))
    nominal_coverage = QUANTILES[-1] - QUANTILES[0]
    print(f"\nInterval Coverage: Empirical={empirical_coverage:.3f}, Nominal={nominal_coverage:.3f}")

    return results


def compute_shap_explanations(model, X_train, X_test, feature_names, quantile_label="q50"):
    print(f"\nComputing SHAP values for quantile model {quantile_label}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop 20 Features by Mean |SHAP|:")
    print(importance_df.head(20).to_string(index=False))
    importance_df.to_csv(f"shap_importance_{quantile_label}.csv", index=False)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"shap_summary_{quantile_label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to shap_summary_{quantile_label}.png")

    return shap_values, importance_df


def run_analysis_pipeline():
    print("Loading preprocessed cohort...")
    df = load_cohort()

    print("Preparing features...")
    X, y, feature_names = prepare_features(df)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape}, Calibration: {X_calib.shape}, Test: {X_test.shape}")

    print("\nRunning cross-validation...")
    cv_results = cross_validate_quantile_models(X_train, y_train)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE
    )
    print("\nTraining final quantile models on full training set...")
    models = train_final_models(X_tr, y_tr, X_val, y_val)

    print("\nEvaluating on test set...")
    evaluate_models(models, X_test, y_test)

    median_model = models[0.5]
    shap_values, importance_df = compute_shap_explanations(
        median_model, X_train, X_test, feature_names, quantile_label="q50"
    )

    X_calib.to_parquet("X_calib.parquet", index=False)
    y_calib.to_frame().to_parquet("y_calib.parquet", index=False)
    X_test.to_parquet("X_test.parquet", index=False)
    y_test.to_frame().to_parquet("y_test.parquet", index=False)
    print("\nCalibration and test sets saved for conformal pipeline.")


if __name__ == "__main__":
    run_analysis_pipeline()
