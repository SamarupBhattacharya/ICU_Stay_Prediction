import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

MIMIC_PATH = "/path/to/mimic-iv/"
FIRST_N_HOURS = 24


def load_table(name, module="hosp"):
    return pd.read_csv(f"{MIMIC_PATH}{module}/{name}.csv.gz", compression="gzip", low_memory=False)


def compute_icu_los(icustays):
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    icustays["los_hours"] = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600
    icustays["los_days"] = icustays["los_hours"] / 24
    return icustays[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los_hours", "los_days"]]


def extract_demographics(patients, admissions):
    patients["anchor_year"] = patients["anchor_year"].astype(int)
    patients["anchor_age"] = patients["anchor_age"].astype(int)

    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["admit_year"] = admissions["admittime"].dt.year

    demo = admissions.merge(patients[["subject_id", "anchor_year", "anchor_age",
                                      "anchor_year_group", "gender", "dod"]], on="subject_id", how="left")

    demo["age_at_admission"] = demo["anchor_age"] + (demo["admit_year"] - demo["anchor_year"])
    demo["age_at_admission"] = demo["age_at_admission"].clip(lower=0, upper=120)

    le_gender = LabelEncoder()
    demo["gender_enc"] = le_gender.fit_transform(demo["gender"].fillna("Unknown"))

    le_insurance = LabelEncoder()
    demo["insurance_enc"] = le_insurance.fit_transform(demo["insurance"].fillna("Unknown"))

    le_lang = LabelEncoder()
    demo["language_enc"] = le_lang.fit_transform(demo["language"].fillna("Unknown"))

    le_marital = LabelEncoder()
    demo["marital_enc"] = le_marital.fit_transform(demo["marital_status"].fillna("Unknown"))

    le_ethnicity = LabelEncoder()
    demo["ethnicity_enc"] = le_ethnicity.fit_transform(demo["race"].fillna("Unknown"))

    keep_cols = ["subject_id", "hadm_id", "admittime", "age_at_admission",
                 "gender_enc", "insurance_enc", "language_enc", "marital_enc", "ethnicity_enc",
                 "admission_type", "admission_location"]
    return demo[keep_cols]


def extract_omr_baseline(omr, icustays):
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])
    icustays_dates = icustays[["subject_id", "stay_id", "intime"]].copy()
    icustays_dates["intime"] = pd.to_datetime(icustays_dates["intime"])

    omr_merged = omr.merge(icustays_dates, on="subject_id", how="inner")
    omr_merged = omr_merged[omr_merged["chartdate"] <= omr_merged["intime"]]

    omr_merged["days_before_icu"] = (omr_merged["intime"] - omr_merged["chartdate"]).dt.days
    omr_closest = omr_merged.sort_values("days_before_icu").groupby(["subject_id", "stay_id", "result_name"]).first().reset_index()

    omr_pivot = omr_closest.pivot_table(index=["subject_id", "stay_id"],
                                        columns="result_name",
                                        values="result_value",
                                        aggfunc="first").reset_index()

    rename_map = {}
    for col in omr_pivot.columns:
        if "Height" in str(col):
            rename_map[col] = "height_cm"
        elif "Weight" in str(col):
            rename_map[col] = "weight_kg"
        elif "BMI" in str(col):
            rename_map[col] = "bmi"
        elif "Blood Pressure" in str(col):
            rename_map[col] = "baseline_bp"
        elif "eGFR" in str(col):
            rename_map[col] = "baseline_egfr"

    omr_pivot.rename(columns=rename_map, inplace=True)

    for col in ["height_cm", "weight_kg", "bmi", "baseline_egfr"]:
        if col in omr_pivot.columns:
            omr_pivot[col] = pd.to_numeric(omr_pivot[col], errors="coerce")

    return omr_pivot


def extract_lab_features(labevents, icustays, itemids_of_interest=None):
    labevents["charttime"] = pd.to_datetime(labevents["charttime"])
    icustays["intime"] = pd.to_datetime(icustays["intime"])

    lab_icu = labevents.merge(icustays[["subject_id", "hadm_id", "stay_id", "intime"]],
                              on=["subject_id", "hadm_id"], how="inner")

    lab_icu["hours_from_admit"] = (lab_icu["charttime"] - lab_icu["intime"]).dt.total_seconds() / 3600
    lab_window = lab_icu[(lab_icu["hours_from_admit"] >= 0) & (lab_icu["hours_from_admit"] <= FIRST_N_HOURS)]

    if itemids_of_interest:
        lab_window = lab_window[lab_window["itemid"].isin(itemids_of_interest)]

    lab_window["valuenum"] = pd.to_numeric(lab_window["valuenum"], errors="coerce")

    lab_agg = lab_window.groupby(["stay_id", "itemid"])["valuenum"].agg(["mean", "min", "max", "std"]).reset_index()
    lab_agg.columns = ["stay_id", "itemid", "lab_mean", "lab_min", "lab_max", "lab_std"]

    lab_pivot = lab_agg.pivot_table(index="stay_id", columns="itemid",
                                    values=["lab_mean", "lab_min", "lab_max", "lab_std"])
    lab_pivot.columns = [f"lab_{stat}_{iid}" for stat, iid in lab_pivot.columns]
    lab_pivot.reset_index(inplace=True)

    return lab_pivot


def extract_chart_features(chartevents, icustays, vital_itemids=None):
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
    icustays["intime"] = pd.to_datetime(icustays["intime"])

    chart_icu = chartevents.merge(icustays[["stay_id", "intime"]], on="stay_id", how="inner")
    chart_icu["hours_from_admit"] = (chart_icu["charttime"] - chart_icu["intime"]).dt.total_seconds() / 3600
    chart_window = chart_icu[(chart_icu["hours_from_admit"] >= 0) & (chart_icu["hours_from_admit"] <= FIRST_N_HOURS)]

    if vital_itemids:
        chart_window = chart_window[chart_window["itemid"].isin(vital_itemids)]

    chart_window["valuenum"] = pd.to_numeric(chart_window["valuenum"], errors="coerce")

    chart_agg = chart_window.groupby(["stay_id", "itemid"])["valuenum"].agg(["mean", "min", "max", "std"]).reset_index()
    chart_agg.columns = ["stay_id", "itemid", "vital_mean", "vital_min", "vital_max", "vital_std"]

    chart_pivot = chart_agg.pivot_table(index="stay_id", columns="itemid",
                                        values=["vital_mean", "vital_min", "vital_max", "vital_std"])
    chart_pivot.columns = [f"vital_{stat}_{iid}" for stat, iid in chart_pivot.columns]
    chart_pivot.reset_index(inplace=True)

    return chart_pivot


def extract_fluid_features(inputevents, icustays):
    inputevents["starttime"] = pd.to_datetime(inputevents["starttime"])
    icustays["intime"] = pd.to_datetime(icustays["intime"])

    fluid_icu = inputevents.merge(icustays[["stay_id", "intime"]], on="stay_id", how="inner")
    fluid_icu["hours_from_admit"] = (fluid_icu["starttime"] - fluid_icu["intime"]).dt.total_seconds() / 3600
    fluid_window = fluid_icu[(fluid_icu["hours_from_admit"] >= 0) & (fluid_icu["hours_from_admit"] <= FIRST_N_HOURS)]

    fluid_window["amount"] = pd.to_numeric(fluid_window["amount"], errors="coerce")

    fluid_agg = fluid_window.groupby("stay_id").agg(
        total_fluid_input=("amount", "sum"),
        num_fluid_events=("amount", "count"),
        mean_fluid_rate=("rate", "mean")
    ).reset_index()

    return fluid_agg


def apply_mice_imputation(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    id_cols = [c for c in exclude_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in id_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    imputer = IterativeImputer(max_iter=10, random_state=42, min_value=0)
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df_imputed


def build_master_cohort():
    print("Loading tables...")
    icustays = load_table("icustays", module="icu")
    patients = load_table("patients")
    admissions = load_table("admissions")
    omr = load_table("omr")
    labevents = load_table("labevents")
    chartevents = load_table("chartevents", module="icu")
    inputevents = load_table("inputevents", module="icu")

    print("Computing ICU LOS targets...")
    los = compute_icu_los(icustays)

    print("Extracting demographics...")
    demo = extract_demographics(patients, admissions)

    print("Extracting OMR baseline physiology...")
    omr_features = extract_omr_baseline(omr, icustays)

    print("Extracting lab features (first 24h)...")
    lab_features = extract_lab_features(labevents, icustays)

    print("Extracting chart/vital features (first 24h)...")
    chart_features = extract_chart_features(chartevents, icustays)

    print("Extracting fluid input features (first 24h)...")
    fluid_features = extract_fluid_features(inputevents, icustays)

    print("Merging all feature sets...")
    cohort = los.merge(demo, on=["subject_id", "hadm_id"], how="left")
    cohort = cohort.merge(omr_features, on=["subject_id", "stay_id"], how="left")
    cohort = cohort.merge(lab_features, on="stay_id", how="left")
    cohort = cohort.merge(chart_features, on="stay_id", how="left")
    cohort = cohort.merge(fluid_features, on="stay_id", how="left")

    cohort = cohort[cohort["los_days"] > 0].copy()

    print("Applying MICE imputation...")
    id_cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "admittime",
               "los_hours", "los_days", "admission_type", "admission_location"]
    cohort = apply_mice_imputation(cohort, exclude_cols=id_cols)

    print(f"Final cohort shape: {cohort.shape}")
    cohort.to_parquet("cohort_preprocessed.parquet", index=False)
    print("Saved to cohort_preprocessed.parquet")
    return cohort


if __name__ == "__main__":
    build_master_cohort()
