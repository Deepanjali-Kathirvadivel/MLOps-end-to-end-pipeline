import os
import sys
import json
import time
import platform
import warnings
import inspect
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Hint: Inferred schema")

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "used_cars.csv"
SRC_DIR = ROOT / "src"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_PATH = REPORTS_DIR / "comparison.csv"
MLRUNS_DIR = ROOT / "mlruns"

for d in (MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
os.environ["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 7
PRIMARY_METRIC = "test_rmse"

# True = retrain even when best_model.pkl already exists
FORCE_RETRAIN = True

# True = use small grids for a quick test
QUICK_RUN = False

np.random.seed(RANDOM_STATE)

# -------------------------------------------------------------------
# MLflow
# -------------------------------------------------------------------
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.sklearn
import xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "used-car-price-prediction"
TRACKING_URI = f"file:{MLRUNS_DIR.as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

_params = inspect.signature(mlflow.sklearn.log_model).parameters
MODEL_ARG = "name" if "name" in _params else "artifact_path"

# -------------------------------------------------------------------
# Scikit-learn
# -------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor

from car_transformers import BrandTargetEncoder


# -------------------------------------------------------------------
# Data preparation - same logic as the notebook
# -------------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
n_rows_raw = len(df)

df["milage"] = df["milage"].str.replace(" mi.", "").str.replace(",", "").astype(int)
df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype(int)

df["hp"] = df["engine"].str.extract(r"(\d+\.\d+)HP").astype(float, errors="ignore")
df["engine displacement"] = df["engine"].str.extract(r"(\d+\.\d+)\s*L")
df["engine displacement"] = df["engine displacement"].fillna(
    df["engine"].str.extract(r"(\d+\.\d+)\s*LITER")[0]
)
df["engine displacement"] = df["engine displacement"].astype(float, errors="ignore")
df["is_v_engine"] = df["engine"].str.contains(r"V\d+", case=False, na=False)

df["fuel_type"] = (
    df["fuel_type"]
    .str.strip()
    .str.upper()
    .replace({
        "PLUG-IN HYBRID": "HYBRID",
        "NOT SUPPORTED": "OTHER",
        "–": "OTHER",
    })
)


def classify_transmission(transmission):
    t = str(transmission).upper()
    if "M/T" in t or "MT" in t or "MANUAL" in t:
        return "M/T"
    elif "A/T" in t or "AT" in t or "AUTOMATIC" in t:
        return "A/T"
    elif "CVT" in t or "VARIABLE" in t or "SINGLE-SPEED" in t:
        return "CVT"
    return "OTHER"


df["transmission"] = df["transmission"].apply(classify_transmission)

df["hp"] = df.groupby("brand")["hp"].transform(lambda x: x.fillna(x.mean()))
df.dropna(subset=["hp"], inplace=True)

most_common_fuel = df.groupby("brand")["fuel_type"].agg(
    lambda x: x.mode()[0] if not x.mode().empty else None
)
df["fuel_type"] = df.apply(
    lambda row: most_common_fuel[row["brand"]]
    if pd.isna(row["fuel_type"])
    else row["fuel_type"],
    axis=1,
)
df["fuel_type"] = df["fuel_type"].fillna("OTHER")

most_common_displacement = df.groupby("brand")["engine displacement"].agg(
    lambda x: x.mode()[0] if not x.mode().empty else None
)
df["engine displacement"] = df.apply(
    lambda row: most_common_displacement[row["brand"]]
    if pd.isna(row["engine displacement"])
    else row["engine displacement"],
    axis=1,
)
df["engine displacement"] = df["engine displacement"].fillna(
    df["engine displacement"].median()
)

for col in ["engine displacement", "hp", "price", "milage"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

df["Accident_Impact"] = df["accident"].apply(
    lambda x: 1 if x == "At least 1 accident or damage reported" else 0
)
df["clean_title"] = df["clean_title"].apply(lambda x: 1 if x == "Yes" else 0)

categorical_columns = ["fuel_type", "transmission", "is_v_engine"]
label_encoder_classes = {}

for cat_col in categorical_columns:
    encoder = LabelEncoder()
    df[cat_col] = encoder.fit_transform(df[cat_col])
    label_encoder_classes[cat_col] = [str(c) for c in encoder.classes_]

df["Vehicle_Age"] = 2025 - df["model_year"]
df["Mileage_per_Year"] = df.apply(
    lambda row: row["milage"] / row["Vehicle_Age"]
    if row["Vehicle_Age"] > 0
    else row["milage"],
    axis=1,
)

df["Vehicle_Age_Bin"], age_bin_edges = pd.qcut(
    df["Vehicle_Age"],
    q=4,
    labels=["New", "Mid", "Old", "Very Old"],
    retbins=True,
)
df["Mileage_Bin"], mileage_bin_edges = pd.qcut(
    df["milage"],
    q=4,
    labels=["Low", "Medium", "High", "Very High"],
    retbins=True,
)

df = pd.get_dummies(
    df,
    columns=["Vehicle_Age_Bin", "Mileage_Bin"],
    prefix=["Age", "Milage"],
    drop_first=True,
    dtype=int,
)

df.drop(
    ["model", "model_year", "engine", "milage", "int_col", "ext_col", "accident"],
    axis=1,
    inplace=True,
)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# -------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------
X = df.drop(["price"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

y_train_log = np.log1p(y_train)
assert (y > 0).all(), "MAPE needs strictly positive targets"

FEATURE_COLUMNS = list(X.columns)

BASE_PARAMS = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "n_rows_raw": n_rows_raw,
    "n_rows_after_cleaning": len(df),
    "n_features": X.shape[1],
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "target": "price",
    "target_transform": "log1p (inverse: expm1)",
    "outlier_removal": "IQR 1.5x on engine displacement, hp, price, milage",
    "categorical_encoding": "LabelEncoder (fuel_type, transmission, is_v_engine) + one-hot age/mileage bins",
    "brand_encoding": "target mean of log-price, fit on train only",
    "feature_scaling": "StandardScaler",
    "feature_selection": "manual: dropped model, model_year, engine, milage, int_col, ext_col, accident",
}

DATASET_SUMMARY = {
    "feature_columns": FEATURE_COLUMNS,
    "price_describe": y.describe().round(2).to_dict(),
    "n_rows_raw": n_rows_raw,
    "n_rows_after_cleaning": len(df),
}


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
def build_pipeline(estimator):
    return Pipeline([
        ("brand_encoder", BrandTargetEncoder(column="brand")),
        ("scaler", StandardScaler()),
        ("model", clone(estimator)),
    ])


def wrap_target_transform(pipeline):
    return TransformedTargetRegressor(
        regressor=clone(pipeline),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


if QUICK_RUN:
    CV_FOLDS = 3
    rf_grid = {
        "model__n_estimators": [100],
        "model__max_depth": [8, 12],
        "model__min_samples_split": [5],
        "model__min_samples_leaf": [2],
        "model__max_features": ["sqrt"],
    }
    svr_grid = {
        "model__C": [1, 10],
        "model__epsilon": [0.1],
        "model__kernel": ["rbf"],
        "model__gamma": ["scale"],
    }
    xgb_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03],
        "model__max_depth": [5],
        "model__subsample": [0.9],
        "model__colsample_bytree": [0.8],
        "model__reg_lambda": [1],
        "model__reg_alpha": [0],
    }
else:
    rf_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [8, 10, 12],
        "model__min_samples_split": [5, 10, 15],
        "model__min_samples_leaf": [2, 4, 6],
        "model__max_features": ["sqrt"],
    }
    svr_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 0.2, 0.5],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }
    xgb_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.03],
        "model__max_depth": [5, 7],
        "model__subsample": [0.7, 0.9],
        "model__colsample_bytree": [0.6, 0.8],
        "model__reg_lambda": [1, 10],
        "model__reg_alpha": [0, 1],
    }

ridge_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}

CANDIDATES = {
    "LinearRegression": {
        "estimator": LinearRegression(),
        "param_grid": None,
        "log_params": ["fit_intercept"],
    },
    "Ridge": {
        "estimator": Ridge(random_state=RANDOM_STATE),
        "param_grid": ridge_grid,
        "log_params": ["alpha", "fit_intercept"],
    },
    "RandomForest": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=1
        ),
        "param_grid": rf_grid,
        "log_params": [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        ],
    },
    "SVR": {
        "estimator": SVR(),
        "param_grid": svr_grid,
        "log_params": ["C", "epsilon", "kernel", "gamma"],
    },
    "XGBoost": {
        "estimator": XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "param_grid": xgb_grid,
        "log_params": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
        ],
    },
}

CV_SCORING = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
}


def regression_metrics(y_true, y_pred, prefix):
    mse = mean_squared_error(y_true, y_pred)
    return {
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": float(np.sqrt(mse)),
        f"{prefix}_r2": r2_score(y_true, y_pred),
        f"{prefix}_mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    return {
        **regression_metrics(y_tr, pred_tr, "train"),
        **regression_metrics(y_te, pred_te, "test"),
    }, pred_te


# -------------------------------------------------------------------
# MLflow training
# -------------------------------------------------------------------
SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
CODE_PATHS = [str(SRC_DIR / "car_transformers.py")]


def train_and_log(name, spec):
    pipeline = build_pipeline(spec["estimator"])
    t0 = time.time()

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({
            "run_type": "candidate",
            "model_name": name,
            "training_session": SESSION_ID,
            "sklearn_version": __import__("sklearn").__version__,
            "xgboost_version": xgboost.__version__,
            "python_version": platform.python_version(),
        })

        mlflow.log_params(BASE_PARAMS)
        mlflow.log_param("model_type", name)
        mlflow.log_param("estimator_class", type(spec["estimator"]).__name__)

        search = None

        if spec["param_grid"]:
            search = GridSearchCV(
                pipeline,
                spec["param_grid"],
                scoring=CV_SCORING,
                refit="RMSE",
                cv=KFold(
                    n_splits=CV_FOLDS,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                ),
                n_jobs=-1,
            )
            search.fit(X_train, y_train_log)
            best_pipeline = search.best_estimator_

            mlflow.log_params({
                "tuning_method": "GridSearchCV",
                "cv_folds": CV_FOLDS,
                "cv_refit_metric": "RMSE (log-price scale)",
                "grid_candidates": len(search.cv_results_["params"]),
            })
        else:
            best_pipeline = pipeline
            mlflow.log_param("tuning_method", "none")

        model = wrap_target_transform(best_pipeline)
        model.fit(X_train, y_train)

        fit_seconds = time.time() - t0

        est_params = model.regressor_.named_steps["model"].get_params()
        mlflow.log_params({
            k: est_params[k] for k in spec["log_params"]
        })

        metrics, pred_te = evaluate_model(
            model, X_train, y_train, X_test, y_test
        )
        metrics["fit_seconds"] = fit_seconds

        if search is not None:
            i = search.best_index_
            metrics["cv_rmse_logscale"] = -search.cv_results_["mean_test_RMSE"][i]
            metrics["cv_mae_logscale"] = -search.cv_results_["mean_test_MAE"][i]
            metrics["cv_r2_logscale"] = search.cv_results_["mean_test_R2"][i]

        mlflow.log_metrics(metrics)

        if search is not None:
            cv_df = pd.DataFrame(search.cv_results_)
            keep = [
                c for c in cv_df.columns
                if c.startswith("param_")
                or c.startswith("mean_test_")
                or c == "rank_test_RMSE"
            ]
            mlflow.log_text(
                cv_df[keep]
                .sort_values("rank_test_RMSE")
                .to_csv(index=False),
                "cv/cv_results.csv",
            )

        mlflow.log_dict(
            {
                "model_name": name,
                "best_hyperparameters": {
                    k: str(est_params[k])
                    for k in spec["log_params"]
                },
                "base_params": BASE_PARAMS,
                "primary_metric": PRIMARY_METRIC,
            },
            "config/training_config.json",
        )

        mlflow.log_dict(
            DATASET_SUMMARY,
            "config/dataset_summary.json",
        )

        signature = infer_signature(
            X_train, model.predict(X_train)
        )

        mlflow.sklearn.log_model(
            model,
            **{MODEL_ARG: "model"},
            signature=signature,
            input_example=X_train.head(3),
            code_paths=CODE_PATHS,
            serialization_format="cloudpickle",
        )

    print(
        f"{name:<17} "
        f"RMSE={metrics['test_rmse']:,.2f}  "
        f"MAE={metrics['test_mae']:,.2f}  "
        f"R2={metrics['test_r2']:.3f}  "
        f"time={fit_seconds:.1f}s"
    )

    return {
        "run_id": run.info.run_id,
        "model": model,
        "metrics": metrics,
    }


# -------------------------------------------------------------------
# Run training
# -------------------------------------------------------------------
if MODEL_PATH.exists() and not FORCE_RETRAIN:
    print(f"Model already exists: {MODEL_PATH}")
    print("Set FORCE_RETRAIN = True to retrain.")
    raise SystemExit(0)

print("=" * 70)
print("USED CAR PRICE - MLOPS TRAINING")
print("=" * 70)
print(f"Dataset : {DATA_PATH}")
print(f"Rows    : {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Models  : {len(CANDIDATES)}")
print(f"MLflow  : {TRACKING_URI}")
print()

results = {}

for model_name, spec in CANDIDATES.items():
    results[model_name] = train_and_log(model_name, spec)

comparison_cols = [
    "test_mae",
    "test_mse",
    "test_rmse",
    "test_r2",
    "test_mape",
    "train_rmse",
    "train_r2",
    "fit_seconds",
]

comparison = pd.DataFrame([
    {
        "model": name,
        "run_id": result["run_id"],
        **{
            col: result["metrics"][col]
            for col in comparison_cols
        },
    }
    for name, result in results.items()
])

comparison = comparison.sort_values(
    PRIMARY_METRIC, ascending=True
).reset_index(drop=True)

comparison.to_csv(COMPARISON_PATH, index=False)

best_row = comparison.iloc[0]
best_name = best_row["model"]
best_run_id = best_row["run_id"]
best_model = results[best_name]["model"]

print()
print("=" * 70)
print(f"BEST MODEL : {best_name}")
print(f"TEST RMSE  : {best_row['test_rmse']:,.2f}")
print(f"TEST R2    : {best_row['test_r2']:.4f}")
print(f"RUN ID     : {best_run_id}")
print("=" * 70)

# Mark the best candidate run
client = MlflowClient()
for name, result in results.items():
    client.log_param(
        result["run_id"],
        "is_best_model",
        name == best_name,
    )

# Save model
joblib.dump(best_model, MODEL_PATH)

metadata = {
    "model_name": best_name,
    "selection_metric": PRIMARY_METRIC,
    "selection_rule": "lowest value wins",
    "metrics": {
        k: float(v)
        for k, v in results[best_name]["metrics"].items()
    },
    "trained_at_utc": datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ),
    "mlflow": {
        "tracking_uri": TRACKING_URI,
        "experiment_name": EXPERIMENT_NAME,
        "run_id": best_run_id,
        "model_artifact_path": "model",
    },
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "target": {
        "name": "price",
        "unit": "USD",
        "model_trains_on": "log1p(price)",
        "predict_returns": "price in USD",
    },
    "model_input": {
        "feature_columns": FEATURE_COLUMNS,
        "dtypes": {
            c: str(t) for c, t in X_train.dtypes.items()
        },
    },
    "feature_preparation": {
        "label_encoder_classes": label_encoder_classes,
        "vehicle_age_bin_edges": [
            float(v) for v in age_bin_edges
        ],
        "mileage_bin_edges": [
            float(v) for v in mileage_bin_edges
        ],
        "vehicle_age_reference_year": 2025,
    },
    "library_versions": {
        "python": platform.python_version(),
        "scikit-learn": __import__("sklearn").__version__,
        "xgboost": xgboost.__version__,
        "mlflow": mlflow.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
}

METADATA_PATH.write_text(
    json.dumps(metadata, indent=2)
)

print()
print("Saved model    :", MODEL_PATH)
print("Saved metadata :", METADATA_PATH)
print("Saved report   :", COMPARISON_PATH)
print("MLflow runs    :", MLRUNS_DIR)
