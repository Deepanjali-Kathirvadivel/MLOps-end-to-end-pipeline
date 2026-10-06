from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from swagger_ui_bundle import swagger_ui_path


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"

MODEL_PATH = BASE_DIR / "artifacts" / "models" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "artifacts" / "models" / "model_metadata.json"

SRC_DIR = BASE_DIR / "src"

# Required so joblib can resolve BrandTargetEncoder
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from car_transformers import BrandTargetEncoder  # noqa: F401
except Exception:
    pass


# =========================================================
# Load Model + Metadata
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

metadata = {}

if METADATA_PATH.exists():
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )


# =========================================================
# Metadata
# =========================================================

feature_columns = metadata.get(
    "model_input", {}
).get(
    "feature_columns",
    []
)

feature_prep = metadata.get(
    "feature_preparation",
    {}
)

reference_year = int(
    feature_prep.get(
        "vehicle_age_reference_year",
        2025
    )
)

label_encoder_classes = feature_prep.get(
    "label_encoder_classes",
    {}
)

age_edges = feature_prep.get(
    "vehicle_age_bin_edges",
    []
)

mileage_edges = feature_prep.get(
    "mileage_bin_edges",
    []
)

model_name = metadata.get(
    "model_name",
    "XGBoost"
)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Used Car Price Prediction API",
    description="MLOps API for used car price prediction",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)


# =========================================================
# OpenAPI 3.0 Compatibility
# =========================================================

_original_openapi = app.openapi


def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    schema = _original_openapi()

    schema["openapi"] = "3.0.3"

    app.openapi_schema = schema

    return app.openapi_schema


app.openapi = custom_openapi


# =========================================================
# Swagger Static Files
# =========================================================

app.mount(
    "/swagger-static",
    StaticFiles(
        directory=swagger_ui_path
    ),
    name="swagger-ui",
)


# =========================================================
# Frontend Static Files
# =========================================================

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)


# =========================================================
# Home Page
# =========================================================

@app.get(
    "/",
    include_in_schema=False
)
def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found"
        )

    return HTMLResponse(
        content=index_file.read_text(
            encoding="utf-8"
        )
    )


# =========================================================
# Health Check
# =========================================================

@app.get(
    "/health",
    tags=["Health"]
)
def health():

    return {
        "status": "healthy",
        "model": model_name
    }


# =========================================================
# Prediction Input
# =========================================================

class CarInput(BaseModel):

    brand: str = Field(
        ...,
        example="BMW"
    )

    model_year: int = Field(
        ...,
        ge=1980,
        le=2025,
        example=2018
    )

    mileage: float = Field(
        ...,
        ge=0,
        example=45000
    )

    horsepower: float = Field(
        ...,
        gt=0,
        example=300
    )

    engine_displacement: float = Field(
        ...,
        gt=0,
        example=3.0
    )

    fuel_type: str = Field(
        ...,
        example="GASOLINE"
    )

    transmission: str = Field(
        ...,
        example="A/T"
    )

    is_v_engine: bool = Field(
        ...,
        example=True
    )

    accident: str = Field(
        ...,
        example="None reported"
    )

    clean_title: str = Field(
        ...,
        example="Yes"
    )


# =========================================================
# Encoding Helpers
# =========================================================

def get_classes(
    key,
    fallback
):

    classes = label_encoder_classes.get(
        key
    )

    if classes:

        return [
            str(value)
            for value in classes
        ]

    return fallback


fuel_options = get_classes(
    "fuel_type",
    [
        "GASOLINE",
        "DIESEL",
        "HYBRID",
        "ELECTRIC",
        "OTHER",
    ]
)


transmission_options = get_classes(
    "transmission",
    [
        "A/T",
        "M/T",
        "CVT",
        "OTHER",
    ]
)


v_engine_classes = get_classes(
    "is_v_engine",
    [
        "False",
        "True",
    ]
)


def encoded_value(
    value,
    classes,
    fallback_name="OTHER"
):

    classes = [
        str(c)
        for c in classes
    ]

    value = str(value)

    if value not in classes:

        if fallback_name in classes:

            value = fallback_name

        else:

            raise ValueError(
                f"'{value}' is not present in "
                f"the model's learned classes: {classes}"
            )

    return classes.index(value)


# =========================================================
# Bin One-Hot Encoding
# =========================================================

def bin_one_hot(
    value,
    edges,
    prefix,
    labels
):

    if len(edges) != 5:

        raise ValueError(
            f"Invalid {prefix} bin edges in model metadata."
        )

    clipped = float(
        np.clip(
            value,
            edges[0],
            edges[-1]
        )
    )

    binned = pd.cut(
        pd.Series([clipped]),
        bins=edges,
        labels=labels,
        include_lowest=True
    )

    dummies = pd.get_dummies(
        binned,
        prefix=prefix,
        drop_first=True,
        dtype=int
    )

    expected = [
        f"{prefix}_{label}"
        for label in labels[1:]
    ]

    for column in expected:

        if column not in dummies.columns:

            dummies[column] = 0

    return {
        column: int(
            dummies.iloc[0][column]
        )
        for column in expected
    }


# =========================================================
# Feature Builder
# =========================================================

def build_features(data: CarInput):

    vehicle_age = (
        reference_year
        - int(data.model_year)
    )

    if vehicle_age < 0:

        raise ValueError(
            f"Model year cannot be after "
            f"{reference_year}."
        )

    mileage_per_year = (
        float(data.mileage)
        / vehicle_age
        if vehicle_age > 0
        else float(data.mileage)
    )

    row = {

        "brand":
            data.brand.strip(),

        "fuel_type":
            encoded_value(
                data.fuel_type,
                fuel_options
            ),

        "transmission":
            encoded_value(
                data.transmission,
                transmission_options
            ),

        "is_v_engine":
            encoded_value(
                str(data.is_v_engine),
                v_engine_classes
            ),

        "clean_title":
            1
            if data.clean_title == "Yes"
            else 0,

        "hp":
            float(data.horsepower),

        "engine displacement":
            float(
                data.engine_displacement
            ),

        "Accident_Impact":
            1
            if data.accident
            == "At least 1 accident or damage reported"
            else 0,

        "Vehicle_Age":
            float(vehicle_age),

        "Mileage_per_Year":
            float(mileage_per_year),
    }


    # -------------------------------
    # Vehicle Age Bins
    # -------------------------------

    row.update(
        bin_one_hot(
            vehicle_age,
            age_edges,
            "Age",
            [
                "New",
                "Mid",
                "Old",
                "Very Old"
            ]
        )
    )


    # -------------------------------
    # Mileage Bins
    # -------------------------------

    row.update(
        bin_one_hot(
            float(data.mileage),
            mileage_edges,
            "Milage",
            [
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )
    )


    frame = pd.DataFrame(
        [row]
    )


    # -------------------------------
    # Validate Features
    # -------------------------------

    missing = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing:

        raise ValueError(
            "Generated input is missing "
            "model features: "
            + ", ".join(missing)
        )


    # Exact training column order
    return frame[
        feature_columns
    ]


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post(
    "/predict",
    tags=["Prediction"]
)
def predict(data: CarInput):

    try:

        X_input = build_features(
            data
        )

        prediction = float(
            model.predict(
                X_input
            )[0]
        )

        prediction = max(
            0.0,
            prediction
        )

        return {

            "success": True,

            "predicted_price":
                round(
                    prediction,
                    2
                ),

            "currency":
                "USD",

            "model":
                model_name,

            "vehicle_age":
                reference_year
                - data.model_year,

            "mileage_per_year":
                round(
                    data.mileage
                    / (
                        reference_year
                        - data.model_year
                    )
                    if data.model_year
                    != reference_year
                    else data.mileage,
                    2
                )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# =========================================================
# Swagger UI
# =========================================================

@app.get(
    "/docs",
    include_in_schema=False
)
def custom_swagger_ui():

    html = """
    <!DOCTYPE html>

    <html>

    <head>

        <title>
            Used Car Price Prediction API
        </title>

        <link
            type="text/css"
            rel="stylesheet"
            href="/swagger-static/swagger-ui.css"
        >

    </head>

    <body>

        <div id="swagger-ui"></div>

        <script
            src="/swagger-static/swagger-ui-bundle.js">
        </script>

        <script>

            window.onload = function() {

                window.ui =
                    SwaggerUIBundle({

                        url: "/openapi.json",

                        dom_id:
                            "#swagger-ui",

                        deepLinking:
                            true,

                        displayRequestDuration:
                            true,

                        defaultModelsExpandDepth:
                            -1,

                        defaultModelExpandDepth:
                            2,

                        docExpansion:
                            "list",

                        filter:
                            true,

                        showExtensions:
                            true,

                        showCommonExtensions:
                            true,

                        presets: [
                            SwaggerUIBundle
                                .presets
                                .apis
                        ],

                        layout:
                            "BaseLayout"
                    });

            };

        </script>

    </body>

    </html>
    """

    return HTMLResponse(
        content=html
    )