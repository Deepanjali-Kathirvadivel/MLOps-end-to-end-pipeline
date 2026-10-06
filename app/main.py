from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from swagger_ui_bundle import swagger_ui_path

# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "best_model.pkl"

# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Used Car Price Prediction API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)


# =========================================================
# OpenAPI 3.0 compatibility
# =========================================================

_original_openapi = app.openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = _original_openapi()

    # Swagger UI 4.15.5 works with OpenAPI 3.0.x
    schema["openapi"] = "3.0.3"

    app.openapi_schema = schema

    return app.openapi_schema


app.openapi = custom_openapi
# =========================================================
# Swagger UI static files
# =========================================================

app.mount(
    "/swagger-static",
    StaticFiles(directory=swagger_ui_path),
    name="swagger-ui",
)

# =========================================================
# Health Check
# =========================================================

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model": "XGBoost"
    }


# =========================================================
# Prediction Input
# =========================================================

from pydantic import BaseModel


class CarInput(BaseModel):
    brand: str
    fuel_type: str
    transmission: str
    clean_title: str
    hp: float
    engine_displacement: float
    is_v_engine: int
    Accident_Impact: str
    Vehicle_Age: float
    Mileage_per_Year: float
    Age_Mid: int
    Age_Old: int
    Age_Very_Old: int
    Milage_Medium: int
    Milage_High: int
    Milage_Very_High: int


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post("/predict", tags=["Prediction"])
def predict(data: CarInput):

    # Temporary response until we connect the final
    # preprocessing/model prediction logic.

    return {
        "message": "Prediction endpoint working",
        "model": "XGBoost"
    }


# =========================================================
# Custom Swagger UI
# =========================================================

@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Used Car Price Prediction API</title>

        <link
            type="text/css"
            rel="stylesheet"
            href="/swagger-static/swagger-ui.css"
        >
    </head>

    <body>

        <div id="swagger-ui"></div>

        <script src="/swagger-static/swagger-ui-bundle.js"></script>

        <script>
            window.onload = function() {{
                window.ui = SwaggerUIBundle({{
                    url: "/openapi.json",
                    dom_id: "#swagger-ui",
                    deepLinking: true,
                    displayRequestDuration: true,
                    defaultModelsExpandDepth: -1,
                    defaultModelExpandDepth: 2,
                    docExpansion: "list",
                    filter: true,
                    showExtensions: true,
                    showCommonExtensions: true,
                    presets: [
                        SwaggerUIBundle.presets.apis
                    ],
                    layout: "BaseLayout"
                }});
            }};
        </script>

    </body>
    </html>
    """

    return HTMLResponse(content=html)