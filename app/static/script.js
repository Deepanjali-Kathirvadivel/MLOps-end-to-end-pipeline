const form =
    document.getElementById("predictionForm");

const predictButton =
    document.getElementById("predictButton");

const loading =
    document.getElementById("loading");

const result =
    document.getElementById("result");

const errorBox =
    document.getElementById("error");

const statusDot =
    document.getElementById("statusDot");

const statusText =
    document.getElementById("statusText");


// =========================================================
// API Health Check
// =========================================================

async function checkAPI() {

    try {

        const response =
            await fetch("/health");

        if (!response.ok) {
            throw new Error(
                "API unavailable"
            );
        }

        const data =
            await response.json();

        statusDot.classList.add(
            "online"
        );

        statusText.textContent =
            "API Online";

    }

    catch (error) {

        statusDot.classList.remove(
            "online"
        );

        statusText.textContent =
            "API Offline";

    }

}


// =========================================================
// Show / Hide Helpers
// =========================================================

function showLoading() {

    loading.classList.remove(
        "hidden"
    );

    result.classList.add(
        "hidden"
    );

    errorBox.classList.add(
        "hidden"
    );

    predictButton.disabled = true;

    predictButton.textContent =
        "Predicting...";

}


function hideLoading() {

    loading.classList.add(
        "hidden"
    );

    predictButton.disabled =
        false;

    predictButton.textContent =
        "Predict Used Car Price";

}


function showError(message) {

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );

    result.classList.add(
        "hidden"
    );

}


function showResult(data) {

    const price =
        Number(
            data.predicted_price
        );

    document.getElementById(
        "predictedPrice"
    ).textContent =
        "$" +
        price.toLocaleString(
            "en-US",
            {
                maximumFractionDigits: 0
            }
        );


    document.getElementById(
        "modelName"
    ).textContent =
        data.model;


    document.getElementById(
        "vehicleAge"
    ).textContent =
        data.vehicle_age +
        " years";


    document.getElementById(
        "mileagePerYear"
    ).textContent =
        Number(
            data.mileage_per_year
        ).toLocaleString(
            "en-US",
            {
                maximumFractionDigits: 0
            }
        );


    result.classList.remove(
        "hidden"
    );

}


// =========================================================
// Prediction
// =========================================================

form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        showLoading();

        try {

            const vEngine =
                document.getElementById(
                    "vEngine"
                ).value;


            const requestData = {

                brand:
                    document.getElementById(
                        "brand"
                    ).value,

                model_year:
                    Number(
                        document.getElementById(
                            "modelYear"
                        ).value
                    ),

                mileage:
                    Number(
                        document.getElementById(
                            "mileage"
                        ).value
                    ),

                horsepower:
                    Number(
                        document.getElementById(
                            "horsepower"
                        ).value
                    ),

                engine_displacement:
                    Number(
                        document.getElementById(
                            "engineDisplacement"
                        ).value
                    ),

                fuel_type:
                    document.getElementById(
                        "fuelType"
                    ).value,

                transmission:
                    document.getElementById(
                        "transmission"
                    ).value,

                is_v_engine:
                    vEngine === "true",

                accident:
                    document.getElementById(
                        "accident"
                    ).value,

                clean_title:
                    document.getElementById(
                        "cleanTitle"
                    ).value

            };


            const response =
                await fetch(
                    "/predict",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                requestData
                            )

                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Prediction failed"
                );

            }


            showResult(data);

        }

        catch (error) {

            showError(
                error.message
            );

        }

        finally {

            hideLoading();

        }

    }
);


// =========================================================
// Initial API Check
// =========================================================

checkAPI();


// Check every 30 seconds

setInterval(
    checkAPI,
    30000
);