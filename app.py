
# Student Performance Prediction
# Predict final Mathematics grade (G3) using 8 selected features.

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Estimate a student's final Mathematics grade using eight "
    "selected student and study-related inputs."
)

st.caption("Machine Learning • UCI Student Performance Dataset")


# --------------------------------------------------
# 2. Configuration and dataset loading
# --------------------------------------------------

DATA_PATH = Path(__file__).parent / "student-mat.csv"

FEATURES = [
    "age",
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "famsup",
    "internet",
    "paid",
]

TARGET = "G3"

NUMERIC_FEATURES = [
    "age",
    "studytime",
    "failures",
    "absences",
]

CATEGORICAL_FEATURES = [
    "schoolsup",
    "famsup",
    "internet",
    "paid",
]


@st.cache_data
def load_data():
    """Load the actual Mathematics dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "student-mat.csv was not found. Place it in the same "
            "folder as app.py."
        )

    data = pd.read_csv(DATA_PATH, sep=";")

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )

    data = data[required_columns].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])

    return data


# --------------------------------------------------
# 3. Preprocessing and model definitions
# --------------------------------------------------

def build_preprocessor():
    """Build preprocessing that is fitted only on training folds."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models():
    """Return models using the same input features and evaluation setup."""

    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }


@st.cache_resource
def train_and_evaluate():
    """
    Compare models fairly:
    - Same eight input features
    - Same training/test split
    - Same shuffled cross-validation folds
    - Same MAE selection metric
    - Preprocessing happens inside each training fold
    """

    data = load_data()

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []
    fitted_models = {}

    for model_name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        # Cross-validation uses only the training partition.
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )

        cv_mae = -cv_scores.mean()

        # Fit on the full training partition.
        pipeline.fit(X_train, y_train)

        # Evaluate on the same untouched test partition.
        predictions = pipeline.predict(X_test)

        test_mae = mean_absolute_error(y_test, predictions)
        test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        test_r2 = r2_score(y_test, predictions)

        results.append(
            {
                "Model": model_name,
                "CV MAE": cv_mae,
                "Test MAE": test_mae,
                "Test RMSE": test_rmse,
                "Test R²": test_r2,
            }
        )

        fitted_models[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        by="CV MAE",
        ascending=True,
    ).reset_index(drop=True)

    # Select using CV MAE, not test-set performance.
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    return (
        data,
        results_df,
        best_model_name,
        best_model,
        X_test,
        y_test,
    )


# --------------------------------------------------
# 4. Train and show model evaluation
# --------------------------------------------------

try:
    with st.spinner("Training and comparing the three models..."):
        (
            data,
            results_df,
            best_model_name,
            best_model,
            X_test,
            y_test,
        ) = train_and_evaluate()

except Exception as error:
    st.error(f"Could not load or train the model: {error}")
    st.info(
        "Check that student-mat.csv is in the same folder as app.py "
        "and that all required packages are installed."
    )
    st.stop()


st.success("Models trained and evaluated successfully.")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Students in dataset", f"{len(data)}")

with metric2:
    st.metric("Input features", f"{len(FEATURES)}")

with metric3:
    st.metric("Selected model", best_model_name)


with st.expander("📊 Compare all three models", expanded=True):
    st.write(
        "**CV MAE** is the average absolute error across five "
        "cross-validation folds on the training data. Lower is better."
    )

    st.write(
        "**Test MAE and Test RMSE** measure errors on the held-out test "
        "data. Lower is better. **Test R²** measures how well predictions "
        "explain grade variation; values closer to 1 are generally better."
    )

    display_results = results_df.copy()

    for column in ["CV MAE", "Test MAE", "Test RMSE", "Test R²"]:
        display_results[column] = display_results[column].round(3)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The selected model is the one with the lowest cross-validation "
        "MAE. Test metrics are reported for comparison, not used to select "
        "the model."
    )


# --------------------------------------------------
# 5. Simple prediction interface
# --------------------------------------------------

st.divider()
st.header("🔍 Predict a student's final grade")
st.write("Enter the student's details below.")

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        age = st.slider(
            "Age",
            min_value=15,
            max_value=22,
            value=17,
            help="Student's age in years.",
        )

        studytime = st.selectbox(
            "Weekly study time",
            options=[1, 2, 3, 4],
            index=1,
            format_func=lambda value: {
                1: "1 — Less than 2 hours",
                2: "2 — 2 to 5 hours",
                3: "3 — 5 to 10 hours",
                4: "4 — More than 10 hours",
            }[value],
        )

        failures = st.selectbox(
            "Past class failures",
            options=[0, 1, 2, 3],
            index=0,
            help="Number of past class failures; 3 represents three or more.",
        )

        absences = st.number_input(
            "School absences",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

    with right:
        schoolsup = st.radio(
            "Extra educational support at school?",
            options=["yes", "no"],
            index=1,
            horizontal=True,
        )

        famsup = st.radio(
            "Family educational support?",
            options=["yes", "no"],
            index=0,
            horizontal=True,
        )

        internet = st.radio(
            "Internet access at home?",
            options=["yes", "no"],
            index=0,
            horizontal=True,
        )

        paid = st.radio(
            "Paid extra classes for Mathematics?",
            options=["yes", "no"],
            index=1,
            horizontal=True,
        )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        type="primary",
        use_container_width=True,
    )


if submitted:
    student = pd.DataFrame(
        [
            {
                "age": age,
                "studytime": studytime,
                "failures": failures,
                "absences": absences,
                "schoolsup": schoolsup,
                "famsup": famsup,
                "internet": internet,
                "paid": paid,
            }
        ],
        columns=FEATURES,
    )

    prediction = float(best_model.predict(student)[0])

    # G3 is recorded on a 0–20 scale in this dataset.
    prediction = float(np.clip(prediction, 0, 20))

    st.divider()
    st.subheader("Prediction result")

    result_col, info_col = st.columns([1, 2])

    with result_col:
        st.metric(
            "Estimated final grade",
            f"{prediction:.1f} / 20",
        )

    with info_col:
        st.write(f"**Model used:** {best_model_name}")
        st.write(
            "This is a machine-learning estimate based on patterns in "
            "the historical dataset. It is not a guaranteed grade or an "
            "official academic assessment."
        )

    st.progress(prediction / 20)

    if prediction < 10:
        st.warning(
            "The estimate is below 10/20. Consider reviewing study habits "
            "and available learning support. This is not a judgment of the "
            "student's ability."
        )
    else:
        st.info(
            "The estimate is 10/20 or higher. Actual results can still "
            "differ from this prediction."
        )


# --------------------------------------------------
# 6. Footer and limitations
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ About this application"):
    st.write(
        """
        **Target:** G3, the final Mathematics grade in the dataset, on a
        0–20 scale.

        **Inputs:** age, studytime, failures, absences, schoolsup, famsup,
        internet, and paid.

        **Models:** Linear Regression, Random Forest, and Gradient Boosting.

        **Evaluation:** All models use the same 80/20 train-test split and
        the same five-fold cross-validation procedure. Preprocessing is
        fitted inside each pipeline to reduce data leakage.

        **Limitations:** This is a teaching and demonstration application
        based on historical data from Portuguese schools. It may not
        generalize to other schools or education systems. Associations in
        the data do not establish cause and effect.
        """
    )

st.caption("Student Performance Prediction • Educational demonstration")
