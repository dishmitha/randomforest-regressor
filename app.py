import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor


@st.cache_resource
def get_trained_model():
    data = fetch_california_housing()
    X = data.data
    y = data.target

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, data.feature_names


def main():
    st.set_page_config(page_title="Random Forest Regression Predictor", layout="centered")
    st.title("Random Forest Regression Predictor")

    model, feature_names = get_trained_model()

    with st.form("prediction_form"):
        st.write("Enter feature values:")
        inputs = []
        for i, name in enumerate(feature_names):
            # If user changes values, prediction must update on next click
            # (Streamlit forms only update state on submit).
            val = st.number_input(
                name,
                value=0.0,
                format="%.6f",
                key=f"feature_{i}",
            )
            inputs.append(val)

        submitted = st.form_submit_button("Predict")

    # Always show the latest inputs used for prediction
    # (helps confirm UI is changing.)
    st.caption("Latest input vector used: " + ", ".join([f"{v:.4f}" for v in inputs]))

    if submitted:
        # Use plain python list-of-floats to avoid pandas feature-name checks
        X_new = [[float(v) for v in inputs]]
        pred = float(model.predict(X_new)[0])
        st.success(f"Predicted output: {pred:.6f}")

        st.write("Debug: model input ->", inputs)


if __name__ == "__main__":
    main()

