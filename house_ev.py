import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Jiji House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("house_price_model.pkl")
        scaler = joblib.load("house_scaler_features.pkl")
        label_encoders = joblib.load("house_label_encoders.pkl")
        feature_cols   = joblib.load("house_feature_columns.pkl")
        return model, scaler, label_encoders, feature_cols
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("jiji_housing_cleaned.csv")
        return df
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.stop()

# ── Filtered options ──────────────────────────────────────────
def get_filtered_options(df, state=None, furnishing=None):
    filtered = df.copy()

    if state:
        filtered = filtered[filtered["state"] == state]
    if furnishing:
        filtered = filtered[filtered["furnishing"] == furnishing]

    options = {
        "states":         sorted(df["state"].unique().tolist()),
        "furnishings":    sorted(df["furnishing"].unique().tolist()),
        "bedrooms":       sorted(filtered["bedrooms"].unique().tolist()),
        "bathrooms":      sorted(filtered["bathrooms"].unique().tolist()),
        "property_sizes": sorted(filtered["property_size"].unique().tolist()),
    }
    return options

# ── Prediction ────────────────────────────────────────────────
def predict_price(house_data, model, scaler, label_encoders, feature_cols):
    try:
        cat_cols = feature_cols["categorical"]   # ['state', 'furnishing']
        num_cols = feature_cols["numeric"]        # ['property_size', 'bedrooms', 'bathrooms']

        # Step 1: label-encode categoricals
        input_df = pd.DataFrame([house_data])
        for col in cat_cols:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except Exception:
                    st.warning(f"Unknown value for '{col}'. Using default encoding.")
                    input_df[col] = 0

        # Step 2: scale numeric columns — pass as named DataFrame to avoid warnings
        num_df = pd.DataFrame([[house_data[c] for c in num_cols]], columns=num_cols)
        num_scaled = scaler.transform(num_df)  # returns ndarray shape (1, 3)

        # Step 3: build final feature DataFrame in correct column order
        cat_df = input_df[cat_cols].reset_index(drop=True)
        num_scaled_df = pd.DataFrame(num_scaled, columns=num_cols)
        features = pd.concat([cat_df, num_scaled_df], axis=1)

        # Step 4: predict
        predicted_price = model.predict(features)[0]

        margin = 0.15
        return {
            "predicted_price":     predicted_price,
            "min_predicted_price": predicted_price * (1 - margin),
            "max_predicted_price": predicted_price * (1 + margin),
        }

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────
def main():
    st.title("🏠 Free House Price Evaluation")
    st.write("Just fill all the fields and get an immediate result.")

    model, scaler, label_encoders, feature_cols = load_model_artifacts()
    df = load_dataset()

    if "show_result" not in st.session_state:
        st.session_state.show_result = False

    options = get_filtered_options(df)

    # ── 1. State (Row 1) ──
    state = st.selectbox(
        "State *",
        options=[""] + options["states"],
        format_func=lambda x: "Select State" if x == "" else x
    )

    # ── 2. Furnishing (Row 2) ──
    current_options = get_filtered_options(df, state=state) if state else options
    furnishing = st.selectbox(
        "Furnishing Status *",
        options=[""] + current_options["furnishings"],
        format_func=lambda x: "Select Furnishing Status" if x == "" else x,
        disabled=(state == "")
    )

    # ── 3. Bedrooms (Row 3) ──
    bedrooms = st.number_input(
        "Number of Bedrooms *",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        disabled=(furnishing == "")
    )

    # ── 4. Bathrooms (Row 4) ──
    bathrooms = st.number_input(
        "Number of Bathrooms *",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        disabled=(furnishing == "")
    )

    # ── 5. Property Size (Row 5) ──
    property_size = st.number_input(
        "Property Size (sqm) *",
        min_value=1,
        max_value=10000,
        value=100,
        step=1,
        disabled=(furnishing == "")
    )

    st.markdown("---")

    # ── GET RESULT button ──
    if st.button("GET RESULT"):
        if state == "" or furnishing == "":
            st.warning("⚠️ Please select both State and Furnishing.")
        else:
            house_data = {
                "state": state,
                "furnishing": furnishing,
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "property_size": int(property_size),
            }
            
            with st.spinner("Calculating estimated price..."):
                result = predict_price(house_data, model, scaler, label_encoders, feature_cols)

            if result:
                st.session_state.show_result = True
                st.session_state.result = result
                st.session_state.house_data = house_data

    # ── Result display ──
    if st.session_state.show_result and "result" in st.session_state:
        res = st.session_state.result
        hd = st.session_state.house_data

        
        st.success(f"**Price Range:** ₦{res['min_predicted_price']:,.0f} — ₦{res['max_predicted_price']:,.0f}")

        st.write(f"**Summary:** {hd['bedrooms']} Bed | {hd['bathrooms']} Bath | {hd['state']} | {hd['property_size']} sqm")
if __name__ == "__main__":
    main()
