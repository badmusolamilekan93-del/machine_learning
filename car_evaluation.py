import streamlit as st
import pandas as pd
import joblib
import numpy as np
# set page config.
st.set_page_config(
page_title="Jiji car price prediction",
page_icon="🚗",
layout="wide",
initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_artifact():
    try:
        model =joblib.load("rf_model_prediction.pkl")
        print('model', model)
        scaler = joblib.load("scaler_features.pkl")
        print("scaler",scaler)
        label_encoder = joblib.load("label_encoders.pkl")
        print('label_encoder', label_encoder)
        feature_columns = joblib.load("feature_columns.pkl")
        print('feature_columns', feature_columns)
        return model, scaler, label_encoder, feature_columns,
    except FileNotFoundError as e:
        st.error(f"Model file not found{e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occured{e}")
        st.stop()
@st.cache_data
def  load_dataset():
    try:
        df = pd.read_csv("cleaned_jiji_car_dataset.csv")
        return df
    except FileNotFoundError as e:
        st.error(f"Dataset not found{e}")
        st.stop()

# get filtered option
def get_filtered_options(df, model=None, make=None, condition=None):

    filtered_df= df.copy()

    if model:
        filtered_df = filtered_df[filtered_df['model'] == model]
    if make:
        filtered_df = filtered_df[filtered_df['make']== make]
    if condition:
        filtered_df= filtered_df[filtered_df['condition'] == condition]

    options= {
        'years': sorted(filtered_df['year'].unique().tolist(), reverse=True),
        'makes': sorted(filtered_df['make'].unique().tolist()),
        'models': sorted(filtered_df['model'].unique().tolist()) if make else sorted(df['model'].unique().tolist()),
        'conditions': sorted(filtered_df['condition'].unique().tolist()) if make or model else sorted(df['condition'].unique().tolist()),
        'transmissions': sorted(filtered_df['transmission'].unique().tolist()) if make or model or condition else sorted(df['transmission'].unique().tolist())

    }

    return options

def predict_price(car_data, model, scaler, label_encoder, feature_column):
    
    try:
        input_data = pd.DataFrame([car_data])

        columns = ['make', 'model', 'condition', 'transmission']
        for col in columns:
            if col in label_encoder:
                try:
                    input_data[col + '_encoded'] = label_encoder[col].transform(input_data[col])
                except Exception as e:
                    st.Warning(f'Unknown {col}: {car_data[col]}.using default encoding')
                    input_data[col + '_encoded'] = 0
    # features preparing
        feature_dict = {
            'year': car_data['year'], 
            'model_encoded': input_data['model_encoded'].values[0],
            'make_encoded' : input_data['make_encoded'].values[0],
            'condition_encoded': input_data['condition_encoded'].values[0],
            'transmission_encoded': input_data['transmission_encoded'].values[0]
            
            }
        # create feature array
        features = np.array([[feature_dict[col] for col in feature_column]])

        # scale feature
        scale = scaler.transform(features)

        # model_prediction
        predicted_price = model.predict(scale)[0]

        margin_percentage = 0.15
        min_predicted_price = predicted_price * (1 - margin_percentage)
        max_predicted_price  = predicted_price * (1 + margin_percentage)
    
        return {
            "predicted_price": predicted_price,
            'min_predicted_price': min_predicted_price,
            'max_predicted_price': max_predicted_price
        }

    except Exception as e :
        st.error(f"An error occured: {e}")
        
def main():
    st.title("🚔Free Car Evaluation")
    st.write("Just fill all the fields and get immediate result.")

    # load model and dataset
    model, scaler, label_encoder, feature_columns =load_model_artifact()
    df = load_dataset()

    # initializing the session state
    if 'selected_make' not in st.session_state:
        st.session_state.selected_make = None

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

    if 'selectec_condition' not in st.session_state:
        st.session_state.seleceted_condition = None
        
    if 'show_result' not in  st.session_state:
        st.session_state.show_result = False    

    # get filter options 
    options = get_filtered_options(df)
    make = st.selectbox(
        "Make*",
        options=['']+ options ['makes'],
        format_func=lambda x: 'Select Make' if x == '' else x
    )
   
    # update make option
    if  make and make != st.session_state.selected_make:
        st.session_state.selected_make = make
        st.session_state.selected_model = None
        st.session_state.selected_condition = None
        st.session_state.show_result = None

    if make :
        options = get_filtered_options(df, make=make)
    # select model
    
    model_name = st.selectbox(
        "Model*",
        options =[""] + options["models"],
        format_func=lambda x: 'Select Model' if x =='' else x,
        disabled= not make
    )

    # update model option
    if  model_name and model_name != st.session_state.selected_model:
        st.session_state.selected_model = model_name
        st.session_state.selected_condition = None
        st.session_state.show_result = None
    if make and model_name :
        options = get_filtered_options(df, make=make, model=model_name)
    # select year
    year = st.selectbox(
        "Year of manufacture*",
        options= [''] + options['years'],
        format_func= lambda x: 'Select Year' if x == '' else str(x),
        disabled= not (make and model_name)
    )

    if year:
        st.session_state.show_reslut = False
    # select condition 
    condition = st.selectbox(
        "Condition*",
        options = [""] + options["conditions"],
        format_func=lambda x: "Select Condition" if x == "" else x,
        disabled= not (make and model and year)
    ) 
    # update condition options
    if condition and condition != st.session_state.selected_condition:
        st.session_state.selected_codition = condition
        st.session_state.show_result = False
    if condition and model_name and make:
        options = get_filtered_options(df, make=make, model= model_name, condition=condition)

    # select transmission 
    transmission = st.selectbox(
        "Transmission*",
        options = [""] + options["transmissions"],
        format_func=lambda x: "Select Transmission" if x == "" else x,
        disabled= not (make and model and year and condition)
    ) 
    if transmission:
        st.session_state.show_result = False

    if st.button("GET RESULT"):
        if not all([make, model, year, condition, transmission]):
            st.warning("⚠ Please fill all the field")
        else:
            car_data={
                "make": make,
                "model": model_name,
                "year":year,
                "condition":condition,
                "transmission":transmission
            }
            with st.spinner("Calculating predicted price..."):
                result=predict_price(car_data, model, scaler, label_encoder, feature_columns)
            if result:
                st.session_state.show_result= True
                st.session_state.result = result
                st.session_state.car_data = car_data
    # Display output
    if st.session_state.show_result and 'result' in st.session_state:
        result = st.session_state.result
        car_data = st.session_state.car_data

        st.markdown("---")
        st.subheader(f"Estimated car price: {car_data['year']} {car_data["make"]} {car_data["model"]}")

        # predicted price
        st.success(f"## ₦{result['predicted_price']:,.0f}")

        # price range
        st.success(f"**price range** {result['min_predicted_price']:,.0f} - {result['max_predicted_price']:,.0f}")

        # display car detail
        st.write(f"**condition :** {car_data['condition']} | **Transmision :** {car_data['transmission']}")












if __name__ == "__main__":
    main()