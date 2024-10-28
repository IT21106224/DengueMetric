import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Custom CSS to style the application
st.markdown(
    """
    <style>
    .stSelectbox label {
        font-size: 1.2rem; /* Increase label font size */
    }
    .stSelectbox {
        width: 150px; /* Set the width you prefer for the dropdown */
    }
    .stSelectbox .css-1wa3eu0 { /* Dropdown options style */
        font-size: 1rem; /* Set font size for options */
    }
    </style>
    """,
    unsafe_allow_html=True
)


def run():
    # Load the ensemble model
    ensemble_model = joblib.load('ensemble_model.pkl')

    # Mapping dictionaries for feature preprocessing
    yes_no_mapping = {True: 1, False: 0}

    # Streamlit UI components
    st.title("Disease Prediction Application")

    # General Information section with expander
    with st.expander("General Info", expanded=False):  # Closed by default
        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox("Sex", options=['', 'Male', 'Female', 'Other'], index=0, key="sex_dropdown")
        with col2:
            age = st.slider("Age", min_value=0, max_value=100, value=0)
        col3, col4 = st.columns(2)
        with col3:
            gestational_age = st.selectbox("Gestational Age", options=[
                '', '1st Quarter', '2nd Quarter', 'Gestational age ignored',
                'No', 'Not applicable', 'Ignored'
            ], index=0)
        with col4:
            days_capped = st.number_input("Days Capped", value=0)

    # Symptoms section with expander
    with st.expander("Symptoms", expanded=False):  # Closed by default
        col1, col2 = st.columns(2)
        with col1:
            fever = st.checkbox("Fever")
            myalgia = st.checkbox("Myalgia")
            headache = st.checkbox("Headache")
            rash = st.checkbox("Rash")
            tourniquet_test = st.checkbox("Tourniquet Test")
        with col2:
            vomiting = st.checkbox("Vomiting")
            nausea = st.checkbox("Nausea")
            back_pain = st.checkbox("Back Pain")
            conjunctivitis = st.checkbox("Conjunctivitis")
            retroorbital_pain = st.checkbox("Retroorbital Pain")

    # Medical History section with expander
    with st.expander("Medical History", expanded=False):  # Closed by default
        col1, col2 = st.columns(2)
        with col1:
            arthritis = st.checkbox("Arthritis")
            arthralgia = st.checkbox("Arthralgia")
            petechiae = st.checkbox("Petechiae")
            diabetes = st.checkbox("Diabetes")
        with col2:
            hematological_disease = st.checkbox("Hematological Disease")
            liver_disease = st.checkbox("Liver Disease")
            kidney_disease = st.checkbox("Kidney Disease")
            hypertension = st.checkbox("Hypertension")
            auto_immune = st.checkbox("Auto Immune")

    # Collect and map inputs to DataFrame
    input_data = {
        'Sex_Female': [1 if sex == 'Female' else 0],
        'Sex_Male': [1 if sex == 'Male' else 0],
        'Sex_other': [1 if sex == 'Other' else 0],
        'Gestational Age_No': [1 if gestational_age == 'No' else 0],
        'Gestational Age_Not applicable': [1 if gestational_age == 'Not applicable' else 0],
        'Gestational Age_Ignored': [1 if gestational_age == 'Gestational age ignored' else 0],
        'Gestational Age_1st Quarter': [1 if gestational_age == '1st Quarter' else 0],
        'Gestational Age_2nd Quarter': [1 if gestational_age == '2nd Quarter' else 0],
        'Gestational Age_3rd Quarter': [1 if gestational_age == '3rd Quarter' else 0],
        'Gestational Age_Gestational age ignored': [1 if gestational_age == 'Gestational age ignored' else 0],
        'Fever': [yes_no_mapping.get(fever, 2)],
        'Myalgia': [yes_no_mapping.get(myalgia, 2)],
        'Headache': [yes_no_mapping.get(headache, 2)],
        'Rash': [yes_no_mapping.get(rash, 2)],
        'Vomiting': [yes_no_mapping.get(vomiting, 2)],
        'Nausea': [yes_no_mapping.get(nausea, 2)],
        'Back Pain': [yes_no_mapping.get(back_pain, 2)],
        'Conjunctivitis': [yes_no_mapping.get(conjunctivitis, 2)],
        'Arthritis': [yes_no_mapping.get(arthritis, 2)],
        'Arthralgia': [yes_no_mapping.get(arthralgia, 2)],
        'Petechiae': [yes_no_mapping.get(petechiae, 2)],
        'Tourniquet test': [yes_no_mapping.get(tourniquet_test, 2)],
        'Retroorbital pain': [yes_no_mapping.get(retroorbital_pain, 2)],
        'Diabetes': [yes_no_mapping.get(diabetes, 2)],
        'Hematological Disease': [yes_no_mapping.get(hematological_disease, 2)],
        'Liver Disease': [yes_no_mapping.get(liver_disease, 2)],
        'Kidney Disease': [yes_no_mapping.get(kidney_disease, 2)],
        'Hypertension': [yes_no_mapping.get(hypertension, 2)],
        'Auto Immune': [yes_no_mapping.get(auto_immune, 2)],
        'Age_Group_0-10': [1 if 0 <= age <= 10 else 0],
        'Age_Group_11-20': [1 if 11 <= age <= 20 else 0],
        'Age_Group_21-30': [1 if 21 <= age <= 30 else 0],
        'Age_Group_31-40': [1 if 31 <= age <= 40 else 0],
        'Age_Group_41-50': [1 if 41 <= age <= 50 else 0],
        'Age_Group_51-60': [1 if 51 <= age <= 60 else 0],
        'Age_Group_61-70': [1 if 61 <= age <= 70 else 0],
        'Age_Group_71-80': [1 if 71 <= age <= 80 else 0],
        'Age_Group_81-90': [1 if 81 <= age <= 90 else 0],
        'Age_Group_91-100': [1 if 91 <= age <= 100 else 0],
        'Days_Capped_BoxCox_Scaled': [days_capped]  # Ensure this is the same scaling as in training
    }
    input_df = pd.DataFrame(input_data)
    scaler = StandardScaler()
    input_df['Days_Capped_BoxCox_Scaled'] = scaler.fit_transform(input_df[['Days_Capped_BoxCox_Scaled']])

    # Prediction
    if st.button("Predict"):
        prediction_proba = ensemble_model.predict_proba(input_df)

        # Display prediction probabilities
        proba_df = pd.DataFrame(prediction_proba, columns=['CHIKUNGUNIYA', 'DENGUE', 'OTHER DISEASE'])
        st.write("Prediction Probabilities:")
        st.write(proba_df)

        # Identify the disease with the highest probability
        predicted_disease = proba_df.idxmax(axis=1)[0]
        st.write(f"According to the probabilities, the person might have developed '{predicted_disease}'.")

if __name__ == "__main__":
    run()
