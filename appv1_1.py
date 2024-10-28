import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import base64

# Add CSS for background image
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def run():

    # Load the trained model
    model = joblib.load('best_rf_model.pkl')

    # Define feature names to ensure consistency
    feature_names = [
        'male', 'Female', 'DURATION OF FEVER', 'LETHARGY',
        'BLEEDING EVIDENCE EPISTAXIS', 'HEMATEMESIS',
        'BLEEDING EVIDENCE MALENA/pr bleed',
        'BLEEDING EVIDENCE INTERMENSTRUAL BLEEDING/menorhagia', 'VOMITING',
        'DIARRHEA', 'ABDOMINAL PAIN', 'PETECHIAE RASHES',
        'SYSTOLIC BP (HYPOTENSION)', 'ASCITES', 'HEPATOMEGALY',
        'LIVER TENDERNESS', 'SEROLOGY NS-1', 'SEROLOGY IgG', 'SEROLOGY IgM',
        'SEROLOGY others', 'Full blood count Hb (min)',
        'Full blood count WBC (min)', 'Full blood count neutrophil',
        'Full blood count lymphocytes', 'Full blood count monocytes',
        'Full blood count Platelet (min)', 'Full blood count hematocrit (max)'
    ]

    # Define preprocessing functions
    def preprocess_data(data):
        categorical_features = ['male', 'Female', 'LETHARGY', 'BLEEDING EVIDENCE EPISTAXIS',
                                'HEMATEMESIS', 'BLEEDING EVIDENCE MALENA/pr bleed',
                                'BLEEDING EVIDENCE INTERMENSTRUAL BLEEDING/menorhagia',
                                'VOMITING', 'DIARRHEA', 'ABDOMINAL PAIN', 'PETECHIAE RASHES',
                                'SYSTOLIC BP (HYPOTENSION)', 'ASCITES', 'HEPATOMEGALY',
                                'LIVER TENDERNESS', 'SEROLOGY NS-1', 'SEROLOGY IgG', 'SEROLOGY IgM',
                                'SEROLOGY others']

        for feature in categorical_features:
            data[feature] = data[feature].astype(int)

        # Handle scaling for numerical features
        numerical_features = ['DURATION OF FEVER', 'Full blood count Hb (min)',
                              'Full blood count WBC (min)', 'Full blood count neutrophil',
                              'Full blood count lymphocytes', 'Full blood count monocytes',
                              'Full blood count Platelet (min)', 'Full blood count hematocrit (max)']

        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        return data

    st.write("Most people with dengue have mild or no symptoms and will get better in 1–2 weeks. Rarely, dengue can be severe and lead to death. If symptoms occur, they usually begin 4–10 days after infection and last for 2–7 days. Symptoms may")

    # Inject CSS
    st.markdown("""
        <style>
        /* Increase font size for headers */
        .header-style {
            font-size: 32px; /* Adjust font size as needed */
            color: #000000; /* Dark blue */
            font-weight: bold;
        }
        .subheader-style {
            font-size: 32px; /* Adjust font size as needed */
            color: #000000; /* Lighter blue */
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Use containers to organize content
    with st.container():
        st.markdown('<h1 class="header-style">General Information</h1>', unsafe_allow_html=True)
        
        # Gender input
        male = st.selectbox('Gender', ('Male', 'Female'))
        duration_of_fever = st.text_input('Duration of Fever (Days)', '0.0')

        # Symptoms section
        with st.expander("Symptoms", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                lethargy = st.checkbox('Lethargy')
                bleeding_epistaxis = st.checkbox('Bleeding - Epistaxis')
                hematemesis = st.checkbox('Hematemesis')
                bleeding_malena = st.checkbox('Bleeding - Malena')
            with col2:
                bleeding_intermenstrual = st.checkbox('Intermenstrual Bleeding')
                vomiting = st.checkbox('Vomiting')
                diarrhea = st.checkbox('Diarrhea')
                abdominal_pain = st.checkbox('Abdominal Pain')
                petechiae_rashes = st.checkbox('Petechiae Rashes')

        # Blood Symptoms section
        with st.expander("Blood Symptoms", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                systolic_bp_hypotension = st.checkbox('Systolic BP Hypotension')
                ascites = st.checkbox('Ascites')
                hepatomegaly = st.checkbox('Hepatomegaly')
                liver_tenderness = st.checkbox('Liver Tenderness')
            with col2:
                serology_ns1 = st.selectbox('Serology NS-1', ('Positive', 'Negative'))
                serology_igg = st.selectbox('Serology IgG', ('Positive', 'Negative'))
                serology_igm = st.selectbox('Serology IgM', ('Positive', 'Negative'))
                serology_others = st.selectbox('Serology Others', ('Positive', 'Negative'))

        # Full Blood Count section
        with st.expander("Full Blood Count", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                full_blood_count_hb = st.text_input('Full Blood Count Hb (min)', '0.0')
                full_blood_count_wbc = st.text_input('Full Blood Count WBC (min)', '0.0')
            with col2:
                full_blood_count_neutrophil = st.text_input('Neutrophil Count', '0.0')
                full_blood_count_lymphocytes = st.text_input('Lymphocytes Count', '0.0')
                full_blood_count_monocytes = st.text_input('Monocytes Count', '0.0')
                full_blood_count_platelet = st.text_input('Platelet Count (min)', '0.0')
                full_blood_count_hematocrit = st.text_input('Hematocrit (max)', '0.0')

        # Create a dictionary of user input
        user_data = {
            'male': 1 if male == 'Male' else 0,
            'Female': 1 if male == 'Female' else 0,
            'DURATION OF FEVER': float(duration_of_fever),
            'LETHARGY': 1 if lethargy else 0,
            'BLEEDING EVIDENCE EPISTAXIS': 1 if bleeding_epistaxis else 0,
            'HEMATEMESIS': 1 if hematemesis else 0,
            'BLEEDING EVIDENCE MALENA/pr bleed': 1 if bleeding_malena else 0,
            'BLEEDING EVIDENCE INTERMENSTRUAL BLEEDING/menorhagia': 1 if bleeding_intermenstrual else 0,
            'VOMITING': 1 if vomiting else 0,
            'DIARRHEA': 1 if diarrhea else 0,
            'ABDOMINAL PAIN': 1 if abdominal_pain else 0,
            'PETECHIAE RASHES': 1 if petechiae_rashes else 0,
            'SYSTOLIC BP (HYPOTENSION)': 1 if systolic_bp_hypotension else 0,
            'ASCITES': 1 if ascites else 0,
            'HEPATOMEGALY': 1 if hepatomegaly else 0,
            'LIVER TENDERNESS': 1 if liver_tenderness else 0,
            'SEROLOGY NS-1': 1 if serology_ns1 == 'Positive' else 0,
            'SEROLOGY IgG': 1 if serology_igg == 'Positive' else 0,
            'SEROLOGY IgM': 1 if serology_igm == 'Positive' else 0,
            'SEROLOGY others': 1 if serology_others == 'Positive' else 0,
            'Full blood count Hb (min)': float(full_blood_count_hb),
            'Full blood count WBC (min)': float(full_blood_count_wbc),
            'Full blood count neutrophil': float(full_blood_count_neutrophil),
            'Full blood count lymphocytes': float(full_blood_count_lymphocytes),
            'Full blood count monocytes': float(full_blood_count_monocytes),
            'Full blood count Platelet (min)': float(full_blood_count_platelet),
            'Full blood count hematocrit (max)': float(full_blood_count_hematocrit)
        }

                # Convert dictionary to DataFrame for prediction
        user_data_df = pd.DataFrame([user_data])

        # Predict button
        if st.button('Predict'):
            prediction_proba = model.predict_proba(user_data_df)[0]
            prediction = model.predict(user_data_df)[0]

            # Display the result
            st.header("Prediction Result")
            if prediction == 1:
                st.write("The model predicts the patient is prone to severe dengue.")
                # Highlighting the relevant probability in blue
                st.markdown(f"<p style='color: blue;'>Positive: {prediction_proba[1]:.2f}</p>", unsafe_allow_html=True)
                st.write(f"Negative: {prediction_proba[0]:.2f}")
            else:
                st.write("The model predicts the patient is not prone to severe dengue.")
                st.write(f"Positive: {prediction_proba[1]:.2f}")
                st.write(f"Negative: {prediction_proba[0]:.2f}")

if __name__ == '__main__':
    run()

