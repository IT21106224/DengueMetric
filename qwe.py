import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import time
from streamlit_shap import st_shap  # Import streamlit_shap for interactive SHAP plots

def run():
    # Load the trained model
    model = joblib.load('model.sav')

    # Feature names for the model
    feature_names = ['gender', 'in_patient', 'age', 'diabetic', 'hypertension', 'other_disease', 
            'heart_disease', 'obesity', 'smoking_status', 'severe_symptoms', 'weak_immune']

    # Title of the app
    # st.title("Post-Dengue Effects Prediction⚕")
    st.image("china.jpg", caption="Post dengue Effects", use_column_width=True)

    # Create empty columns to center the form
    col1, col2, col3 = st.columns([1, 10, 1])

    # Place the form in the middle column (col2) to center it on the page
    with col2:
        st.header("Please enter your health details📊")

        # Form to collect user input on the main page
        with st.form("user_input_form"):
            st.markdown("<h4 style='color: #333;'>Personal Information</h4>", unsafe_allow_html=True)
            # Collect user inputs inside the form
            age = st.slider('Please enter your age', 0, 100, 30)
            gender = st.selectbox('Select your gender', options=[ 'Male', 'Female'])
          
            height_cm = st.number_input('What is your height in centimeters?', min_value=50, max_value=250, value=170)
            weight_kg = st.number_input('What is your weight in kilograms?', min_value=20, max_value=300, value=70)
           

            # in_patient = st.radio('Have you been hospitalized for your dengue infection?', ('Yes', 'No'))
            st.markdown("<h4 style='color: #333;'>Health Information</h4>", unsafe_allow_html=True)
            col4, col5 = st.columns(2)
            with col4:
                diabetic = st.checkbox('Have you been diagnosed with diabetes?')
            with col5:
                hypertension = st.checkbox('Do you have high blood pressure (hypertension)?')
                


            # diabetic = st.radio('Have you been diagnosed with diabetes?', ('Yes', 'No'))
            # hypertension = st.radio('Do you have high blood pressure (hypertension)?', ('Yes', 'No'))
            other_disease = st.radio('Do you have any other existing medical conditions (Asthma, Chronic Kidney Disease, Cancer, Autoimmune Diseases)?', ('Yes', 'No'))
            heart_disease = st.radio('Have you been diagnosed with any heart conditions, such as Cardiovascular Diseases or Heart Failure?', ('Yes', 'No'))
            smoking_status = st.selectbox('Select your gender', options=[ 'I currently smoke', 'I occasionally smoke', 'I have never smoked', 'I used to smoke but quit'])
            severe_symptoms = st.radio('Have you been hospitalized for dengue due to severe symptoms, such as persistent vomiting, severe abdominal pain, or bleeding?', ('Yes', 'No'))
            weak_immune = st.radio('Do you have a weak immune system due to a long-term illness or medication, or have you experienced frequent infections?', ('Yes', 'No'))

            # Submit button
            submitted = st.form_submit_button("Submit")

    # If the form is submitted, process the user input
    if submitted:
        # Create an empty placeholder for the success message
        success_placeholder = st.empty()
        success_placeholder.success("Thank you for submitting your information! Processing results...")
        
        # Pause for 2 seconds
        time.sleep(3)
        
        # Clear the success message after the sleep duration
        success_placeholder.empty()
        

        # BMI calculation
        height_m = height_cm / 100  # Convert cm to meters
        bmi = weight_kg / (height_m ** 2)  # BMI formula

        # Classify based on BMI
        obesity = 0 if bmi < 25 else 1  # 0 for Underweight or Normal weight, 1 for Overweight or Obese

        # Create a dictionary of user input with feature names exactly as per the model's training data
        user_data = {
            'gender': 0 if gender == 'Female' else 1,
            'in_patient': 0,
            'age': age,
            'diabetic': 1 if diabetic == 1 else 0,
            'hypertension': 1 if hypertension == 1 else 0,
            'other_disease': 1 if other_disease == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'obesity': obesity,
            'Smoking_status': 1 if smoking_status in ['I currently smoke', 'I occasionally smoke'] else 0,
            'severe_symptoms': 1 if severe_symptoms == 'Yes' else 0,
            'weak_immune': 1 if weak_immune == 'Yes' else 0,
        }

        # Create a DataFrame from the dictionary and ensure the columns are in the correct order
        report_data = pd.DataFrame(user_data, index=[0])

        # Reindex the columns to match the feature names from the training data
        report_data = report_data.reindex(columns=feature_names)

        # Make predictions using the user data
        prediction = model.predict(report_data)
        #st.write(f"{user_data}")

        # Handle and display the predictions for each effect
        fatigue_result = 'Fatigue predicted' if prediction[0][0] == 1 else 'No Fatigue'
        skin_hair_result = 'Skin/Hair Issues predicted' if prediction[0][1] == 1 else 'No Skin/Hair Issues'
        joint_pain_result = 'Joint/Muscle Pain predicted' if prediction[0][2] == 1 else 'No Joint/Muscle Pain'

        # # Display the results
        # st.subheader("\n\nPrediction Results")
        # st.write(f"Fatigue: {fatigue_result}")
        # st.write(f"Skin/Hair Issues: {skin_hair_result}")
        # st.write(f"Joint/Muscle Pain: {joint_pain_result}")


        # Get probabilities for the user
        fatigue_prob = model.estimators_[0].predict_proba(report_data)[0][1]  # Probability of fatigue = 1
        skin_hair_prob = model.estimators_[1].predict_proba(report_data)[0][1]  # Probability of skin/hair = 1
        joint_muscle_pain_prob = model.estimators_[2].predict_proba(report_data)[0][1]  # Probability of joint/muscle pain = 1

        # st.write(f"Fatigue: {fatigue_prob}")
        # st.write(f"Skin/Hair Issues: {skin_hair_prob}")
        # st.write(f"Joint/Muscle Pain: {joint_muscle_pain_prob}")

        fatigue_pred, skin_hair_pred, joint_muscle_pain_pred = prediction[0]

        # st.write(f"Fatigue: {fatigue_pred}")
        # st.write(f"Skin/Hair Issues: {skin_hair_pred}")
        # st.write(f"Joint/Muscle Pain: {joint_muscle_pain_pred}")    

        # Fatigue
        explainer_fatigue = shap.TreeExplainer(model.estimators_[0])  # Fatigue model
        shap_values_fatigue = explainer_fatigue.shap_values(report_data)

        # Skin/Hair
        explainer_skin_hair = shap.TreeExplainer(model.estimators_[1])  # Skin/Hair model
        shap_values_skin_hair = explainer_skin_hair.shap_values(report_data)

        # Joint/Muscle Pain
        explainer_joint_muscle_pain = shap.TreeExplainer(model.estimators_[2])  # Joint/Muscle Pain model
        shap_values_joint_muscle_pain = explainer_joint_muscle_pain.shap_values(report_data)

        # SHAP values for fatigue prediction (fatigue = 1)
        shap_values_fatigue_1 = shap_values_fatigue[1]

        # Create DataFrame of feature contributions
        fatigue_contributions = pd.DataFrame({
            'Feature': report_data.columns,
            'SHAP Value': shap_values_fatigue_1[0]
        })

        # Sort by absolute SHAP value
        fatigue_contributions['abs_SHAP'] = np.abs(fatigue_contributions['SHAP Value'])
        fatigue_contributions = fatigue_contributions.sort_values(by='abs_SHAP', ascending=False)

        # Top 3 factors
        top_fatigue_factors = fatigue_contributions.head(3)

        # SHAP values for skin/hair prediction (skin_hair = 1)
        shap_values_skin_hair_1 = shap_values_skin_hair[1]

        # Create DataFrame of feature contributions
        skin_hair_contributions = pd.DataFrame({
            'Feature': report_data.columns,
            'SHAP Value': shap_values_skin_hair_1[0]
        })

        # Sort by absolute SHAP value
        skin_hair_contributions['abs_SHAP'] = np.abs(skin_hair_contributions['SHAP Value'])
        skin_hair_contributions = skin_hair_contributions.sort_values(by='abs_SHAP', ascending=False)

        # Top 3 factors
        top_skin_hair_factors = skin_hair_contributions.head(3)

        # SHAP values for joint/muscle pain prediction (joint_muscle_pain = 1)
        shap_values_joint_muscle_pain_1 = shap_values_joint_muscle_pain[1]

        # Create DataFrame of feature contributions
        joint_muscle_pain_contributions = pd.DataFrame({
            'Feature': report_data.columns,
            'SHAP Value': shap_values_joint_muscle_pain_1[0]
        })

        # Sort by absolute SHAP value
        joint_muscle_pain_contributions['abs_SHAP'] = np.abs(joint_muscle_pain_contributions['SHAP Value'])
        joint_muscle_pain_contributions = joint_muscle_pain_contributions.sort_values(by='abs_SHAP', ascending=False)

        # Top 3 factors
        top_joint_muscle_pain_factors = joint_muscle_pain_contributions.head(3)


        def explain_fatigue_prediction(contributions, prediction, prob, user_input_df):
            # Get the values of the user's features as a dictionary {feature_name: value}
            user_values_dict = user_input_df.iloc[0].to_dict()
            
            # Filter contributions to only include features where the user's value is 1 or always relevant (age, gender)
            valid_factors = contributions[contributions['Feature'].apply(lambda x: (x == 'age') or (x == 'gender') or (user_values_dict.get(x, 0) == 1))]

            if prediction == 1:
                pred_text = f"🟧You have a {prob*100:.1f}% chance of experiencing fatigue."
                
                if len(valid_factors) == 1:
                    explanation = f"{pred_text} The most important factor contributing to this prediction is:"
                else:
                    explanation = f"{pred_text} The most important factors contributing to this prediction are:"
                
                for idx, row in valid_factors.iterrows():
                    feature = row['Feature']
                    if feature == 'age':
                        explanation += f"\n- Your age plays a significant role."
                    elif feature == 'gender':
                        gender_text = "male" if user_values_dict['gender'] == 1 else "female"
                        explanation += f"\n- Being {gender_text} affects this prediction."
                    elif feature == 'smoking_status':
                        explanation += f"\n- Smoking status has a strong positive influence."
                    else:
                        explanation += f"\n- {feature.replace('_', ' ').title()} impacts this prediction."
            else:
                explanation = f"🟦 With a {prob*100:.1f}% chance, it’s less likely that you will experience fatigue."
            
            return explanation

        # fatigue_explanation
        fatigue_explanation = explain_fatigue_prediction(top_fatigue_factors, fatigue_pred, fatigue_prob, report_data)


        def explain_skin_hair_prediction(contributions, prediction, prob, user_input_df):
            # Get the values of the user's features as a dictionary {feature_name: value}
            user_values_dict = user_input_df.iloc[0].to_dict()
            
            # Filter contributions to include features where the user's value is 1 or always relevant (age, gender)
            valid_factors = contributions[contributions['Feature'].apply(lambda x: (x == 'age') or (x == 'gender') or (user_values_dict.get(x, 0) == 1))]

            if prediction == 1:
                pred_text = f"🟧You have a {prob*100:.1f}% chance of experiencing skin or hair issues."
                
                if len(valid_factors) == 1:
                    explanation = f"{pred_text} The most important factor contributing to this prediction is:"
                else:
                    explanation = f"{pred_text} The most important factors contributing to this prediction are:"
                
                for idx, row in valid_factors.iterrows():
                    feature = row['Feature']
                    if feature == 'age':
                        explanation += f"\n- Your age plays a significant role."
                    elif feature == 'gender':
                        gender_text = "male" if user_values_dict['gender'] == 1 else "female"
                        explanation += f"\n- Being {gender_text} affects this prediction."
                    elif feature == 'smoking_status':
                        explanation += f"\n- Smoking status has a strong positive influence."
                    else:
                        explanation += f"\n- {feature.replace('_', ' ').title()} impacts this prediction."
            else:
                explanation = f"🟦 With a {prob*100:.1f}% chance, it’s less likely that you will experience skin or hair issues."
            
            return explanation

        # Now you can run this explanation
        skin_hair_explanation = explain_skin_hair_prediction(top_skin_hair_factors, skin_hair_pred, skin_hair_prob, report_data)

        

        def explain_joint_muscle_pain_prediction(contributions, prediction, prob, user_input_df):
            # Get the values of the user's features as a dictionary {feature_name: value}
            user_values_dict = user_input_df.iloc[0].to_dict()
            
            # Filter contributions to include features where the user's value is 1 or always relevant (age, gender)
            valid_factors = contributions[contributions['Feature'].apply(lambda x: (x == 'age') or (x == 'gender') or (user_values_dict.get(x, 0) == 1))]

            if prediction == 1:
                pred_text = f"🟧You have a {prob*100:.1f}% chance of experiencing joint or muscle pain."
                
                if len(valid_factors) == 1:
                    explanation = f"{pred_text} The most important factor contributing to this prediction is:"
                else:
                    explanation = f"{pred_text} The most important factors contributing to this prediction are:"
                
                for idx, row in valid_factors.iterrows():
                    feature = row['Feature']
                    if feature == 'age':
                        explanation += f"\n- Your age plays a significant role."
                    elif feature == 'gender':
                        gender_text = "male" if user_values_dict['gender'] == 1 else "female"
                        explanation += f"\n- Being {gender_text} affects this prediction."
                    elif feature == 'smoking_status':
                        explanation += f"\n- Smoking status has a strong positive influence."
                    else:
                        explanation += f"\n- {feature.replace('_', ' ').title()} impacts this prediction."
            else:
                explanation = f"🟦 With a {prob * 100:.1f}% chance, it’s less likely that you will experience joint or muscle pain."
            
            return explanation

        # Now you can run this explanation
        joint_muscle_pain_explanation = explain_joint_muscle_pain_prediction(top_joint_muscle_pain_factors, joint_muscle_pain_pred, joint_muscle_pain_prob, report_data)

        st.write(f"{fatigue_explanation}")
        st.write(f"{skin_hair_explanation}")
        st.write(f"{joint_muscle_pain_explanation}")


        # Precautionary Advice
        st.subheader("🛡️ Precautionary Advice")

        # Fatigue Precaution
        if prediction[0][0] == 1:  # Fatigue Detected
            st.markdown("### 😴 **Fatigue Precaution**")
            st.markdown("""
            - **💤 Rest:** Get plenty of rest to help your body recover.
            - **🥗 Balanced Diet:** Eat nutrient-rich meals to regain energy.
            - **⚠️ Avoid Strain:** Refrain from engaging in strenuous activities.
            - **🩺 Monitor:** If fatigue persists for a long time, consult a healthcare provider.
            """)

        # Skin/Hair Issues Precaution
        if prediction[0][1] == 1:  # Skin/Hair Issues Detected
            st.markdown("### 🧴 **Skin & Hair Precaution**")
            st.markdown("""
            - **💦  Moisturize:** Keep your skin hydrated with a good moisturizer.
            - **🧼 Mild Products:** Use gentle skincare products to avoid irritation.
            - **❌ Avoid Harsh Chemicals:** Stay away from products with strong chemicals.
            - **🍏Balanced Diet:** Include vitamins and nutrients in your diet for healthy skin and hair.
            """)

        # Joint/Muscle Pain Precaution
        if prediction[0][2] == 1:  # Joint/Muscle Pain Detected
            st.markdown("### 🏃‍♂️ **Joint & Muscle Pain Precaution**")
            st.markdown("""
            - **🧘‍♂️ Light Exercises:** Gentle activities like stretching or yoga can relieve tension.
            - **⏳ Rest & Recovery:** Take breaks to avoid overworking your joints and muscles.
            - **🏥 Consult a Doctor:** If pain persists or worsens, seek medical advice.
            """)

    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # SHAP Explainer
    # st.subheader("Model Interpretation using SHAP")

    # # Function to get SHAP values for each target
    # def get_shap_values(model, X_data):
    #     explainer = shap.TreeExplainer(model)
    #     shap_values = explainer.shap_values(X_data)
    #     return explainer, shap_values

    # # Extract the base models for each target
    # models = model.estimators_

    # # Force plot for 'Fatigue'
    # explainer_fatigue, shap_values_fatigue = get_shap_values(models[0], report_data)
    # st.write("### SHAP Force Plot (Fatigue)")
    # shap.initjs()  # Initialize JS for interactive plots
    # st_shap(shap.force_plot(explainer_fatigue.expected_value[0], 
    #                         shap_values_fatigue[0], 
    #                         report_data.iloc[0], 
    #                         feature_names=feature_names))

    # # Force plot for 'Skin/Hair Issues'
    # explainer_skin_hair, shap_values_skin_hair = get_shap_values(models[1], report_data)
    # st.write("### SHAP Force Plot (Skin/Hair Issues)")
    # st_shap(shap.force_plot(explainer_skin_hair.expected_value[0], 
    #                         shap_values_skin_hair[0], 
    #                         report_data.iloc[0], 
    #                         feature_names=feature_names))

    # # Force plot for 'Joint/Muscle Pain'
    # explainer_joint_pain, shap_values_joint_pain = get_shap_values(models[2], report_data)
    # st.write("### SHAP Force Plot (Joint/Muscle Pain)")
    # st_shap(shap.force_plot(explainer_joint_pain.expected_value[0], 
    #                         shap_values_joint_pain[0], 
    #                         report_data.iloc[0], 
    #                         feature_names=feature_names))