import streamlit as st
import requests
import xgboost as xgb
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
from PIL import Image
from datetime import datetime, timedelta
import concurrent.futures
import time
import sqlite3
from PIL import Image
import asyncio
import aiohttp


def run():

    def create_table():
        conn = sqlite3.connect('weather_data.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS weather_cache (
                district_name TEXT,
                date TEXT,
                avg_temp REAL,
                avg_humidity REAL,
                total_rainfall REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    # Load the XGBoost models
    hotspot_model = xgb.Booster()
    hotspot_model.load_model('xgboost_hotspot_model.json')  # Load the hotspot prediction model

    dengue_model = xgb.Booster()
    dengue_model.load_model('xgboost_dengue_model.json')  # Load the dengue cases prediction model

    from database import create_table, get_cached_weather, store_weather_in_cache, close_connection

    # Initialize the database table (only needs to run once, so you can call it here or during setup)
    create_table()

    # OpenWeather API key
    API_KEY = '2217ffb5e3b0ced736c0e97d3455e330'  # Replace with your actual API key
    BASE_URL = 'http://history.openweathermap.org/data/2.5/history/city'

    # Dictionary of districts with their latitude, longitude, and elevation
    district_data = {
    'Colombo': {'lat': 6.9271, 'lon': 79.8612, 'elevation': 5, 'city_id': 1248991},
        'Kandy': {'lat': 7.2906, 'lon': 80.6337, 'elevation': 477, 'city_id': 1241622},
        'Galle': {'lat': 6.0535, 'lon': 80.2200, 'elevation': 13, 'city_id': 1246294},
        'Matara': {'lat': 5.9549, 'lon': 80.5540, 'elevation': 10, 'city_id': 1235846},
        'Hambantota': {'lat': 6.1248, 'lon': 81.1185, 'elevation': 15, 'city_id': 1244926},
        'Ratnapura': {'lat': 6.6828, 'lon': 80.3990, 'elevation': 34, 'city_id': 1228737},
        'Kurunegala': {'lat': 7.4863, 'lon': 80.3647, 'elevation': 116, 'city_id': 1237980},
        'Puttalam': {'lat': 8.0322, 'lon': 79.8361, 'elevation': 2, 'city_id': 1229388},
        'Anuradhapura': {'lat': 8.3114, 'lon': 80.4037, 'elevation': 81, 'city_id': 1251071},
        'Polonnaruwa': {'lat': 7.9397, 'lon': 81.0180, 'elevation': 46, 'city_id': 1231681},
        'Trincomalee': {'lat': 8.5874, 'lon': 81.2152, 'elevation': 8, 'city_id': 1226260},
        'Batticaloa': {'lat': 7.7315, 'lon': 81.6778, 'elevation': 6, 'city_id': 1250161},
        'Ampara': {'lat': 7.3024, 'lon': 81.6748, 'elevation': 12, 'city_id': 1253408},
        'Badulla': {'lat': 6.9930, 'lon': 81.0551, 'elevation': 680, 'city_id': 1250615},
        'Monaragala': {'lat': 6.8729, 'lon': 81.3500, 'elevation': 156, 'city_id': 1230161},
        'Nuwara Eliya': {'lat': 6.9497, 'lon': 80.7891, 'elevation': 1868, 'city_id': 1232783},
        'Jaffna': {'lat': 9.6615, 'lon': 80.0255, 'elevation': 5, 'city_id': 1242110},
        'Kilinochchi': {'lat': 9.3802, 'lon': 80.3847, 'elevation': 15, 'city_id': 1235999},
        'Mannar': {'lat': 8.9813, 'lon': 79.9045, 'elevation': 7, 'city_id': 1230716},
        'Mullaitivu': {'lat': 9.2673, 'lon': 80.8122, 'elevation': 5, 'city_id': 1229752},
        'Vavuniya': {'lat': 8.7544, 'lon': 80.4982, 'elevation': 105, 'city_id': 1227889},
        'Kalutara': {'lat': 6.5836, 'lon': 79.9591, 'elevation': 4, 'city_id': 1241964},
        'Gampaha': {'lat': 7.0840, 'lon': 80.0098, 'elevation': 13, 'city_id': 1246267},
        'Kegalle': {'lat': 7.2533, 'lon': 80.3464, 'elevation': 169, 'city_id': 1241960},
        'Matale': {'lat': 7.4677, 'lon': 80.6234, 'elevation': 364, 'city_id': 1235849}
    }

    async def async_fetch_weather(session, url):
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return None

    async def get_weather_data_for_district(city_id, district_name, snapshot_days=14):
        today = datetime.now()
        weather_data = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(14):
                date = today - timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')

                # Check if the data is cached
                cached_data = get_cached_weather(district_name, date_str)
                if cached_data:
                    weather_data.append(cached_data)
                else:
                    unix_timestamp = int(date.timestamp())
                    url = f"{BASE_URL}?id={city_id}&type=hour&start={unix_timestamp}&appid={API_KEY}&units=metric"
                    tasks.append(asyncio.create_task(async_fetch_weather(session, url)))

            responses = await asyncio.gather(*tasks)
            for response in responses:
                if response and 'list' in response:
                    temp = [entry['main']['temp'] for entry in response['list']]
                    humidity = [entry['main']['humidity'] for entry in response['list']]
                    rainfall = [entry.get('rain', {}).get('1h', 0) for entry in response['list']]
                    
                    # Calculate averages and add to weather data
                    if temp and humidity and rainfall:
                        avg_temp = sum(temp) / len(temp)
                        avg_humidity = sum(humidity) / len(humidity)
                        total_rainfall = sum(rainfall)

                        # Cache data
                        store_weather_in_cache(district_name, date_str, avg_temp, avg_humidity, total_rainfall)
                        weather_data.append({'avg_temp': avg_temp, 'avg_humidity': avg_humidity, 'total_rainfall': total_rainfall})

        return {
            'avg_temp': sum(d['avg_temp'] for d in weather_data) / len(weather_data),
            'avg_humidity': sum(d['avg_humidity'] for d in weather_data) / len(weather_data),
            'total_rainfall': sum(d['total_rainfall'] for d in weather_data)
        } if weather_data else None

    def fetch_weather_for_all_districts_async():
        results = []
        
        async def fetch_all_districts():
            tasks = []
            for district, details in district_data.items():
                city_id = details['city_id']
                task = get_weather_data_for_district(city_id, district)
                tasks.append(asyncio.create_task(task))

            responses = await asyncio.gather(*tasks)
            for district, response in zip(district_data.keys(), responses):
                if response:
                    lat = district_data[district]['lat']
                    lon = district_data[district]['lon']
                    elevation = district_data[district]['elevation']
                    results.append({
                        'district': district,
                        'lat': lat,
                        'lon': lon,
                        'avg_temp': response['avg_temp'],
                        'avg_humidity': response['avg_humidity'],
                        'total_rainfall': response['total_rainfall'],
                    })

        asyncio.run(fetch_all_districts())
        return results
    # Retrieve cached weather data if within snapshot period (default: 14 days)
    def get_cached_weather(district_name, date, snapshot_days=14):
        conn = sqlite3.connect('weather_data.db')
        c = conn.cursor()
        snapshot_period = datetime.now() - timedelta(days=snapshot_days)
        c.execute('''
            SELECT avg_temp, avg_humidity, total_rainfall FROM weather_cache
            WHERE district_name = ? AND date = ? AND timestamp >= ?
        ''', (district_name, date, snapshot_period))
        data = c.fetchone()
        conn.close()
        if data:
            return {'avg_temp': data[0], 'avg_humidity': data[1], 'total_rainfall': data[2]}
        return None


    # Function to fetch weather data from OpenWeather API (14 days)
    def get_14_days_weather(city_id, district_name,snapshot_days=14):
        today = datetime.now()
        weather_data = {'temp': [], 'humidity': [], 'rainfall': []}

        for i in range(14):
            # Get the date for each of the past 14 days
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')

            # Initialize avg_temp, avg_humidity, and total_rainfall to avoid UnboundLocalError
            avg_temp = None
            avg_humidity = None
            total_rainfall = None

            # Check if the data is in the cache (using the function from database.py)
            cached_data = get_cached_weather(district_name, date_str)
            if cached_data:
                avg_temp = cached_data['avg_temp']
                avg_humidity = cached_data['avg_humidity']
                total_rainfall = cached_data['total_rainfall']
            else:
                # Fetch data from the OpenWeather API if not in the cache
                unix_timestamp = int(date.timestamp())
                url = f"{BASE_URL}?id={city_id}&type=hour&start={unix_timestamp}&appid={API_KEY}&units=metric"
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    if 'list' in data:
                        temp = [entry['main']['temp'] for entry in data['list']]
                        humidity = [entry['main']['humidity'] for entry in data['list']]
                        rainfall = [entry.get('rain', {}).get('1h', 0) for entry in data['list']]

                    # Convert rainfall, temp, and humidity values to float
                        temp = [float(t) for t in temp if t is not None]
                        humidity = [float(h) for h in humidity if h is not None]
                        rainfall = [float(r) for r in rainfall if r is not None]
                        
                        if temp and humidity and rainfall:
                            avg_temp = np.mean(temp) if temp else 0
                            avg_humidity = np.mean(humidity) if humidity else 0
                            total_rainfall = np.sum(rainfall) if rainfall else 0

                        # Store the fetched data in the cache
                        store_weather_in_cache(district_name, date_str, avg_temp, avg_humidity, total_rainfall)
                else:
                    print(f"Failed to fetch weather data for {district_name} on {date_str}")
                    continue

            # Add to the weather data dictionary
        if avg_temp is not None and avg_humidity is not None and total_rainfall is not None:
            weather_data['temp'].append(avg_temp)
            weather_data['humidity'].append(avg_humidity)
            weather_data['rainfall'].append(total_rainfall)

        else:
            print(f"No valid data for {district_name} on {date_str}. Skipping.")

        # Return averages over the 14 days
        if weather_data['temp']:
            avg_temp = np.mean(weather_data['temp'])
            avg_humidity = np.mean(weather_data['humidity'])
            total_rainfall = np.sum(weather_data['rainfall'])
            return {'avg_temp': avg_temp, 'avg_humidity': avg_humidity, 'total_rainfall': total_rainfall}
        else:
            print(f"No data available for {district_name} over the past 14 days.")
            return None
        
    def fetch_weather_for_all_districts():
        predictions = []

        def fetch_and_predict(district, details):
            print(f"Fetching data for {district}...")
            try:
                time.sleep(1)  # Add delay to prevent API rate limit issues
                weather_data = get_14_days_weather(details['city_id'], district)
                print(f"Weather data for {district}: {weather_data}")
                
                if weather_data:
                    lat = details['lat']
                    lon = details['lon']
                    elevation = details['elevation']

                    # Prepare data for dengue cases prediction
                    X_test = np.array([[weather_data['avg_temp'], weather_data['total_rainfall'], weather_data['avg_humidity'], lon, lat, elevation]])

                    # Normalize features
                    X_test_normalized = normalize_features(X_test)
                    dmatrix_regressor = xgb.DMatrix(X_test_normalized)

                    dengue_cases_prediction = dengue_model.predict(dmatrix_regressor, validate_features=False)

                    # Add predicted dengue cases as a feature
                    X_with_cases = np.append(X_test_normalized, [[dengue_cases_prediction[0]]], axis=1)
                    dmatrix_classifier = xgb.DMatrix(X_with_cases)

                    hotspot_prediction = hotspot_model.predict(dmatrix_classifier, validate_features=False)
                    hotspot_probability = hotspot_prediction[0]
                    is_hotspot = hotspot_probability > 0.5

                    return {
                        'district': district,
                        'lat': lat,
                        'lon': lon,
                        'hotspot_probability': hotspot_probability,
                        'predicted_dengue_cases': dengue_cases_prediction[0],
                        'hotspot': 'Hotspot' if is_hotspot else 'Not a Hotspot',
                        'color': [255, 0, 0] if is_hotspot else [0, 255, 0],
                    }
                else:
                    print(f"No weather data available for {district}")
            except Exception as e:
                print(f"Error processing district {district}: {e}")
            return None


        # Use ThreadPoolExecutor to parallelize the fetching of weather data
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_and_predict, district, details): district for district, details in district_data.items()}
            for future in concurrent.futures.as_completed(futures):
                district = futures[future]
                try:
                    result = future.result()
                    if result:
                        predictions.append(result)
                except Exception as e:
                    print(f"Error processing district {district}: {e}")

        return predictions


    # Function to normalize features with fixed range
    def normalize_features(X):
        # Normalize based on fixed range: temperature: 0-50°C, humidity: 0-100%, rainfall: 0-500mm
        normalized_X = np.copy(X)
        normalized_X[:, 0] = (X[:, 0] - 0) / (50 - 0)  # Normalize temperature
        normalized_X[:, 1] = (X[:, 1] - 0) / (500 - 0)  # Normalize rainfall
        normalized_X[:, 2] = (X[:, 2] - 0) / (100 - 0)  # Normalize humidity
        return normalized_X

    # Function to generate easy-to-understand, bullet-point explanations with emojis for environmental factors
    def generate_natural_language_explanation(is_hotspot, shap_values, features, feature_names):
        explanation = []

        # If the district is classified as a hotspot, explain why
        if is_hotspot:
            explanation.append("🌡️ This district was identified as a dengue hotspot because of the following environmental factors:")
            for i, feature in enumerate(features):
                shap_value = shap_values[0][i]  # Get SHAP value for the feature
                feature_name = feature_names[i]
                feature_value = feature

                # Only include environmental features with positive SHAP values
                if shap_value > 0 and feature_name in ["Temperature", "Rainfall", "Humidity"]:
                    if feature_name == "Temperature":
                        explanation.append(f"• 🌡️ Higher temperature (value: {feature_value:.2f}°C) increased the risk of dengue in this area.")
                    elif feature_name == "Rainfall":
                        explanation.append(f"• 🌧️ Increased rainfall (value: {feature_value:.2f}mm) contributed to a higher dengue risk.")
                    elif feature_name == "Humidity":
                        explanation.append(f"• 💧 High humidity (value: {feature_value:.2f}%) added to the likelihood of a dengue hotspot.")
        
        # If the district is not a hotspot, explain why not
        else:
            explanation.append("✅ This district was not identified as a dengue hotspot because of the following environmental factors:")
            for i, feature in enumerate(features):
                shap_value = shap_values[0][i]  # Get SHAP value for the feature
                feature_name = feature_names[i]
                feature_value = feature

                # Only include environmental features with negative SHAP values
                if shap_value < 0 and feature_name in ["Temperature", "Rainfall", "Humidity"]:
                    if feature_name == "Temperature":
                        explanation.append(f"• 🌡️ Lower temperature (value: {feature_value:.2f}°C) helped reduce the risk of a dengue hotspot.")
                    elif feature_name == "Rainfall":
                        explanation.append(f"• 🌧️ Reduced rainfall (value: {feature_value:.2f}mm) helped decrease the dengue risk.")
                    elif feature_name == "Humidity":
                        explanation.append(f"• 💧 Lower humidity (value: {feature_value:.2f}%) contributed to lowering the likelihood of a dengue hotspot.")
        
        return "\n".join(explanation)  # Return the explanation in bullet-point form with emojis

    # Function to trigger alert if hotspot
    def display_hotspot_alert(district):
        st.error(f"🚨 **ALERT: {district} is identified as a Dengue Hotspot!** 🚨")
        st.write("""
        **Preventive Measures:**
        - 🧹 **Eliminate Standing Water**: Remove stagnant water from containers, flowerpots, and gutters.
        - 🦟 **Use Mosquito Nets and Screens**: Keep windows and doors screened to avoid mosquito entry.
        - 🌿 **Apply Natural Mosquito Repellents**: Use plants like citronella, basil, or lavender.
        - 👕 **Wear Protective Clothing**: Cover arms and legs to reduce exposure.
        - 🕯️ **Consider Insect Repellents**: Use EPA-approved repellents for added protection.
        """)
        
    # Function to provide precautionary advice if dengue cases are above 100
    def display_precautionary_advice(district):
        st.warning(f"⚠️ **Precautionary Advice for {district}: High Dengue Cases Detected** ⚠️")
        st.write("""
        **Precautionary Steps:**
        - 🧹 **Reduce Water Collection**: Regularly empty containers where water can accumulate.
        - 🦟 **Install Screens and Nets**: Minimize mosquito entry by securing windows and doors.
        - 👕 **Wear Protective Clothing**: Use long sleeves and pants, especially at dawn and dusk.
        - 💧 **Use Mosquito Repellents**: Apply repellent on exposed skin to avoid bites.
        """)
        
    # Sidebar layout
    st.sidebar.title("🌍 Dengue Hotspot Prediction App")
    st.sidebar.write("Welcome to the **Dengue Hotspot Prediction App**! This app predicts whether a district in Sri Lanka is a dengue hotspot and the predicted number of cases based on weather and geographical data.")

    # Add an image to the sidebar

    image = Image.open("C:/Users/Stepheny/Desktop/new_1/Rp Project/dengue_hotspot_image.jpg.jpg")  # Replace with the correct path to your image
    st.sidebar.image(image, caption="Dengue Hotspots Prediction", use_column_width=True)

    # Sidebar options: Prediction for all districts or a particular district
    option = st.sidebar.selectbox(
        "🛠️ Choose Prediction Type",
        ["Prediction for All Districts", "Prediction for a Particular District"]
    )


    # Fetch and predict for all districts
    if option == "Prediction for All Districts":
        st.subheader("Prediction for All Districts")

        if st.sidebar.button("📊 Fetch Predictions for All Districts"):
            with st.spinner("Fetching weather data and making predictions..."):
                predictions = fetch_weather_for_all_districts()

                if predictions:
                    df = pd.DataFrame(predictions)

                    st.subheader("🗺️ Hotspot Map for All Districts")

                    tooltip = {
                        "html": "<b>District:</b> {district} <br/>"
                                "<b>Predicted Dengue Cases:</b> {predicted_dengue_cases:.0f} <br/>"
                                "<b>Hotspot Probability:</b> {hotspot_probability:.2f}",
                        "style": {
                            "backgroundColor": "steelblue",
                            "color": "white"
                        }
                    }

                    layer = pdk.Layer(
                        'ScatterplotLayer',
                        data=df,
                        get_position='[lon, lat]',
                        get_color='color',
                        get_radius=1000,
                        pickable=True,
                    )

                    view_state = pdk.ViewState(
                        latitude=7.8731,
                        longitude=80.7718,
                        zoom=7,
                        pitch=50,
                    )

                    r = pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip=tooltip,
                    )

                    st.pydeck_chart(r)
                else:
                    st.error("No valid data for any district. Please try again.")

    if option == "Prediction for a Particular District":
        st.header("Dengue Hotspot Prediction for a Particular District")
        selected_district = st.sidebar.selectbox("Choose a District", list(district_data.keys()))

        if st.sidebar.button(f"Fetch Prediction for {selected_district}"):
            st.spinner(f"Fetching data for {selected_district}...")
            weather_data = get_14_days_weather(district_data[selected_district]['city_id'], selected_district)
            
            if weather_data:
                lat = district_data[selected_district]['lat']
                lon = district_data[selected_district]['lon']
                elevation = district_data[selected_district]['elevation']
                
                # Display Weather Data
                st.write(f"### Weather Data for {selected_district}")
                st.write(f"- **Avg Temperature:** {weather_data['avg_temp']}°C")
                st.write(f"- **Avg Humidity:** {weather_data['avg_humidity']}%")
                st.write(f"- **Total Rainfall:** {weather_data['total_rainfall']} mm")

                # Predict Dengue Cases
                X_test = np.array([[weather_data['avg_temp'], weather_data['total_rainfall'], weather_data['avg_humidity'], lon, lat, elevation]])
                X_test_normalized = normalize_features(X_test)
                dmatrix_regressor = xgb.DMatrix(X_test_normalized)
                dengue_cases_prediction = dengue_model.predict(dmatrix_regressor, validate_features=False)
                
                # Append dengue cases to features and make hotspot prediction
                X_with_cases = np.append(X_test_normalized, [[dengue_cases_prediction[0]]], axis=1)
                dmatrix_classifier = xgb.DMatrix(X_with_cases)
                hotspot_prediction = hotspot_model.predict(dmatrix_classifier, validate_features=False)
                is_hotspot = hotspot_prediction[0] > 0.5

                # Display Results and Trigger Alert if Hotspot
                st.write(f"### Prediction Results for {selected_district}")
                st.write(f"- **Predicted Dengue Cases:** {dengue_cases_prediction[0]:.0f}")
                
                if is_hotspot:
                    display_hotspot_alert(selected_district)
                elif dengue_cases_prediction[0] > 100:
                    display_precautionary_advice(selected_district)
                else:
                    st.info(f"{selected_district} is **Not a Dengue Hotspot** ✅")

                # Generate SHAP Explanations
                st.write("### Explanation of Environmental Factors")
                explainer = shap.TreeExplainer(hotspot_model)
                shap_values = explainer.shap_values(dmatrix_classifier)
                explanation = generate_natural_language_explanation(is_hotspot, shap_values, X_with_cases[0], ["Temperature", "Rainfall", "Humidity", "Longitude", "Latitude", "Elevation", "Predicted Dengue Cases"])
                st.write(explanation)
                
                # Display SHAP Summary Plot
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_with_cases, feature_names=["Temperature", "Rainfall", "Humidity", "Longitude", "Latitude", "Elevation", "Predicted Dengue Cases"], show=False)
                st.pyplot(fig, clear_figure=True)
            else:
                st.error(f"No data for {selected_district}.")