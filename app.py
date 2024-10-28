import io
from PIL import Image
from urllib.parse import urljoin
import streamlit as st
import pandas as pd
import requests
import streamlit as st
from datetime import timedelta
import altair as alt
import pydeck as pdk
from streamlit_option_menu import option_menu

def run():
    FLASK_API_URL = "http://192.168.137.195:5000"

    def fetch_data():
        try:
            response = requests.get(f"{FLASK_API_URL}/images/all")
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame(data)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {e}")
            return None


    # Fetch the data
    data = fetch_data()

    # Streamlit app
    st.title('Dengue Mosquito Warning System')

    selected = option_menu(
        menu_title=None, 
        options=["Dashboard", "Current", "Lab"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container":{"padding":"0!important"},
            "nav-link":{
                "font-size":"18px",
                "text-align":"left",
                "margin":"0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected":{"background-color":"green"},
        },
    )

    st.divider()

    if selected == "Dashboard":
        # Total count of dengue mosquitoes
        total_count = data['mosquito_count'].sum()
        district_count = 1  

        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])    

        with col2:
            st.metric(label="Total Count of Dengue Mosquitoes", value=total_count)

        with col4:
            st.metric(label="Across Districts", value=district_count)

        st.divider()

        # Place the timeline and map side by side
        # col1, col2 = st.columns(2)

        col1, col2 = st.columns([1, 1])  # Here 3:2 ratio is used to make the chart larger


        with col1:
            # Timeline Visualization (Altair Chart)
            st.markdown("### Timeline of Sampled Mosquitoes")
            chart = alt.Chart(data).mark_line().encode(
                x='timestamp:T',
                y='mosquito_count:Q',
                tooltip=['timestamp:T', 'mosquito_count:Q']
            ).properties(
                width=300,
                height=250
            )
            st.altair_chart(chart)

        with col2:
            # Map Visualization
            st.markdown("### Traps Deployed")
            
            # Assuming we have lat/lon fields in our dataset for trap locations
            # Dummy example coordinates for demonstration
            map_data = pd.DataFrame({
                'lat': [6.879674],
                'lon': [79.857061],
            })
            
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=6.9271,  # Colombo's lat/lon
                    longitude=79.8612,
                    zoom=10,
                    pitch=50,
                ),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=map_data,
                        get_position='[lon, lat]',
                        get_color='[200, 30, 0, 160]',
                        get_radius=500,
                    ),
                ],
                width=100,  # Matching the chart width
                height=150,
            ))

            # Date selection dropdown at the top
        st.subheader('Filter by Date')
        dates = pd.to_datetime(data['timestamp']).dt.date.unique()
        selected_date = st.selectbox('Select Date', sorted(dates))

        # Filter data by selected date
        filtered_data = data[pd.to_datetime(data['timestamp']).dt.date == selected_date]

        if not filtered_data.empty:
            # Display filtered data in table format
            st.subheader('Filtered Mosquito Data')
            st.dataframe(filtered_data)
        else:
            st.write('No data available for the selected date.')



    if selected == "Current":
        st.title("Mosquitoes Captured Today")
        today_data = data[pd.to_datetime(data['timestamp']).dt.date == pd.to_datetime('today').date()]


        if not today_data.empty:
            for index, row in today_data.iterrows():
                st.markdown(f"*Time / Date:* {row['timestamp']}")

                # Create two columns for Annotated Image and Cropped Images
                col1, col2 = st.columns([1, 2])
                with col1:
                        st.markdown("### *Annotated Image*")
                        annotated_image_url = f"{FLASK_API_URL}{row['annotated_image_url']}"
                        st.image(annotated_image_url, width=200)

                with col2:
                    st.markdown("### *Cropped Mosquito Images and Classifications*")
                    for cropped_image in row['cropped_images']:
                        cropped_image_url = f"{FLASK_API_URL}{cropped_image['image_url']}"
                        classification = cropped_image['classification']

                        st.image(cropped_image_url, width=200)
                        st.markdown(f"*Classification:* {classification.capitalize()}")

                st.divider()

        else:
            st.write("No data available")


    if selected == "Lab":
        st.title("Real-Time Mosquito Detection and Classification")

        upload_url = urljoin(FLASK_API_URL, '/upload')
            
        # File uploader to select an image
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)

            # Convert the uploaded image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)

            # Send the image to Flask for processing
            files = {'image': img_bytes}
            response = requests.post(upload_url, files=files)

            # Handle the response
            if response.status_code == 200:
                result = response.json()

                st.write(f"Total Mosquitoes Detected: {result['mosquito_count']}")
                st.write(f"Dengue Mosquitoes Detected: {result['dengue_mosquito_count']}")

                # Display the annotated image
                annotated_image_url = f"{FLASK_API_URL}{result['annotated_image_url']}"
                st.image(annotated_image_url, caption="Annotated Image", use_column_width=True)

                # Display cropped mosquito images and classifications
                st.write("Cropped Mosquito Images and Classifications")
                for cropped_image in result['cropped_images']:
                    cropped_image_url = f"{FLASK_API_URL}{cropped_image['image_url']}"
                    st.image(cropped_image_url, caption=f"Classification: {cropped_image['classification']}", use_column_width=True)
            else:
                st.error("Error processing image. Please try again.")