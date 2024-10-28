import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="DengueMetric",
    page_icon="🦟",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling for a modern look
st.markdown("""
    <style>
    .main { background-color: #F0F8FF; color: #333333; }
    .sidebar { background-color: #2B3A42 !important; color: #FFFFFF !important; }
    .sidebar .option-menu-item a { color: #1ABC9C !important; font-size: 18px; }
    .stCard { background-color: #FFFFFF; padding: 20px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2); margin-bottom: 20px; }
    .header { color: #1ABC9C; font-size: 48px; font-weight: bold; }
    .subheader { color: #2C3E50; font-size: 24px; }
    .stButton>button { background-color: #1ABC9C; color: #FFFFFF; font-size: 18px; padding: 10px 20px; border-radius: 10px; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #16A085; }
    .stTitle { color: #2C3E50 !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "Future Dengue Hotspots", "Detect Mosquito", "Detect Dengue", "Predict Severe Prognosis", "Post Dengue Effect"],
        icons=["house", "cloud", "bug", "thermometer-half", "activity", "heart-fill"],
        menu_icon="cast",
        default_index=0
    )

# Navigate based on sidebar selection
if selected != "Home":
    st.session_state.page = selected  # Update session state based on selection

# Home Page
if st.session_state.page == "Home":
    st.markdown("<h1 class='header'>Welcome to DengueMetric 🦟</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Your Comprehensive Dengue Prediction and Monitoring System</p>", unsafe_allow_html=True)

    # Display banner image
    try:
        banner_image = Image.open("C:/Users/Stepheny/Desktop/moss.jpg")
        st.image(banner_image, use_column_width="auto", caption="Track and Predict Dengue Trends with Ease")
    except FileNotFoundError:
        st.error("Banner image not found. Please ensure the image path is correct.")

    # Cards with buttons for each feature
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div class='stCard'><h3 style='color: #16A085;'>🌦️ Dengue Hotspots</h3>
            <p>Predict areas with high risk of dengue outbreaks based on weather data in Sri Lanka.</p></div>""", unsafe_allow_html=True)
        if st.button("Explore Hotspots", key="hotspot"):
            st.session_state.page = "Future Dengue Hotspots"  # Change page in session state

    with col2:
        st.markdown("""<div class='stCard'><h3 style='color: #16A085;'>🩸 Detect Dengue</h3>
            <p>Analyze symptoms like fever and headache to assess the likelihood of dengue fever.</p></div>""", unsafe_allow_html=True)
        if st.button("Detect Dengue", key="detect_dengue"):
            st.session_state.page = "Detect Dengue"  # Change page in session state

    with col3:
        st.markdown("""<div class='stCard'><h3 style='color: #16A085;'>🩺 Predict Severe Prognosis</h3>
            <p>Predict the risk of severe complications in diagnosed dengue cases.</p></div>""", unsafe_allow_html=True)
        if st.button("Predict Prognosis", key="severe_prognosis"):
            st.session_state.page = "Predict Severe Prognosis"  # Change page in session state

    # Additional features in a new row
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("""<div class='stCard'><h3 style='color: #16A085;'>🦟 Detect Mosquito</h3>
            <p>Identify the presence of mosquitoes to assess local dengue transmission risks.</p></div>""", unsafe_allow_html=True)
        if st.button("Detect Mosquito", key="detect_mosquito"):
            st.session_state.page = "Detect Mosquito"  # Change page in session state

    with col5:
        st.markdown("""<div class='stCard'><h3 style='color: #16A085;'>💪 Post Dengue Effect</h3>
            <p>Evaluate potential long-term effects after recovery from dengue.</p></div>""", unsafe_allow_html=True)
        if st.button("Post Dengue Effect", key="post_dengue"):
            st.session_state.page = "Post Dengue Effect"  # Change page in session state

# Page Components
if st.session_state.page == "Future Dengue Hotspots":
    st.title("Future Dengue Hotspots 🌦️")
    import app1
    app1.run()

elif st.session_state.page == "Detect Mosquito":
    st.title("Detect Mosquito 🦟")
    import app
    app.run()

elif st.session_state.page == "Detect Dengue":
    st.title("Detect Dengue 🩸")
    import appv2_1
    appv2_1.run()

elif st.session_state.page == "Predict Severe Prognosis":
    st.title("Predict Severe Prognosis 🩺")
    import appv1_1
    appv1_1.run()

elif st.session_state.page == "Post Dengue Effect":
    st.title("Post Dengue Effect 💪")
    import qwe
    qwe.run()
