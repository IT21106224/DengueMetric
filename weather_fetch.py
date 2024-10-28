import requests

# OpenWeather API key
API_KEY = '57d1d4454c8a14255cd4ed84aed81dee'

# Base URL for OpenWeather's Current Weather API
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'

# Dictionary of districts with city names in Sri Lanka
districts = {
    'Colombo': 'Colombo',
    'Kandy': 'Kandy',
    'Galle': 'Galle',
    'Matara': 'Matara',
    'Hambantota': 'Hambantota',
    'Ratnapura': 'Ratnapura',
    'Kurunegala': 'Kurunegala',
    'Puttalam': 'Puttalam',
    'Anuradhapura': 'Anuradhapura',
    'Polonnaruwa': 'Polonnaruwa',
    'Trincomalee': 'Trincomalee',
    'Batticaloa': 'Batticaloa',
    'Ampara': 'Ampara',
    'Badulla': 'Badulla',
    'Monaragala': 'Monaragala',
    'Nuwara Eliya': 'Nuwara Eliya',
    'Jaffna': 'Jaffna',
    'Kilinochchi': 'Kilinochchi',
    'Mannar': 'Mannar',
    'Mullaitivu': 'Mullaitivu',
    'Vavuniya': 'Vavuniya',
    'Kalutara': 'Kalutara',
    'Gampaha': 'Gampaha',
    'Kegalle': 'Kegalle',
    'Matale': 'Matale'
}

# Function to fetch weather data from OpenWeather for a given city
def get_weather_data(city_name):
    # Construct the request URL
    url = f"{BASE_URL}?q={city_name},LK&appid={API_KEY}&units=metric"
    
    # Make the request
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()

        # Extract the necessary data
        temperature = data['main']['temp']
        humidity = data['main']['humidity']

        # Rainfall is optional, so check if 'rain' is in the response
        rainfall = data['rain'].get('1h', 0) if 'rain' in data else 0

        return {
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall
        }
    else:
        return None  # Return None if there was an error

# Function to fetch and display weather data for all districts
def fetch_weather_for_all_districts():
    print("Fetching weather data for all districts...\n")
    
    # Loop through each district and fetch its weather data
    for district, city_name in districts.items():
        print(f"Fetching weather data for {district} ({city_name})...")
        weather_data = get_weather_data(city_name)
        
        if weather_data:
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Rainfall: {weather_data['rainfall']} mm\n")
        else:
            print(f"Error fetching data for {district}\n")

# Main function to run the script
if __name__ == "__main__":
    fetch_weather_for_all_districts()