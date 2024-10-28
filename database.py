import sqlite3
from datetime import datetime
from threading import Lock

# Lock for thread-safe database access
db_lock = Lock()

# Step 1: Create a thread-safe connection to the SQLite database
def create_connection():
    conn = sqlite3.connect('weather_cache.db', check_same_thread=False)
    return conn

# Step 2: Create the weather cache table if it doesn't exist
def create_table():
    with db_lock:
        conn = create_connection()
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS weather_cache (
                district TEXT,
                date TEXT,
                avg_temp REAL,
                avg_humidity REAL,
                total_rainfall REAL,
                last_updated TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

# Step 3: Fetch cached weather data from the database
def get_cached_weather(district, date):
    with db_lock:
        conn = create_connection()
        c = conn.cursor()
        c.execute("SELECT avg_temp, avg_humidity, total_rainfall FROM weather_cache WHERE district = ? AND date = ?", (district, date))
        result = c.fetchone()
        conn.close()
        if result:
            return {
                'avg_temp': result[0],
                'avg_humidity': result[1],
                'total_rainfall': result[2]
            }
        return None

# Step 4: Store weather data in the database cache
def store_weather_in_cache(district, date, avg_temp, avg_humidity, total_rainfall):
    with db_lock:
        conn = create_connection()
        c = conn.cursor()
        c.execute(
            "REPLACE INTO weather_cache (district, date, avg_temp, avg_humidity, total_rainfall, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
            (district, date, avg_temp, avg_humidity, total_rainfall, datetime.now())
        )
        conn.commit()
        conn.close()

# Step 5: Close the SQLite connection when not needed (optional, handled in each method)
def close_connection():
    with db_lock:
        conn = create_connection()
        conn.close()
