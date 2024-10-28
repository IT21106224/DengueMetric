import sqlite3

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

create_table()
print("Table created successfully.")
