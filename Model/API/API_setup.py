import pandas as pd
import requests
from datetime import datetime

# coordinates are for the capital of each province
# :cite[*] stands from which source I had the coordinates retrieved using an AI-tool (DeepSeek)
# :cite[1] → Spain’s National Geographic Institute (IGN)
# :cite[2] → Refers to Spain’s IGN (Instituto Geográfico Nacional)
# :cite[3] → Refers to EUROSTAT NUTS

provinces = [
    {"name": "Albacete", "lat": 39.0000, "lon": -1.8500},  # :cite[2]:cite[3]
    {"name": "Alicante", "lat": 38.3453, "lon": -0.4831},  # :cite[2]:cite[3]
    {"name": "Almería", "lat": 36.8381, "lon": -2.4597},   # :cite[2]:cite[3]
    {"name": "Álava", "lat": 42.8500, "lon": -2.6833},     # Capital: Vitoria-Gasteiz :cite[1]:cite[3]
    {"name": "Asturias", "lat": 43.3600, "lon": -5.8450},  # Capital: Oviedo :cite[1]:cite[2]
    {"name": "Ávila", "lat": 40.6561, "lon": -4.7000},     # :cite[2]:cite[3]
    {"name": "Badajoz", "lat": 38.8800, "lon": -6.9700},   # :cite[2]:cite[3]
    {"name": "Baleares", "lat": 39.5667, "lon": 2.6500},   # Capital: Palma :cite[1]:cite[2]
    {"name": "Barcelona", "lat": 41.3825, "lon": 2.1769},  # :cite[1]:cite[2]
    {"name": "Biscay", "lat": 43.2627, "lon": -2.9253},    # Capital: Bilbao :cite[1]:cite[3]
    {"name": "Burgos", "lat": 42.3500, "lon": -3.7000},    # :cite[2]:cite[3]
    {"name": "Cáceres", "lat": 39.4833, "lon": -6.3667},   # :cite[2]:cite[3]
    {"name": "Cádiz", "lat": 36.5350, "lon": -6.2975},     # :cite[2]:cite[3]
    {"name": "Cantabria", "lat": 43.4628, "lon": -3.8050}, # Capital: Santander :cite[1]:cite[2]
    {"name": "Castellón", "lat": 39.9833, "lon": -0.0333}, # Capital: Castellón de la Plana :cite[1]:cite[3]
    {"name": "Ciudad Real", "lat": 38.9833, "lon": -3.9333}, # :cite[2]:cite[3]
    {"name": "Córdoba", "lat": 37.8845, "lon": -4.7796},   # :cite[2]:cite[4]
    {"name": "A Coruña", "lat": 43.3667, "lon": -8.3833},  # :cite[2]:cite[3]
    {"name": "Cuenca", "lat": 40.0667, "lon": -2.1333},    # :cite[2]:cite[3]
    {"name": "Gipuzkoa", "lat": 43.3200, "lon": -1.9920},  # Capital: San Sebastián :cite[1]:cite[3]
    {"name": "Girona", "lat": 41.9833, "lon": 2.8167},     # :cite[2]:cite[3]
    {"name": "Granada", "lat": 37.1781, "lon": -3.6008},   # :cite[2]:cite[3]
    {"name": "Guadalajara", "lat": 40.6333, "lon": -3.1667}, # :cite[2]:cite[3]
    {"name": "Huelva", "lat": 37.2500, "lon": -6.9500},    # :cite[2]:cite[3]
    {"name": "Huesca", "lat": 42.1333, "lon": -0.4167},    # :cite[2]:cite[3]
    {"name": "Jaén", "lat": 37.7667, "lon": -3.7711},      # :cite[2]:cite[3]
    {"name": "La Rioja", "lat": 42.4500, "lon": -2.4500},  # Capital: Logroño :cite[1]:cite[3]
    {"name": "Las Palmas", "lat": 28.1272, "lon": -15.4314}, # :cite[2]:cite[3]
    {"name": "León", "lat": 42.6056, "lon": -5.5700},      # :cite[2]:cite[3]
    {"name": "Lleida", "lat": 41.6167, "lon": 0.6333},     # :cite[2]:cite[3]
    {"name": "Lugo", "lat": 43.0128, "lon": -7.5608},      # :cite[2]:cite[3]
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038},    # :cite[1]:cite[2]
    {"name": "Málaga", "lat": 36.7194, "lon": -4.4200},    # :cite[2]:cite[3]
    {"name": "Murcia", "lat": 37.9861, "lon": -1.1303},    # :cite[2]:cite[3]
    {"name": "Navarre", "lat": 42.8167, "lon": -1.6500},   # Capital: Pamplona :cite[1]:cite[3]
    {"name": "Ourense", "lat": 42.3364, "lon": -7.8633},   # :cite[2]:cite[3]
    {"name": "Palencia", "lat": 42.0167, "lon": -4.5333},  # :cite[2]:cite[3]
    {"name": "Pontevedra", "lat": 42.4333, "lon": -8.6500},# :cite[2]:cite[3]
    {"name": "Salamanca", "lat": 40.9667, "lon": -5.6500}, # :cite[2]:cite[3]
    {"name": "Santa Cruz de Tenerife", "lat": 28.4667, "lon": -16.2500}, # :cite[2]:cite[3]
    {"name": "Segovia", "lat": 40.9500, "lon": -4.1167},   # :cite[2]:cite[3]
    {"name": "Seville", "lat": 37.3900, "lon": -5.9900},   # :cite[2]:cite[3]
    {"name": "Soria", "lat": 41.7667, "lon": -2.4667},     # :cite[2]:cite[3]
    {"name": "Tarragona", "lat": 41.1167, "lon": 1.2500},  # :cite[2]:cite[3]
    {"name": "Teruel", "lat": 40.3333, "lon": -1.1000},    # :cite[2]:cite[3]
    {"name": "Toledo", "lat": 39.8667, "lon": -4.0167},    # :cite[2]:cite[3]
    {"name": "Valencia", "lat": 39.4700, "lon": -0.3764},  # :cite[2]:cite[3]
    {"name": "Valladolid", "lat": 41.6500, "lon": -4.7167},# :cite[2]:cite[3]
    {"name": "Zamora", "lat": 41.5000, "lon": -5.7500},    # :cite[2]:cite[3]
    {"name": "Zaragoza", "lat": 41.6500, "lon": -0.8833},  # :cite[2]:cite[3]
]

def get_weather_forecast(lat, lon, name):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"],
        "timezone": "auto",
        "forecast_days": 14
    }


    response = requests.get(url, params=params)
    data = response.json()


    df = pd.DataFrame(data["daily"])
    df['time'] = pd.to_datetime(df['time'])

    df['province'] = name
    df['latitude'] = lat
    df['longitude'] = lon

    return df

# List to store all data
all_data = []

# Get data for each province
for province in provinces:
    try:
        print(f"Fetching data for {province['name']}...")
        df = get_weather_forecast(
            province['lat'], 
            province['lon'], 
            province['name']
        )
        all_data.append(df)
    except Exception as e:
        print(f"Error fetching data for {province['name']}: {str(e)}")

# Combine all data
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Rename columns
    final_df = final_df.rename(columns={
        'time': 'date',
        'temperature_2m_mean': 'avg_temp',
        'temperature_2m_max': 'max_temp',
        'temperature_2m_min': 'min_temp',
        'precipitation_sum': 'precipitation_mm'
    })
    
    # Reorder columns
    final_df = final_df[['province', 'latitude', 'longitude', 'date', 
                        'avg_temp', 'max_temp', 'min_temp', 'precipitation_mm']]
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spain_weather_forecast_{timestamp}.csv"
    final_df.to_csv(filename, index=False)
    
    print(f"\nData saved to {filename}")
    print(f"Total records: {len(final_df)}")
    print(f"Date range: {final_df['date'].min().date()} to {final_df['date'].max().date()}")
else:
    print("No data was retrieved.")