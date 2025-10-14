import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from tqdm.auto import tqdm

# LLaMA inference helper
from flareon_ai.llama_inference import infer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_fire_data():
    """Load and prepare the fire dataset."""
    print("Loading fire data...")

    # Load the dataset
    df = pd.read_csv('historical_wildfires.csv')


    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Create severity categories (3 classes)
    df['severity'] = pd.cut(
        df['burned_area_ha'],
        bins=[0, 3, 23, float('inf')],
        labels=['low', 'medium', 'high']
    )

    print("Data loaded successfully!")
    print(f"Total samples: {len(df)}")
    print("\nSeverity distribution:")
    print(df['severity'].value_counts())

    return df

def load_historical_weather_data():
    """Load and preprocess historical weather data."""
    print("Loading historical weather data...")
    try:
        # Load and clean data
        weather = pd.read_csv('historical_weather.csv', parse_dates=['DATE'],
                            dtype={'STATION': str, 'NAME': str, 'LATITUDE': float,
                                  'LONGITUDE': float, 'ELEVATION': float, 'PRCP': float,
                                  'TAVG': float, 'TMAX': float, 'TMIN': float})

        # Basic cleaning and renaming
        weather = weather.rename(columns={'DATE': 'date'})
        weather = weather[['date', 'LATITUDE', 'LONGITUDE', 'PRCP', 'TMAX', 'TMIN']].copy()
        weather = weather.drop_duplicates(subset=['date', 'LATITUDE', 'LONGITUDE'])

        # Calculate derived metrics
        weather['TEMP_RANGE'] = weather['TMAX'] - weather['TMIN']
        weather['TEMP_AVG'] = (weather['TMAX'] + weather['TMIN']) / 2

        # Sort and ensure proper datetime index for rolling calculations
        weather = weather.sort_values(['LATITUDE', 'LONGITUDE', 'date'])

        # Function to calculate rolling statistics for each station
        def calculate_rolling_stats(group):
            group = group.set_index('date').sort_index()
            for days in [7, 14, 30]:  # Calculate for 7, 14, and 30-day windows
                group[f'PRCP_{days}D_AVG'] = group['PRCP'].rolling(f'{days}D', min_periods=1).mean()
                group[f'TEMP_AVG_{days}D'] = group['TEMP_AVG'].rolling(f'{days}D', min_periods=1).mean()
                group[f'TEMP_RANGE_{days}D_AVG'] = group['TEMP_RANGE'].rolling(f'{days}D', min_periods=1).mean()
            return group.reset_index()

        # Apply rolling calculations per station
        weather = weather.groupby(['LATITUDE', 'LONGITUDE'], group_keys=False).apply(calculate_rolling_stats)

        # Calculate anomalies (difference from station's historical average)
        weather['date'] = pd.to_datetime(weather['date'])
        weather['day_of_year'] = weather['date'].dt.dayofyear

        # Calculate climatology (average for each day of year)
        climatology = weather.groupby(['LATITUDE', 'LONGITUDE', 'day_of_year']).agg({
            'TEMP_AVG': 'mean',
            'PRCP': 'mean'
        }).rename(columns={
            'TEMP_AVG': 'TEMP_CLIMATOLOGY',
            'PRCP': 'PRCP_CLIMATOLOGY'
        }).reset_index()

        # Merge climatology back to calculate anomalies
        weather = pd.merge(weather, climatology,
                          on=['LATITUDE', 'LONGITUDE', 'day_of_year'],
                          how='left')

        weather['TEMP_ANOMALY'] = weather['TEMP_AVG'] - weather['TEMP_CLIMATOLOGY']
        weather['PRCP_ANOMALY'] = weather['PRCP'] - weather['PRCP_CLIMATOLOGY']

        # Drop intermediate columns
        weather = weather.drop(['day_of_year', 'TEMP_CLIMATOLOGY', 'PRCP_CLIMATOLOGY'], axis=1)

        print(f"Weather data loaded with {len(weather)} records from {weather['date'].min()} to {weather['date'].max()}")
        print(f"Number of unique weather stations: {weather[['LATITUDE', 'LONGITUDE']].drop_duplicates().shape[0]}")

        return weather

    except Exception as e:
        print(f"Error loading weather data: {str(e)}")
        print("Make sure 'historical_weather.csv' is in the correct format and location.")
        raise

def find_nearest_weather_station(fire_row, weather_data, days_before=7):
    """Find the nearest weather station and get weather data for the fire date and previous days."""
    fire_date = pd.to_datetime(fire_row['date'])
    start_date = fire_date - pd.Timedelta(days=days_before)

    # Filter weather data for the date range
    mask = (weather_data['date'] >= start_date) & (weather_data['date'] <= fire_date)
    weather_subset = weather_data[mask].copy()

    if weather_subset.empty:
        return pd.Series()

    # Calculate distances to all weather stations
    weather_subset['distance'] = np.sqrt(
        (weather_subset['LATITUDE'] - fire_row['latitude'])**2 +
        (weather_subset['LONGITUDE'] - fire_row['longitude'])**2
    )

    if weather_subset.empty:
        return pd.Series()

    # Get the closest station
    closest_station = weather_subset.loc[weather_subset['distance'].idxmin()]
    station_mask = weather_subset['distance'] == closest_station['distance']

    # Aggregate weather data for the period
    agg_dict = {
        'TEMP_AVG': ['mean', 'max', 'min', 'std'],
        'PRCP': ['mean', 'sum', 'max'],
        'TEMP_RANGE': ['mean', 'max'],
        'TEMP_ANOMALY': 'mean',
        'PRCP_ANOMALY': 'mean'
    }

    # Filter for just this station's data
    station_data = weather_subset[station_mask]

    # If no data after filtering, return empty Series
    if station_data.empty:
        return pd.Series()

    # Perform aggregation
    weather_agg = station_data.agg(agg_dict)

    # Flatten the multi-index columns
    weather_agg_flat = {}
    for col in weather_agg.columns:
        if isinstance(weather_agg[col], pd.Series):
            for stat, val in weather_agg[col].items():
                weather_agg_flat[f"{col}_{stat}"] = val

    # Add distance and station info
    weather_agg_flat['weather_distance'] = closest_station['distance']
    weather_agg_flat['weather_station_lat'] = closest_station['LATITUDE']
    weather_agg_flat['weather_station_lon'] = closest_station['LONGITUDE']

    return pd.Series(weather_agg_flat)

def encode_cyclical(df, col, max_val):
    """Encode cyclical features using sine and cosine transformations."""
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

def main():
    """Main function to generate a wildfire severity analysis with LLaMA."""
    # Load and prepare data
    df = load_fire_data()
    print(f"Loaded {len(df)} fire records.")

    # Load weather data if needed for context
    try:
        weather_data = load_historical_weather_data()
        print("Weather data loaded successfully!")
    except Exception as e:
        print(f"Could not load weather data: {e}")
        weather_data = None

    # Build a detailed prompt summarizing data statistics for LLaMA
    prompt = (
        f"Analyze wildfire severity risk using the following data summary:\n"
        f"Total fire samples: {len(df)}\n"
        f"Severity distribution:\n{df['severity'].value_counts().to_string()}\n"
    )
    if weather_data is not None:
        prompt += (
            f"Weather data records: {len(weather_data)} from {weather_data['date'].min()} to {weather_data['date'].max()}\n"
            f"Provide a detailed analysis and predictions of wildfire severity trends "
            f"using this historical climate information."
        )

    # Call LLaMA for analysis
    try:
        llama_output = infer(prompt)
        print("\n=== LLaMA Wildfire Severity Analysis ===\n")
        print(llama_output)
    except Exception as e:
        print(f"LLaMA inference failed: {e}")

if __name__ == "__main__":
    main()