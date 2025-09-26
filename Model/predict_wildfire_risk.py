import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from tqdm import tqdm
import joblib
import argparse
import geopandas as gpd
from shapely.geometry import Point, shape
from API.API_setup import provinces
from tabulate import tabulate
# change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_forecast_weather(forecast_date=None):
    #Load and preprocess weather forecast data
    try:
        # load forecast data
        forecast = pd.read_csv('forecast_weather.csv')
        print("\nFirst few rows of forecast data:")
        print(forecast.head())
        
        # convert date column to datetime
        forecast['date'] = pd.to_datetime(forecast['date'])
        
         # sort by date to ensure correct order
        forecast = forecast.sort_values('date')
        
        return forecast
        
    except Exception as e:
        print(f"Error loading forecast weather data: {e}")
        return pd.DataFrame()

def get_nearest_province(lat, lon, provinces):
    # find the nearest province for given coordinates.
    min_distance = float('inf')
    nearest_province = "Unknown"
    
    for province in provinces:
        # calculate distance using simple Euclidean distance
        distance = ((province['lat'] - lat) ** 2 + 
                   (province['lon'] - lon) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            nearest_province = province['name']
    
    return nearest_province

def create_risk_grid(resolution=100):
    # create a grid of Spain for risk prediction, only including points within Spain's borders
    try:
        # update this path to where you extracted the Natural Earth data
        natural_earth_path = r"C:\Users\danie\Desktop\PRJ4-2025-prj4-2025-d07\Model\natural_earth\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
        # load the Natural Earth data
        world = gpd.read_file(natural_earth_path)
        spain = world[world['ADMIN'] == 'Spain'].geometry.iloc[0]
        
        # get the bounds of Spain
        min_lon, min_lat, max_lon, max_lat = spain.bounds
        
        # create a grid of points
        lats = np.linspace(min_lat, max_lat, resolution)
        lons = np.linspace(min_lon, max_lon, resolution)
        lons_grid, lats_grid = np.meshgrid(lons, lats)
        
        # create points and check if they're within Spain
        points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lons_grid.ravel(), lats_grid.ravel())])
        mask = points.within(spain)
        
        # create DataFrame with only points within Spain
        grid = pd.DataFrame({
            'longitude': lons_grid.ravel(),
            'latitude': lats_grid.ravel()
        })[mask]
        
        print(f"Created grid with {len(grid)} points within Spain's borders")
        return grid
        
    except Exception as e:
        print(f"Error creating detailed risk grid: {e}")
        print("Falling back to simple grid...")
        return create_simple_grid(resolution)

def prepare_prediction_features(grid_points, forecast_date, weather_data, scaler=None):
    # prepare features for prediction
    # making a copy of the original grid points to maintain indices
    features = grid_points.copy()
    
    # Add temporal features
    features['year'] = forecast_date.year
    features['month'] = forecast_date.month
    features['day_of_year'] = forecast_date.dayofyear
    features['day_of_week'] = forecast_date.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    features['season'] = ((features['month'] % 12 + 3) // 3).astype(int)
    
    # add cyclical features
    features = encode_cyclical(features, 'month', 12)
    features = encode_cyclical(features, 'day_of_year', 365)
    
    # add interaction terms
    features['lat_long_interaction'] = features['latitude'] * features['longitude']
    
    # filter weather data for the forecast date
    weather_forecast = weather_data[weather_data['date'] == forecast_date].copy()
    
    if weather_forecast.empty:
        raise ValueError(f"No weather forecast data available for {forecast_date}")
    
    # for each grid point, find the nearest weather station
    weather_features = []
    
    for i, row in tqdm(features.iterrows(), total=len(features), desc="Merging weather data"):
        # calculate distances to all weather stations
        weather_forecast['distance'] = np.sqrt(
            (weather_forecast['latitude'] - row['latitude'])**2 +
            (weather_forecast['longitude'] - row['longitude'])**2
        )
        
        # get the closest weather station
        closest_station = weather_forecast.loc[weather_forecast['distance'].idxmin()].copy()
        
        # add weather features - matching exactly what's in training
        temp_avg = closest_station.get('avg_temp', np.nan)
        temp_max = closest_station.get('max_temp', np.nan)
        temp_min = closest_station.get('min_temp', np.nan)
        prcp = closest_station.get('precipitation_mm', np.nan)
        temp_range = temp_max - temp_min if not np.isnan(temp_max) and not np.isnan(temp_min) else np.nan
        
        weather_features.append({
            'TEMP_AVG_mean': temp_avg,
            'TEMP_AVG_max': temp_max,
            'TEMP_AVG_min': temp_min,
            'TEMP_AVG_std': 0,
            'PRCP_mean': prcp,
            'PRCP_sum': prcp,
            'PRCP_max': prcp,
            'TEMP_RANGE_mean': temp_range,
            'TEMP_RANGE_max': temp_range, 
            'TEMP_ANOMALY_mean': 0,
            'PRCP_ANOMALY_mean': 0,
            'weather_distance': closest_station.get('distance', np.nan),
            'weather_station_lat': closest_station.get('latitude', np.nan),
            'weather_station_lon': closest_station.get('longitude', np.nan)
        })
    
    # add weather features to the dataframe
    weather_df = pd.DataFrame(weather_features, index=features.index)
    features = pd.concat([features, weather_df], axis=1)
    
    # add days since last fire (placeholder)
    features['days_since_last_fire'] = 365
    
    # define the exact feature order expected by the model
    expected_features = [
        'latitude', 'longitude', 'year', 'month', 'day_of_year', 
        'day_of_week', 'is_weekend', 'season', 'lat_long_interaction',
        'TEMP_AVG_mean', 'TEMP_AVG_max', 'TEMP_AVG_min', 'TEMP_AVG_std',
        'PRCP_mean', 'PRCP_sum', 'PRCP_max', 'TEMP_RANGE_mean', 'TEMP_RANGE_max',
        'TEMP_ANOMALY_mean', 'PRCP_ANOMALY_mean', 'weather_distance',
        'weather_station_lat', 'weather_station_lon', 'days_since_last_fire',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos'
    ]
    
    # making sure all expected features exist (add with 0 if missing)
    for feature in expected_features:
        if feature not in features.columns:
            features[feature] = 0.0
    
    # selecting only the expected features in the correct order
    features = features[expected_features]
    
    # filling any remaining NaN values with 0
    features = features.fillna(0)
    
    # making sure we return the same number of points we started with
    if len(features) != len(grid_points):
        raise ValueError(f"Number of features ({len(features)}) does not match number of grid points ({len(grid_points)})")
    
    return features

def encode_cyclical(df, col, max_val):
    # encode cyclical features using sine and cosine transformations
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

def load_model_and_scaler(model_path='trained_model.txt', scaler_path='feature_scaler.pkl'):
    # load the trained model and feature scaler
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = lgb.Booster(model_file=model_path)
    
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    return model, scaler

def generate_risk_forecast(forecast_date=None, model_path='trained_model.txt', 
                          output_dir='risk_predictions', resolution=100, forecast_days=14):
    # validate forecast_days
    forecast_days = max(1, min(14, int(forecast_days)))
    
    # set default to tomorrow if no date provided
    if forecast_date is None:
        forecast_date = (datetime.now() + timedelta(days=1)).strftime('%d-%m-%Y')
    
    try:
        forecast_date = pd.to_datetime(forecast_date, dayfirst=True)        
        print(f"\nGenerating wildfire risk forecast for {forecast_date.strftime('%d-%m-%Y')}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # load model and scaler
        print("Loading model and scaler...")
        model, scaler = load_model_and_scaler(model_path)
        
        if model is None:
            print("Failed to load model. Exiting.")
            return None

        # load weather forecast data
        print("Loading weather forecast data...")
        weather_data = load_forecast_weather(forecast_date)
        
        if weather_data is None or weather_data.empty:
            print("No weather data available for the specified date.")
            return None
        
        # unique dates for the forecast period
        weather_dates = weather_data['date'].unique()
        forecast_dates = sorted(weather_dates)[:forecast_days]
        print(f"Forecasting for {len(forecast_dates)} days: {forecast_dates[0].strftime('%d-%m-%Y')} to {forecast_dates[-1].strftime('%d-%m-%Y')}")

        # prediction grid
        print(f"Creating prediction grid (resolution: {resolution}x{resolution})...")
        grid_points = create_risk_grid(resolution)
        
        if grid_points is None or len(grid_points) == 0:
            print("Failed to create prediction grid. Exiting.")
            return None

        all_results = []
        
        # process each forecast day
        for day_date in forecast_dates:
            print(f"\nProcessing {day_date.strftime('%d-%m-%Y')}...")
            
            # features for prediction
            print("Preparing features...")
            features = prepare_prediction_features(grid_points, day_date, weather_data, scaler)
            
            if features is None or len(features) == 0:
                print(f"Failed to prepare features for {day_date}. Skipping...")
                continue

            # predictions
            print("Generating predictions...")
            risk_probs = model.predict(features)
            
            # results DataFrame
            results = grid_points.copy()
            results['date'] = day_date
            results['risk_low'] = risk_probs[:, 0]
            results['risk_medium'] = risk_probs[:, 1]
            results['risk_high'] = risk_probs[:, 2]
            results['max_risk'] = np.argmax(risk_probs, axis=1)
            
            # province information
            results['province'] = results.apply(
                lambda row: get_nearest_province(row['latitude'], row['longitude'], provinces), 
                axis=1
            )
            
            all_results.append(results)
        
        if not all_results:
            print("No valid predictions generated. Exiting.")
            return None
            
        all_results = pd.concat(all_results)
        
        # save predictions
        output_csv = os.path.join(output_dir, f'risk_predictions_{forecast_date.strftime("%d-%m-%Y")}.csv')
        all_results.to_csv(output_csv, index=False)
        print(f"\nPredictions saved to {output_csv}")

        # summary
        print("\n" + "="*80)
        print("WILDFIRE RISK FORECAST SUMMARY".center(80))
        print("="*80)
        print(f"Forecast Period: {forecast_dates[0].strftime('%d-%m-%Y')} to {forecast_dates[-1].strftime('%d-%m-%Y')} ({forecast_days} days)")
        print("="*80)
        
        # risk level legend
        print("\nRISK LEVEL LEGEND:")
        print("Perfect conditions for a wildfire appear: 1.00")
        print("- 🔴 High: High probability of wildfires (0.5-1.0)")
        print("- 🟡 Medium: Medium probability of wildfires (0.3-0.5)")
        print("- 🟢 Low: Low probability of wildfires (0.0-0.3)")

        print("\n" + "="*80)
        print("TODAY'S RISK (All Provinces)".center(80))
        print("="*80)

        # Get today's data
        today = forecast_dates[0]
        today_data = all_results[all_results['date'] == today].copy()

        # Prepare today's risk table
        today_risks = []
        for province in sorted(all_results['province'].unique()):
            risk_data = today_data[today_data['province'] == province]
            if not risk_data.empty:
                risk = risk_data['risk_high'].max()
                risk_level = '🔴 High' if risk >= 0.5 else '🟡 Medium' if risk >= 0.3 else '🟢 Low'
                today_risks.append([province, risk_level, f"{risk:.2f}"])

        # Print today's risk table
        print(tabulate(today_risks, 
                      headers=["Province", "Risk Level", "Risk Score"], 
                      tablefmt='grid',
                      stralign='center'))

        # 2. TOP 10 HIGHEST RISK AREAS (14-day trend)
        print("\n" + "="*90)
        print("TOP 10 HIGHEST RISK AREAS (Next 14 days)".center(90))
        print("="*90)

        # Prepare top 10 data
        top_risks = all_results.groupby('province')['risk_high'].agg(['max', 'mean']).nlargest(10, 'max').reset_index()
        top_risks.columns = ['Province', 'Max Risk', 'Avg Risk']
        top_risks['Rank'] = range(1, len(top_risks) + 1)

        # Prepare table data with daily risks
        table_data = []
        for _, row in top_risks.iterrows():
            province = row['Province']
            daily_risks = []
            for date in forecast_dates:
                risk = all_results[(all_results['province'] == province) & 
                                 (all_results['date'] == date)]['risk_high'].max()
                daily_risks.append(f"{risk:.2f}")
            
            # Add max risk and average to the end
            table_row = [row['Rank'], province] + daily_risks + [f"{row['Max Risk']:.2f}", f"{row['Avg Risk']:.2f}"]
            table_data.append(table_row)

        # Print first 8 days
        headers = ["Rank", "Province"] + [f"D+{i+1}" for i in range(8)] + ["Avg"]
        first_table = [row[:10] for row in table_data]  # First 8 days + rank + province
        print(tabulate(first_table, headers=headers, tablefmt='grid', stralign='center'))

        # Print remaining days + average
        if len(forecast_dates) > 8:
            print("\n" + " " * 15 + "CONTINUED...".center(60) + "\n")
            remaining_days = len(forecast_dates) - 8
            headers = ["Rank", "Province"] + [f"D+{i+9}" for i in range(remaining_days)] + ["Max", "Avg"]
            second_table = [[row[0], row[1]] + row[10:10+remaining_days] + [row[-2], row[-1]] for row in table_data]
            print(tabulate(second_table, headers=headers, tablefmt='grid', stralign='center'))

        
        
        return all_results

    except Exception as e:
        print(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate wildfire risk predictions')
    parser.add_argument('--date', type=str, default=None,
                       help='Forecast date (DD-MM-YYYY), defaults to tomorrow')
    parser.add_argument('--model', type=str, default='trained_model.txt',
                       help='Path to trained model file')
    parser.add_argument('--output', type=str, default='risk_predictions',
                       help='Output directory for prediction results')
    parser.add_argument('--resolution', type=int, default=100,
                       help='Grid resolution (NxN points)')
    parser.add_argument('--days', type=int, default=14,
                       help='Number of days to forecast (1-14)')
    
    args = parser.parse_args()
    
    try:
        results = generate_risk_forecast(
            forecast_date=args.date,
            model_path=args.model,
            output_dir=args.output,
            resolution=args.resolution,
            forecast_days=args.days
        )
        print("\nRisk forecast generation completed successfully!")
    except Exception as e:
        print(f"\nError generating risk forecast: {str(e)}")
        raise