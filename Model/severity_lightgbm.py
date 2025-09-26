import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
from pathlib import Path
from tqdm.auto import tqdm

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

def prepare_features(df, weather_data, days_before=7):
    """Prepare features for training by combining fire and weather data."""
    print("\nPreparing features...")
    
    # Add temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['season'] = ((df['month'] % 12 + 3) // 3).astype(int)
    
    # Add cyclical features
    df = encode_cyclical(df, 'month', 12)
    df = encode_cyclical(df, 'day_of_year', 365)
    
    # Add spatial interaction term
    df['lat_long_interaction'] = df['latitude'] * df['longitude']
    
    # Find nearest weather station data for each fire
    print("Merging weather data with fire data...")
    weather_features = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing fire events"):
        weather_row = find_nearest_weather_station(row, weather_data, days_before)
        weather_features.append(weather_row)
    
    # Combine weather features with fire data
    weather_df = pd.DataFrame(weather_features)
    df = pd.concat([df.reset_index(drop=True), weather_df.reset_index(drop=True)], axis=1)
    
    # Add days since last fire (placeholder)
    df['days_since_last_fire'] = 365  # Default to 1 year
    
    # Define features and target
    feature_cols = [
        'latitude', 'longitude', 'year', 'month', 'day_of_year', 
        'day_of_week', 'is_weekend', 'season', 'lat_long_interaction',
        'TEMP_AVG_mean', 'TEMP_AVG_max', 'TEMP_AVG_min', 'TEMP_AVG_std',
        'PRCP_mean', 'PRCP_sum', 'PRCP_max', 'TEMP_RANGE_mean', 'TEMP_RANGE_max',
        'TEMP_ANOMALY_mean', 'PRCP_ANOMALY_mean', 'weather_distance',
        'weather_station_lat', 'weather_station_lon', 'days_since_last_fire',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos'
    ]
    
    # Select only the columns that exist in the dataframe
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols]
    y = df['severity']
    
    # Fill any remaining NaN values with column means
    X = X.fillna(X.mean())
    
    print(f"Prepared {len(X)} samples with {len(X.columns)} features")
    return X, y

def prepare_prediction_features(grid_points, forecast_date, weather_data, scaler=None):
    """Prepare features for prediction."""
    # Debug: Print available columns in weather_data
    print("\nAvailable columns in weather_data:", weather_data.columns.tolist())
    
    # Convert forecast_date to datetime if it's a string
    if isinstance(forecast_date, str):
        forecast_date = pd.to_datetime(forecast_date)
    
    # Initialize features with grid points
    features = grid_points.copy()
    
    # Add temporal features
    features['year'] = forecast_date.year
    features['month'] = forecast_date.month
    features['day_of_year'] = forecast_date.dayofyear
    features['day_of_week'] = forecast_date.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    features['season'] = ((features['month'] % 12 + 3) // 3).astype(int)
    
    # Add cyclical features
    features = encode_cyclical(features, 'month', 12)
    features = encode_cyclical(features, 'day_of_year', 365)
    
    # Add interaction terms
    features['lat_long_interaction'] = features['latitude'] * features['longitude']
    
    # Filter weather data for the forecast date
    weather_forecast = weather_data[weather_data['date'] == forecast_date].copy()
    
    if weather_forecast.empty:
        raise ValueError(f"No weather forecast data available for {forecast_date}")
    
    # For each grid point, find the nearest weather station
    weather_features = []
    
    for i, row in tqdm(features.iterrows(), total=len(features), desc="Merging weather data"):
        # Calculate distances to all weather stations
        weather_forecast['distance'] = np.sqrt(
            (weather_forecast['latitude'] - row['latitude'])**2 +
            (weather_forecast['longitude'] - row['longitude'])**2
        )
        
        # Get the closest weather station
        closest_station = weather_forecast.loc[weather_forecast['distance'].idxmin()].copy()
        
        # Debug: Print available columns in closest_station
        if i == 0:  # Only print for the first iteration to avoid too much output
            print("\nColumns in closest_station:", closest_station.index.tolist())
            print("Sample closest_station data:", closest_station.to_dict())
        
        # Add weather features
        weather_features.append({
            'TEMP_AVG': closest_station.get('avg_temp', np.nan),
            'TMAX': closest_station.get('max_temp', np.nan),
            'TMIN': closest_station.get('min_temp', np.nan),
            'PRCP': closest_station.get('precipitation_mm', np.nan),
            'TEMP_RANGE': closest_station.get('max_temp', np.nan) - closest_station.get('min_temp', np.nan),
            'weather_distance': closest_station.get('distance', np.nan),
            'weather_station_lat': closest_station.get('latitude', np.nan),
            'weather_station_lon': closest_station.get('longitude', np.nan)
        })
    
    # Add weather features to the dataframe
    weather_df = pd.DataFrame(weather_features)
    features = pd.concat([features, weather_df], axis=1)
    
    # Add days since last fire (placeholder)
    features['days_since_last_fire'] = 365  # Default to 1 year
    
    # Scale features if scaler is provided
    if scaler is not None:
        numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols if col not in ['is_weekend', 'season']]
        
        # Store the column order before scaling
        col_order = features.columns
        
        # Scale the features
        features[numerical_cols] = scaler.transform(features[numerical_cols])
        
        # making sure the order of columns remains the same
        features = features[col_order]
    
    return features

def train_lightgbm(X, y):
    """Train and evaluate a LightGBM model with cross-validation."""
    print("\nTraining LightGBM model...")
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded
    )
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[class_label] for class_label in y_train])
    
    # Create dataset for LightGBM with balanced weights
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    valid_data = lgb.Dataset(X_test, label=y_test, 
                           weight=compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)[y_test])
    
    # Parameters with regularization
    params = {
        # Core parameters
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 30,
        'feature_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_iterations': 1000,
        'max_depth': -1,
        'feature_fraction_bynode': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_split_gain': 0.1,
        'is_unbalance': False,
        'verbosity': -1,
        'seed': 42
    }
    
    # Train model with early stopping
    print("\nTraining final model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=True),
            lgb.log_evaluation(50),
            lgb.callback.reset_parameter(learning_rate=lambda x: 0.05 * (0.99 ** x))
        ]
    )
    
    # Make predictions
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Evaluate model
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Feature importances (gain):")
    print(importance.head(20).to_string(index=False))
    
    with open('feature_names.txt', 'w') as f:
        f.write(','.join(X.columns))

    model.save_model('trained_model.txt')
    return model

def main():
    """Main function to run the pipeline."""
    # Load and prepare data
    df = load_fire_data()

    # For testing: frac=0.05 -> Take only 5% of the data
    df = df.sample(frac=1.00, random_state=42)
    print(f"\nUsing sample of {len(df)} records for testing...")

    # Load weather data
    try:
        weather_data = load_historical_weather_data()
        print("Weather data loaded successfully!")
    except Exception as e:
        print(f"Could not load weather data: {e}")
        weather_data = None

    # Prepare features
    X, y = prepare_features(df, weather_data)
    
    # Train and evaluate model
    model = train_lightgbm(X, y)

if __name__ == "__main__":
    main()