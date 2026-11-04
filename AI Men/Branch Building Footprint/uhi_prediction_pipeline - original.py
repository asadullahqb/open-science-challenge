#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UHI Prediction Pipeline

This script implements the complete pipeline for Urban Heat Island (UHI) index prediction,
from data loading to submission file generation.
"""

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# Data Science
import numpy as np
import pandas as pd

# Multi-dimensional arrays and datasets
import xarray as xr

# Geospatial raster data handling
import rioxarray as rxr

# Geospatial data analysis
import geopandas as gpd

# Geospatial operations
import rasterio
from rasterio import windows  
from rasterio import features  
from rasterio import warp
from rasterio.warp import transform_bounds 
from rasterio.windows import from_bounds 

# Image Processing
from PIL import Image

# Coordinate transformations
from pyproj import Proj, Transformer, CRS

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Others
import os
from tqdm import tqdm
from datetime import datetime

def load_training_data():
    """Load the training data with UHI index values."""
    print("Loading training data...")
    ground_df = pd.read_csv("Training_data_uhi_index.csv")
    return ground_df


def load_weather_data():
    """Load and process weather station data."""
    print("Loading weather station data...")
    mesonet_df = pd.read_csv("mesonet_transformed.csv")
    
    # Convert datetime column to datetime type
    mesonet_df['Datetime'] = pd.to_datetime(mesonet_df['Datetime'])
    
    # Create mask for 3-4 PM timeframe
    mask_3to4 = (mesonet_df['Datetime'].dt.hour >= 15) & (mesonet_df['Datetime'].dt.hour < 16)
    
    # List of base features to aggregate
    base_features = ['temp_C', 'humidity_perc', 'windspeed_ms', 'winddir_deg', 'flux_wm2']
    stations = ['brx', 'mhtn']
    
    # Initialize empty dictionary to store aggregated features
    agg_features = {}
    
    # Get station coordinates
    brx_lat = mesonet_df['Lat_brx'].iloc[0]
    brx_long = mesonet_df['Long_brx'].iloc[0]
    mhtn_lat = mesonet_df['lat_mhtn'].iloc[0] 
    mhtn_long = mesonet_df['long_mhtn'].iloc[0]
    
    # Add coordinates to features
    agg_features['brx_lat'] = brx_lat
    agg_features['brx_long'] = brx_long
    agg_features['mhtn_lat'] = mhtn_lat
    agg_features['mhtn_long'] = mhtn_long
    
    # Calculate aggregates for each feature and station
    for station in stations:
        for feature in base_features:
            col = f"{feature}_{station}"
            
            # Calculate 3-4 PM average
            agg_features[f"{col}_3to4"] = mesonet_df.loc[mask_3to4, col].mean()
            
            # Calculate all-day average
            agg_features[f"{col}_all"] = mesonet_df[col].mean()
    
    # Convert to DataFrame for easier viewing
    agg_df = pd.DataFrame([agg_features])
    print("Weather data aggregated successfully.")
    
    return agg_df


def calculate_distance_features(ground_df, agg_df):
    """Calculate distance-based features from weather stations."""
    print("Calculating distances to weather stations...")
    
    # Get station coordinates from agg_df
    brx_lat = agg_df['brx_lat'].iloc[0]
    brx_long = agg_df['brx_long'].iloc[0]
    mhtn_lat = agg_df['mhtn_lat'].iloc[0]
    mhtn_long = agg_df['mhtn_long'].iloc[0]
    
    # Calculate distances to Bronx weather station
    ground_df['distance_to_bronx_km'] = ground_df.apply(
        lambda row: haversine_distance(
            row['Latitude'], row['Longitude'],
            brx_lat, brx_long
        ), axis=1
    )
    
    # Calculate distances to Manhattan weather station 
    ground_df['distance_to_manhattan_km'] = ground_df.apply(
        lambda row: haversine_distance(
            row['Latitude'], row['Longitude'],
            mhtn_lat, mhtn_long
        ), axis=1
    )
    
    return ground_df


def calculate_weighted_weather_features(ground_df, agg_df):
    """Calculate inverse distance weighted weather features."""
    print("Calculating inverse distance weights...")
    total_inverse_dist = (1/ground_df['distance_to_manhattan_km']) + (1/ground_df['distance_to_bronx_km'])
    ground_df['manhattan_weight'] = (1/ground_df['distance_to_manhattan_km']) / total_inverse_dist
    ground_df['bronx_weight'] = (1/ground_df['distance_to_bronx_km']) / total_inverse_dist
    
    print("Calculating weighted weather features...")
    
    # Temperature
    ground_df['weighted_temp_C'] = (
        ground_df['manhattan_weight'] * agg_df['temp_C_mhtn_all'].iloc[0] + 
        ground_df['bronx_weight'] * agg_df['temp_C_brx_all'].iloc[0]
    )
    
    # Humidity
    ground_df['weighted_humidity'] = (
        ground_df['manhattan_weight'] * agg_df['humidity_perc_mhtn_all'].iloc[0] + 
        ground_df['bronx_weight'] * agg_df['humidity_perc_brx_all'].iloc[0]
    )
    
    # Wind speed
    ground_df['weighted_windspeed'] = (
        ground_df['manhattan_weight'] * agg_df['windspeed_ms_mhtn_all'].iloc[0] + 
        ground_df['bronx_weight'] * agg_df['windspeed_ms_brx_all'].iloc[0]
    )
    
    return ground_df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def map_satellite_data(tiff_path, csv_path):
    
    # Load the GeoTIFF data
    data = rxr.open_rasterio(tiff_path)
    tiff_crs = data.rio.crs

    # Read the Excel file using pandas
    df = pd.read_csv(csv_path)
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values

    # 3. Convert lat/long to the GeoTIFF's CRS
    # Create a Proj object for EPSG:4326 (WGS84 - lat/long) and the GeoTIFF's CRS
    proj_wgs84 = Proj(init='epsg:4326')  # EPSG:4326 is the common lat/long CRS
    proj_tiff = Proj(tiff_crs)
    
    # Create a transformer object
    transformer = Transformer.from_proj(proj_wgs84, proj_tiff)

    # Initialize lists for all bands
    B01_values = []
    B02_values = []
    B03_values = []
    B04_values = []
    B05_values = []
    B06_values = []
    B07_values = []
    B08_values = []
    B8A_values = []
    B11_values = []
    B12_values = []

    # Iterate over the latitudes and longitudes, and extract the corresponding band values
    for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Mapping values"):
        # Extract values for all bands
        B01_value = data.sel(x=lon, y=lat, band=1, method="nearest").values
        B01_values.append(B01_value)
        
        B02_value = data.sel(x=lon, y=lat, band=2, method="nearest").values
        B02_values.append(B02_value)
        
        B03_value = data.sel(x=lon, y=lat, band=3, method="nearest").values
        B03_values.append(B03_value)
        
        B04_value = data.sel(x=lon, y=lat, band=4, method="nearest").values
        B04_values.append(B04_value)
        
        B05_value = data.sel(x=lon, y=lat, band=5, method="nearest").values
        B05_values.append(B05_value)
        
        B06_value = data.sel(x=lon, y=lat, band=6, method="nearest").values
        B06_values.append(B06_value)
        
        B07_value = data.sel(x=lon, y=lat, band=7, method="nearest").values
        B07_values.append(B07_value)
        
        B08_value = data.sel(x=lon, y=lat, band=8, method="nearest").values
        B08_values.append(B08_value)
        
        B8A_value = data.sel(x=lon, y=lat, band=9, method="nearest").values
        B8A_values.append(B8A_value)
        
        B11_value = data.sel(x=lon, y=lat, band=10, method="nearest").values
        B11_values.append(B11_value)
        
        B12_value = data.sel(x=lon, y=lat, band=11, method="nearest").values
        B12_values.append(B12_value)

    # Create a DataFrame with all band values
    df = pd.DataFrame()
    df['B01'] = B01_values
    df['B02'] = B02_values
    df['B03'] = B03_values
    df['B04'] = B04_values
    df['B05'] = B05_values
    df['B06'] = B06_values
    df['B07'] = B07_values
    df['B08'] = B08_values
    df['B8A'] = B8A_values
    df['B11'] = B11_values
    df['B12'] = B12_values
    
    return df


def map_lst_data(tiff_path, csv_path):
    """
    Extract Land Surface Temperature (LST) data for each coordinate in the CSV file.
    
    Args:
        tiff_path: Path to the LST GeoTIFF file
        csv_path: Path to the CSV file with coordinates
        
    Returns:
        DataFrame with LST values
    """
    print(f"Mapping LST data from {tiff_path}...")
    
    # Load the GeoTIFF data
    data = rxr.open_rasterio(tiff_path)
    tiff_crs = data.rio.crs
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values
    
    # Set up coordinate transformation
    proj_wgs84 = Proj(init='epsg:4326')
    proj_tiff = Proj(tiff_crs)
    transformer = Transformer.from_proj(proj_wgs84, proj_tiff)
    
    # Initialize empty list for LST values
    lst_values = []
    
    # Extract LST values for each coordinate pair
    for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Mapping values"):
        lst_value = data.sel(x=lon, y=lat, method="nearest").values
        lst_values.append(lst_value[0])  # Take first band only for LST
    
    # Create DataFrame with LST values
    df_lst = pd.DataFrame({'LST': lst_values})
    
    return df_lst


def calculate_indices(final_data):
    """Calculate spectral indices from satellite bands."""
    print("Calculating spectral indices...")
    
    # Calculate median NDVI (Normalized Difference Vegetation Index)
    ndvi = (final_data['B08'] - final_data['B04']) / (final_data['B08'] + final_data['B04'])
    ndvi = ndvi.replace([np.inf, -np.inf], np.nan)
    final_data['NDVI'] = ndvi.median()
    
    # Calculate median NDBI (Normalized Difference Built-up Index)
    ndbi = (final_data['B11'] - final_data['B08']) / (final_data['B11'] + final_data['B08'])
    ndbi = ndbi.replace([np.inf, -np.inf], np.nan)
    final_data['NDBI'] = ndbi.median()
    
    # Calculate median NDWI (Normalized Difference Water Index)
    ndwi = (final_data['B03'] - final_data['B08']) / (final_data['B03'] + final_data['B08'])
    ndwi = ndwi.replace([np.inf, -np.inf], np.nan)
    final_data['NDWI'] = ndwi.median()
    
    return final_data


def combine_datasets(dataset1, dataset2):
    """
    Combine two datasets vertically (along columns) using pandas concat function.
    
    Args:
        dataset1: Dataset 1 to be combined 
        dataset2: Dataset 2 to be combined
    
    Returns:
        Combined dataset
    """
    data = pd.concat([dataset1, dataset2], axis=1)
    return data


def train_model(uhi_data, feature_names):
    """Train the model using the selected features."""
    print("Training the model...")
    
    # Split the data into features (X) and target (y), and then into training and testing sets
    X = uhi_data[feature_names].values
    y = uhi_data['UHI Index'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
    
    # Scale the training and test data using standardscaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Model R² score: {r2:.4f}")
    
    return model, sc


def predict_test_data(test_file, model, sc, feature_names, agg_df):
    """Make predictions on test data."""
    print("Preparing test data for prediction...")
    
    # Calculate inverse distance weights for test data
    total_inverse_dist = (1/test_file['distance_to_manhattan_km']) + (1/test_file['distance_to_bronx_km'])
    test_file['manhattan_weight'] = (1/test_file['distance_to_manhattan_km']) / total_inverse_dist
    test_file['bronx_weight'] = (1/test_file['distance_to_bronx_km']) / total_inverse_dist
    
    # Calculate weighted weather features for test data
    test_file['weighted_temp_C'] = (
        test_file['manhattan_weight'] * agg_df['temp_C_mhtn_all'].iloc[0] + 
        test_file['bronx_weight'] * agg_df['temp_C_brx_all'].iloc[0]
    )
    
    test_file['weighted_humidity'] = (
        test_file['manhattan_weight'] * agg_df['humidity_perc_mhtn_all'].iloc[0] + 
        test_file['bronx_weight'] * agg_df['humidity_perc_brx_all'].iloc[0]
    )
    
    test_file['weighted_windspeed'] = (
        test_file['manhattan_weight'] * agg_df['windspeed_ms_mhtn_all'].iloc[0] + 
        test_file['bronx_weight'] * agg_df['windspeed_ms_brx_all'].iloc[0]
    )
    
    # Select features for prediction
    submission_val_data = test_file[feature_names]
    
    # Feature Scaling 
    submission_val_data = submission_val_data.values
    transformed_submission_data = sc.transform(submission_val_data)
    
    # Making predictions
    print("Making predictions...")
    predictions = model.predict(transformed_submission_data)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'Longitude': test_file['Longitude'].values, 
        'Latitude': test_file['Latitude'].values, 
        'UHI Index': predictions
    })
    
    return submission_df


def main(prepare_data=True, submit=True):
    """Main function to run the entire pipeline."""
    print("Starting UHI prediction pipeline...")
    
    if prepare_data:
        # Load training data
        ground_df = load_training_data()
        
        # Calculate distance features
        ground_df = calculate_distance_features(ground_df, agg_df)
        
        # Calculate weighted weather features
        ground_df = calculate_weighted_weather_features(ground_df, agg_df)
        
        # Map satellite data
        final_data = map_satellite_data('S2_sample.tiff', 'Training_data_uhi_index.csv')
        
        # Map LST data
        lst_data = map_lst_data('Landsat_LST.tiff', 'Training_data_uhi_index.csv')
        
        # Join LST data to final_data
        final_data = pd.concat([final_data, lst_data], axis=1)
        
        # Calculate spectral indices
        final_data = calculate_indices(final_data)
        
        # Combine ground data and satellite data
        uhi_data = combine_datasets(ground_df, final_data)
        
        # Reset index
        uhi_data = uhi_data.reset_index(drop=True)
        
        # Drop any rows containing null values
        uhi_data = uhi_data.dropna()

        uhi_data.to_csv("uhi_data.csv", index=False)
    
    else:
        uhi_data = pd.read_csv("uhi_data.csv")
        
        # Load and process weather data
        agg_df = load_weather_data()

    # Define feature names for model training
    feature_names = ['nearest_building_count_500m', 'B01', 'LST']
    
    # Train model
    model, sc = train_model(uhi_data, feature_names)
    
    if submit:

        # Load test data
        print("Loading test data...")
        test_file = pd.read_csv("submission.csv")
        
        # Calculate distance features for test data
        test_file = calculate_distance_features(test_file, agg_df)
        
        # Map satellite data for test data
        test_satellite_data = map_satellite_data('S2_sample.tiff', 'submission.csv')
        
        # Map LST data for test data
        test_lst_data = map_lst_data('Landsat_LST.tiff', 'submission.csv')
        
        # Join satellite and LST data to test_file
        test_file = pd.concat([test_file, test_satellite_data, test_lst_data], axis=1)
        
        # Calculate spectral indices for test data
        test_file = calculate_indices(test_file)
        
        # Make predictions
        submission_df = predict_test_data(test_file, model, sc, feature_names, agg_df)
        
        # Save predictions to CSV
        today = datetime.today().date()
        output_file = f"submission-{today}.csv"
        submission_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        print("Pipeline completed successfully!")
    else:
        pass


if __name__ == "__main__":
    main(prepare_data=False, submit=False)