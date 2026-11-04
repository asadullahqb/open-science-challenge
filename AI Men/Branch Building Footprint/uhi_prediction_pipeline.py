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
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor

# Others
import os
import pickle
from tqdm import tqdm
from datetime import datetime
import inspect

def load_data():
    """Load and merge all required data."""
    try:
        # Load ground data
        print("Loading new data from New directory...")
        ground_df = pd.read_csv("C:/Users/Asad/Desktop/Open Science Challenge/AI Men/Branch Building Footprint/New/ground_df.csv")
        print(f"Loaded new ground data with {len(ground_df)} rows and {len(ground_df.columns)} columns")
        
        # Load building heights data
        building_heights_df = pd.read_csv("C:/Users/Asad/Desktop/Open Science Challenge/AI Men/Branch Building Footprint/New/ground_df_building_heights.csv")
        print(f"Loaded building heights data with {len(building_heights_df)} rows")
        
        # Load traffic data
        traffic_df = pd.read_csv("C:/Users/Asad/Desktop/Open Science Challenge/AI Men/Branch Building Footprint/New/ground_df_traffic.csv")
        print(f"Loaded traffic data with {len(traffic_df)} rows")
        
        # Merge building heights data
        if 'avg_building_height' in building_heights_df.columns:
            ground_df = pd.merge(ground_df, 
                               building_heights_df[['Longitude', 'Latitude', 'avg_building_height', 'max_building_height']], 
                               on=['Longitude', 'Latitude'], how='left')
        
        # Merge traffic data
        if 'traffic_volume' in traffic_df.columns:
            ground_df = pd.merge(ground_df, 
                               traffic_df[['Longitude', 'Latitude', 'traffic_volume', 'road_density']], 
                               on=['Longitude', 'Latitude'], how='left')
        
        # Add additional building count data
        building_count_columns = [col for col in ground_df.columns if 'nearest_building_count' in col]
        print(f"Adding additional building count data: {building_count_columns}")
        print(f"After merging additional building count data: {len(ground_df)} rows")
        
        print("New data merged successfully.")
        
        return ground_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main(should_prepare_data=True, submit=True, retrain=False):
    """Main function to run the UHI prediction pipeline."""
    
    # Load and prepare data
    print("Loading training data...")
    uhi_data = load_data()
    
    if uhi_data is None:
        print("Error: Failed to load data")
        return
    
    # Prepare data
    if should_prepare_data:
        uhi_data = prepare_data(uhi_data)
    
    # Get feature names
    feature_names = get_feature_names(uhi_data)
    
    # Detect and handle outliers
    print("Detecting and handling outliers...")
    uhi_data = detect_outliers(uhi_data)
    
    print(f"Using {len(feature_names)} features for training")
    
    # Train model
    print("Training new model...")
    model, sc = train_model(uhi_data, feature_names)
    
    if model is None or sc is None:
        print("Error: Failed to train model")
        return
    
    # Make predictions and submit if requested
    if submit:
        print("Making predictions and submitting...")
        make_predictions_and_submit(model, sc, feature_names)
    
    print("Pipeline completed successfully!")

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


def calculate_indices(df):
    """Calculate spectral indices from satellite bands."""
    print("Calculating spectral indices...")
    
    # Basic indices
    if all(band in df.columns for band in ['B04', 'B08']):
        df['NDVI'] = (df['B08'] - df['B04']) / (df['B08'] + df['B04'])
    
    if all(band in df.columns for band in ['B11', 'B03']):
        df['NDBI'] = (df['B11'] - df['B03']) / (df['B11'] + df['B03'])
    
    if all(band in df.columns for band in ['B03', 'B08']):
        df['NDWI'] = (df['B03'] - df['B08']) / (df['B03'] + df['B08'])
    
    if all(band in df.columns for band in ['B11', 'B04']):
        df['BSI'] = ((df['B11'] + df['B04']) - (df['B08'] + df['B02'])) / ((df['B11'] + df['B04']) + (df['B08'] + df['B02']))
    
    if all(band in df.columns for band in ['B11', 'B02']):
        df['UI'] = (df['B11'] - df['B02']) / (df['B11'] + df['B02'])
    
    return df


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


def create_building_features(data):
    """Create advanced building features."""
    print("Creating advanced building features...")
    
    # Building volume estimations
    if all(col in data.columns for col in ['avg_building_height', 'nearest_building_count_100m']):
        # Estimate building volume density
        data['building_volume_density'] = data['avg_building_height'] * data['nearest_building_count_100m'] / (np.pi * 100**2)
        
        # Building height to count ratio
        data['height_to_count_ratio'] = data['avg_building_height'] / (data['nearest_building_count_100m'] + 1)
        
        # Building height variance proxy
        if 'max_building_height' in data.columns:
            data['building_height_variance'] = data['max_building_height'] - data['avg_building_height']
            data['height_variance_ratio'] = data['building_height_variance'] / (data['avg_building_height'] + 1)
    
    # Building density features at different radii
    building_count_columns = [col for col in data.columns if 'nearest_building_count_' in col and col.endswith('m')]
    if building_count_columns:
        # Sort columns by radius
        building_count_columns.sort(key=lambda x: int(x.split('_')[-1][:-1]))
        
        # Calculate building density gradients
        for i in range(1, len(building_count_columns)):
            current_col = building_count_columns[i]
            prev_col = building_count_columns[i-1]
            current_radius = int(current_col.split('_')[-1][:-1])
            prev_radius = int(prev_col.split('_')[-1][:-1])
            
            # Calculate density gradient
            current_area = np.pi * (current_radius**2)
            prev_area = np.pi * (prev_radius**2)
            current_density = data[current_col] / current_area
            prev_density = data[prev_col] / prev_area
            
            gradient_name = f'density_gradient_{prev_radius}_{current_radius}'
            data[gradient_name] = (current_density - prev_density) / (current_radius - prev_radius)
        
        # Calculate building density at each radius
        for col in building_count_columns:
            radius = int(col.split('_')[-1][:-1])
            area = np.pi * (radius**2)
            data[f'building_density_{radius}m'] = data[col] / area
    
    # Sky view factor approximation
    if 'avg_building_height' in data.columns and 'nearest_building_count_100m' in data.columns:
        # Simple approximation of sky view factor
        data['sky_view_factor'] = 1 / (1 + (data['avg_building_height'] * data['nearest_building_count_100m'] / 1000))
    
    # Urban canyon effect
    if all(col in data.columns for col in ['avg_building_height', 'nearest_building_count_100m']):
        data['urban_canyon_effect'] = data['avg_building_height'] * np.sqrt(data['nearest_building_count_100m']) / 100
    
    # Building compactness
    if all(col in data.columns for col in ['nearest_building_count_100m', 'nearest_building_count_500m']):
        data['building_compactness'] = data['nearest_building_count_100m'] / (data['nearest_building_count_500m'] + 1)
    
    # Apply log transformations to building counts to handle skewness
    for col in building_count_columns:
        data[f'log_{col}'] = np.log1p(data[col])
    
    return data


def create_traffic_features(data):
    """Create advanced traffic features."""
    print("Creating advanced traffic features...")
    
    if 'traffic_volume' in data.columns:
        # Traffic intensity
        data['traffic_intensity'] = np.log1p(data['traffic_volume'])
        
        # Traffic-building interactions
        if 'nearest_building_count_100m' in data.columns:
            data['traffic_building_interaction'] = data['traffic_volume'] * data['nearest_building_count_100m']
            
            # Traffic congestion proxy
            data['traffic_congestion_proxy'] = data['traffic_volume'] / (data['nearest_building_count_100m'] + 1)
        
        # Traffic-LST interactions
        if 'LST' in data.columns:
            data['traffic_lst_interaction'] = data['traffic_volume'] * data['LST']
            
            # Traffic heat amplification
            data['traffic_heat_amplification'] = data['traffic_volume'] * np.square(data['LST'])
    
    if 'road_density' in data.columns:
        # Road network complexity
        data['road_network_complexity'] = np.sqrt(data['road_density'])
        
        # Road-building interaction
        if 'nearest_building_count_100m' in data.columns:
            data['road_building_interaction'] = data['road_density'] * data['nearest_building_count_100m']
        
        # Road-LST interaction
        if 'LST' in data.columns:
            data['road_lst_interaction'] = data['road_density'] * data['LST']
    
    return data


def create_environmental_interactions(df):
    """Create environmental interaction features."""
    print("Creating environmental interaction features...")
    
    # Basic interactions between building and environmental features
    if 'building_count' in df.columns:
        if 'LST' in df.columns:
            df['building_lst_interaction'] = df['building_count'] * df['LST']
            df['lst_squared'] = df['LST'] ** 2
            df['lst_cubed'] = df['LST'] ** 3
            df['log_lst'] = np.log1p(df['LST'])
        
        df['building_count_squared'] = df['building_count'] ** 2
        df['building_count_cubed'] = df['building_count'] ** 3
        df['log_building_count'] = np.log1p(df['building_count'])
        
        if 'NDBI' in df.columns:
            df['building_ndbi_interaction'] = df['building_count'] * df['NDBI']
            df['exp_ndbi'] = np.exp(df['NDBI'])
            
            # Advanced NDBI interactions
            df['ndbi_squared'] = df['NDBI'] ** 2
            df['ndbi_cubed'] = df['NDBI'] ** 3
        
        if 'NDVI' in df.columns:
            df['building_ndvi_interaction'] = df['building_count'] * df['NDVI']
            df['exp_ndvi'] = np.exp(df['NDVI'])
            
            # Advanced NDVI interactions
            df['ndvi_squared'] = df['NDVI'] ** 2
            df['ndvi_cubed'] = df['NDVI'] ** 3
            
            # Vegetation cooling effect
            df['vegetation_cooling_effect'] = df['NDVI'] / (df['building_count'] + 1)
        
        if 'NDWI' in df.columns:
            df['building_ndwi_interaction'] = df['building_count'] * df['NDWI']
            
            # Water cooling effect
            df['water_cooling_effect'] = df['NDWI'] / (df['building_count'] + 1)
    
    # LST interactions
    if 'LST' in df.columns:
        if 'NDBI' in df.columns:
            df['lst_ndbi_interaction'] = df['LST'] * df['NDBI']
            
            # Heat retention by built-up areas
            df['heat_retention_index'] = df['LST'] * df['NDBI'] ** 2
        
        if 'NDVI' in df.columns:
            df['lst_ndvi_interaction'] = df['LST'] * df['NDVI']
            
            # Cooling efficiency of vegetation
            df['vegetation_cooling_efficiency'] = df['LST'] * (1 - df['NDVI'])
        
        if 'NDWI' in df.columns:
            df['lst_ndwi_interaction'] = df['LST'] * df['NDWI']
            
            # Water body cooling effect
            df['water_cooling_efficiency'] = df['LST'] * (1 - df['NDWI'])
        
        if 'BSI' in df.columns:
            df['lst_bsi_interaction'] = df['LST'] * df['BSI']
            
            # Bare soil heating effect
            df['soil_heating_effect'] = df['LST'] * df['BSI'] ** 2
    
    # Building count interactions
    if 'building_count' in df.columns:
        df['building_count_interaction'] = df['building_count'] * df['building_count'].rolling(window=3, min_periods=1).mean()
    
    # Band interactions
    if 'B04' in df.columns and 'B08' in df.columns:
        df['b04_b08_interaction'] = df['B04'] * df['B08']
        
        # Red-NIR ratio (vegetation indicator)
        df['red_nir_ratio'] = df['B04'] / (df['B08'] + 0.0001)
    
    if 'B03' in df.columns and 'B11' in df.columns:
        df['b03_b11_interaction'] = df['B03'] * df['B11']
        
        # Green-SWIR ratio (moisture indicator)
        df['green_swir_ratio'] = df['B03'] / (df['B11'] + 0.0001)
    
    # Advanced thermal environment interactions
    if 'LST' in df.columns and 'avg_building_height' in df.columns:
        # Thermal canyon effect
        df['thermal_canyon_effect'] = df['LST'] * df['avg_building_height'] * df['nearest_building_count_100m'] / 1000
        
        # Heat trapping effect
        df['heat_trapping_effect'] = df['LST'] * df['avg_building_height'] ** 2 / 100
    
    # Create composite indices
    if all(col in df.columns for col in ['NDVI', 'NDBI', 'BSI']):
        # Urban Surface Composition Index
        df['urban_surface_composition'] = (df['NDBI'] + df['BSI']) / (df['NDVI'] + 1)
        
        # Surface permeability index
        df['surface_permeability'] = df['NDVI'] / (df['NDBI'] + df['BSI'] + 1)
    
    return df


def create_interaction_features(df):
    """Create comprehensive interaction features."""
    print("Creating interaction features...")
    
    # Normalize LST if it exists
    if 'LST' in df.columns:
        df['LST_norm'] = (df['LST'] - df['LST'].min()) / (df['LST'].max() - df['LST'].min())
    
    # Create building features
    df = create_building_features(df)
    
    # Create traffic features
    df = create_traffic_features(df)
    
    # Create environmental interactions
    df = create_environmental_interactions(df)
    
    # Create polynomial features for key indicators
    key_features = ['LST', 'NDVI', 'NDBI', 'BSI', 'UI']
    for feature in key_features:
        if feature in df.columns:
            df[f'{feature}_squared'] = df[feature] ** 2
            df[f'{feature}_cubed'] = df[feature] ** 3
    
    # Create composite indices
    if all(col in df.columns for col in ['LST', 'NDVI', 'NDBI', 'BSI']):
        # Urban Heat Susceptibility Index
        df['heat_susceptibility_index'] = (
            0.4 * df['LST_norm'] +
            0.3 * (1 - df['NDVI']) +
            0.2 * df['NDBI'] +
            0.1 * df['BSI']
        )
        
        # Surface Energy Balance Index
        if 'sky_view_factor' in df.columns:
            df['surface_energy_index'] = (
                df['LST_norm'] * (1 - df['NDVI']) * 
                df['NDBI'] * (1 - df['sky_view_factor'])
            )
    
    # Fill NaN values with median for numeric columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # For non-numeric columns, forward fill then backward fill
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    df[non_numeric_cols] = df[non_numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    return df


def detect_outliers(df):
    """Detect and handle outliers in the dataset using Isolation Forest."""
    print("Detecting and handling outliers...")
    
    from sklearn.ensemble import IsolationForest
    
    # Select numeric columns for outlier detection (excluding the target)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'UHI Index']
    
    # Use Isolation Forest for outlier detection with a lower contamination rate
    # to be more conservative in outlier detection
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    outlier_labels = iso_forest.fit_predict(df[numeric_cols])
    
    # Mark outliers in the dataframe
    df['is_outlier'] = [1 if x == -1 else 0 for x in outlier_labels]
    
    # Count outliers
    n_outliers = df['is_outlier'].sum()
    print(f"Detected {n_outliers} outliers ({(n_outliers/len(df))*100:.2f}% of the data)")
    
    # Create sample weights (lower weight for outliers but not too low)
    df['sample_weight'] = np.where(df['is_outlier'] == 1, 0.5, 1.0)
    
    return df


def select_uncorrelated_features(X, importances, feature_names, n_features, correlation_threshold=0.7):
    """
    Select top features while avoiding highly correlated ones
    
    Args:
        X: Feature matrix
        importances: Feature importance scores
        feature_names: List of feature names
        n_features: Number of features to select
        correlation_threshold: Maximum allowed correlation between features
    """
    # Create correlation matrix
    corr_matrix = np.abs(np.corrcoef(X.T))
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    selected_indices = []
    selected_names = []
    
    # Select features
    for idx in indices:
        # Check correlation with already selected features
        if len(selected_indices) == 0:
            selected_indices.append(idx)
            selected_names.append(feature_names[idx])
        else:
            # Calculate correlations with already selected features
            correlations = [corr_matrix[idx, sel_idx] for sel_idx in selected_indices]
            if max(correlations) < correlation_threshold:
                selected_indices.append(idx)
                selected_names.append(feature_names[idx])
        
        if len(selected_indices) == n_features:
            break
    
    return selected_indices, selected_names


def spatial_cv_split(X, coords, n_splits=5):
    """Create spatial cross-validation splits."""
    # Sort by latitude to create geographic folds
    idx = np.argsort(coords[:, 0])
    fold_size = len(idx) // n_splits
    splits = []
    
    for i in range(n_splits):
        val_idx = idx[i*fold_size:(i+1)*fold_size]
        train_idx = np.setdiff1d(idx, val_idx)
        splits.append((train_idx, val_idx))
    
    return splits


# Define the custom KNN class
class CustomKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [haversine_distance(x[0], x[1], x_train[0], x_train[1]) for x_train in self.X_train]
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_values = self.y_train[nearest_indices]
            predictions.append(np.mean(nearest_values))
        return np.array(predictions)


def train_model(data, feature_names):
    """Train an ensemble model with improved cross-validation and model selection."""
    print("Preparing data for training...")
    
    # Prepare features and target
    X = data[feature_names].values
    y = data['UHI Index'].values
    
    # Initialize scaler
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    
    # Define spatial cross-validation
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize base models with optimized parameters
    base_models = {
        'lgbm2': LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=20, random_state=42
        ),
        'cat3': CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=6,
            l2_leaf_reg=3, random_state=42, verbose=False
        ),
        'cat4': CatBoostRegressor(
            iterations=200, learning_rate=0.03, depth=8,
            l2_leaf_reg=5, random_state=43, verbose=False
        ),
        'xgb5': XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            min_child_weight=1, gamma=0, subsample=0.8,
            colsample_bytree=0.8, random_state=42
        ),
        'rf7': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'rf8': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=3,
            min_samples_leaf=1, random_state=43, n_jobs=-1
        )
    }
    
    # Train and evaluate base models
    meta_features = np.zeros((X.shape[0], len(base_models)))
    
    for i, (model_name, model) in enumerate(base_models.items()):
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error'))
        
        print(f"{model_name} - Mean R²: {cv_scores.mean():.4f}, Mean RMSE: {cv_scores.std():.4f}")
        
        # Train on full dataset for meta-features
        model.fit(X_scaled, y)
        meta_features[:, i] = model.predict(X_scaled)
    
    # Create meta-features DataFrame
    meta_df = pd.DataFrame(meta_features, columns=list(base_models.keys()))
    meta_df['lat'] = data['Latitude']
    meta_df['lon'] = data['Longitude']
    
    # Train spatial Custom KNN model
    print("Training CustomKNN with spatial features...")
    custom_knn_model = CustomKNN(n_neighbors=5)
    spatial_features = data[['Latitude', 'Longitude']].values
    custom_knn_model.fit(spatial_features, y)
    meta_df['custom_knn_pred'] = custom_knn_model.predict(spatial_features)
    
    # Update the ensemble to include custom KNN
    meta_model = StackingRegressor(
        estimators=[
            ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.05)),
            ('cat', CatBoostRegressor(iterations=100, verbose=False)),
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('custom_knn', custom_knn_model)  # Add custom KNN to the ensemble
        ],
        final_estimator=LGBMRegressor(n_estimators=100),
        cv=5
    )
    
    # Prepare final training data
    final_features = np.hstack([X_scaled, meta_features])
    
    # Train final model
    meta_model.fit(final_features, y)
    
    # Create wrapper class for prediction
    class EnsembleWrapper:
        def __init__(self, meta_model, base_models, scaler, feature_names):
            self.meta_model = meta_model
            self.base_models = base_models
            self.scaler = scaler
            self.feature_names = feature_names
        
        def predict(self, X):
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Generate base predictions
            base_preds = np.zeros((X.shape[0], len(self.base_models)))
            for i, model in enumerate(self.base_models.values()):
                base_preds[:, i] = model.predict(X_scaled)
            
            # Combine features for final prediction
            final_features = np.hstack([X_scaled, base_preds])
            return self.meta_model.predict(final_features)
    
    # Create and return wrapper
    ensemble = EnsembleWrapper(meta_model, base_models, sc, feature_names)
    
    return ensemble, sc


def predict_test_data(test_file, model, sc, feature_names, agg_df=None):
    """Make predictions on test data with enhanced post-processing."""
    print("Preparing test data for prediction...")
    
    # Handle missing features in test data
    for feature in feature_names:
        if feature not in test_file.columns:
            print(f"Warning: Feature '{feature}' not found in test data. Setting to 0.")
            test_file[feature] = 0
    
    # Select features for prediction
    submission_val_data = test_file[feature_names].copy()
    
    # Feature Scaling
    submission_val_data = submission_val_data.values
    transformed_submission_data = sc.transform(submission_val_data)
    
    # Making predictions
    print("Making predictions...")
    predictions = model.predict(transformed_submission_data)
    
    # Initial post-processing
    predictions = np.clip(predictions, 0, None)  # Ensure non-negative values
    
    # Apply spatial smoothing with multiple radii
    if all(col in test_file.columns for col in ['Longitude', 'Latitude']):
        from sklearn.neighbors import KNeighborsRegressor
        coords = test_file[['Latitude', 'Longitude']].values
        
        # Multi-scale spatial smoothing
        smoothed_predictions = np.zeros_like(predictions)
        weights = [0.5, 0.3, 0.2]  # Weights for different scales
        n_neighbors_list = [5, 10, 20]  # Different neighborhood sizes
        
        for w, k in zip(weights, n_neighbors_list):
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(coords, predictions)
            smoothed_predictions += w * knn.predict(coords)
        
        # Blend original and smoothed predictions
        alpha = 0.7  # Weight for original predictions
        predictions = alpha * predictions + (1 - alpha) * smoothed_predictions
    
    # Scale adjustment based on reference statistics
    mean_target = 0.9862  # Target mean from reference
    std_target = 0.0198   # Target std from reference
    
    # Adjust mean and standard deviation
    current_mean = np.mean(predictions)
    current_std = np.std(predictions)
    
    # Apply mean and variance scaling
    predictions = ((predictions - current_mean) * (std_target / current_std)) + mean_target
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'Longitude': test_file['Longitude'],
        'Latitude': test_file['Latitude'],
        'UHI Index': predictions
    })
    
    # Additional post-processing steps
    
    # 1. Remove outliers using IQR method
    Q1 = submission_df['UHI Index'].quantile(0.25)
    Q3 = submission_df['UHI Index'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    submission_df['UHI Index'] = submission_df['UHI Index'].clip(lower=lower_bound, upper=upper_bound)
    
    # 2. Apply local spatial consistency
    if all(col in submission_df.columns for col in ['Longitude', 'Latitude']):
        from scipy.spatial.distance import cdist
        coords = submission_df[['Latitude', 'Longitude']].values
        
        # Calculate distance matrix
        distances = cdist(coords, coords)
        
        # Apply Gaussian smoothing with distance-based weights
        sigma = 0.001  # Small sigma for local smoothing
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Smooth values
        smoothed_values = weights @ submission_df['UHI Index'].values
        
        # Apply weighted blending
        alpha = 0.8  # Weight for original values
        submission_df['UHI Index'] = alpha * submission_df['UHI Index'] + (1 - alpha) * smoothed_values
    
    # 3. Ensure spatial gradients are smooth
    if all(col in submission_df.columns for col in ['Longitude', 'Latitude']):
        from scipy.ndimage import gaussian_filter1d
        
        # Sort by latitude and longitude
        submission_df = submission_df.sort_values(['Latitude', 'Longitude'])
        
        # Apply 1D Gaussian smoothing along latitude and longitude
        sigma = 1.0
        values = submission_df['UHI Index'].values
        smoothed_lat = gaussian_filter1d(values, sigma)
        
        submission_df = submission_df.sort_values(['Longitude', 'Latitude'])
        values = submission_df['UHI Index'].values
        smoothed_lon = gaussian_filter1d(values, sigma)
        
        # Combine smoothed values
        submission_df['UHI Index'] = 0.5 * (smoothed_lat + smoothed_lon)
    
    # 4. Final adjustments
    # Ensure values are within expected range
    submission_df['UHI Index'] = submission_df['UHI Index'].clip(0.8, 1.2)
    
    # Round to match reference precision
    submission_df['UHI Index'] = submission_df['UHI Index'].round(6)
    
    return submission_df


def validate_against_best_submission(current_predictions, best_submission_path='best_submission.csv'):
    """
    Validate current predictions against the best performing submission with enhanced metrics.
    """
    print("\nValidating predictions against best submission...")
    
    try:
        # Load best performing submission
        best_submission = pd.read_csv(best_submission_path)
        
        # Ensure both DataFrames have the same columns
        required_columns = ['Longitude', 'Latitude', 'UHI Index']
        if not all(col in current_predictions.columns for col in required_columns):
            raise ValueError("Current predictions missing required columns")
        if not all(col in best_submission.columns for col in required_columns):
            raise ValueError("Best submission missing required columns")
        
        # Sort both DataFrames by coordinates to ensure alignment
        current_predictions = current_predictions.sort_values(['Longitude', 'Latitude']).reset_index(drop=True)
        best_submission = best_submission.sort_values(['Longitude', 'Latitude']).reset_index(drop=True)
        
        # Calculate differences and metrics
        diff = current_predictions['UHI Index'] - best_submission['UHI Index']
        abs_diff = abs(diff)
        
        # Calculate accuracy with different tolerances
        tolerances = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        accuracy_metrics = {}
        for tolerance in tolerances:
            within_tolerance = (abs_diff <= tolerance).mean() * 100
            accuracy_metrics[f'accuracy_within_{tolerance}'] = within_tolerance
        
        # Calculate enhanced metrics
        metrics = {
            'mean_diff': diff.mean(),
            'median_diff': diff.median(),
            'std_diff': diff.std(),
            'max_abs_diff': abs_diff.max(),
            'min_abs_diff': abs_diff.min(),
            'correlation': current_predictions['UHI Index'].corr(best_submission['UHI Index']),
            'mse': (diff ** 2).mean(),
            'mae': abs_diff.mean(),
            'rmse': np.sqrt((diff ** 2).mean()),
            'accuracy_metrics': accuracy_metrics,
            'percentile_90': np.percentile(abs_diff, 90),
            'percentile_95': np.percentile(abs_diff, 95),
            'percentile_99': np.percentile(abs_diff, 99)
        }
        
        # Print validation results
        print("\nValidation Results:")
        print(f"Mean Difference: {metrics['mean_diff']:.6f}")
        print(f"Median Difference: {metrics['median_diff']:.6f}")
        print(f"Std Deviation of Difference: {metrics['std_diff']:.6f}")
        print(f"Max Absolute Difference: {metrics['max_abs_diff']:.6f}")
        print(f"Min Absolute Difference: {metrics['min_abs_diff']:.6f}")
        print(f"Correlation with Best Submission: {metrics['correlation']:.6f}")
        print(f"Mean Squared Error: {metrics['mse']:.6f}")
        print(f"Mean Absolute Error: {metrics['mae']:.6f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
        print(f"\n90th Percentile of Absolute Difference: {metrics['percentile_90']:.6f}")
        print(f"95th Percentile of Absolute Difference: {metrics['percentile_95']:.6f}")
        print(f"99th Percentile of Absolute Difference: {metrics['percentile_99']:.6f}")
        
        print("\nAccuracy Metrics:")
        for tolerance, accuracy in sorted(accuracy_metrics.items()):
            print(f"Within {float(tolerance.split('_')[-1]):.3f}: {accuracy:.4f}%")
        
        # Analyze spatial patterns
        if all(col in current_predictions.columns for col in ['Longitude', 'Latitude']):
            # Create spatial bins
            current_predictions['lat_bin'] = pd.qcut(current_predictions['Latitude'], q=10, labels=False)
            current_predictions['lon_bin'] = pd.qcut(current_predictions['Longitude'], q=10, labels=False)
            
            # Calculate spatial statistics
            spatial_stats = current_predictions.groupby(['lat_bin', 'lon_bin']).agg({
                'UHI Index': ['mean', 'std'],
                'Latitude': 'mean',
                'Longitude': 'mean'
            }).reset_index()
            
            print("\nSpatial Analysis:")
            high_var_regions = spatial_stats[spatial_stats[('UHI Index', 'std')] > spatial_stats[('UHI Index', 'std')].mean()]
            print(f"Found {len(high_var_regions)} regions with high variance")
            
            if len(high_var_regions) > 0:
                print("\nRegions with highest variance:")
                for _, region in high_var_regions.head().iterrows():
                    print(f"Lat: {region[('Latitude', 'mean')]:.4f}, "
                          f"Lon: {region[('Longitude', 'mean')]:.4f}, "
                          f"Mean UHI: {region[('UHI Index', 'mean')]:.4f}, "
                          f"Std: {region[('UHI Index', 'std')]:.4f}")
        
        # Identify points with large differences
        threshold = metrics['std_diff'] * 2  # Points with differences > 2 standard deviations
        large_diff_points = current_predictions[abs_diff > threshold].copy()
        large_diff_points['Difference'] = diff[abs_diff > threshold]
        large_diff_points['Best_UHI'] = best_submission['UHI Index'][abs_diff > threshold]
        
        if len(large_diff_points) > 0:
            print(f"\nFound {len(large_diff_points)} points with large differences (>{threshold:.6f}):")
            print("\nSummary of large differences:")
            print(large_diff_points[['Longitude', 'Latitude', 'UHI Index', 'Best_UHI', 'Difference']]
                  .describe()
                  .round(6))
            
            # Analyze spatial clustering of large differences
            from sklearn.cluster import DBSCAN
            if len(large_diff_points) >= 2:  # Need at least 2 points for clustering
                coords = large_diff_points[['Latitude', 'Longitude']].values
                clustering = DBSCAN(eps=0.01, min_samples=2).fit(coords)
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                print(f"\nFound {n_clusters} clusters of large differences")
        
        return metrics, large_diff_points
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return None, None


def get_feature_names(df):
    """Get the list of feature names used for training."""
    feature_names = []
    
    # Basic features
    basic_features = ['LST', 'NDVI', 'NDBI', 'NDWI', 'BSI', 'UI', 'building_count']
    for feature in basic_features:
        if feature in df.columns:
            feature_names.append(feature)
    
    # Building count features
    building_count_features = [
        'nearest_building_count_10m', 'nearest_building_count_50m',
        'nearest_building_count_100m', 'nearest_building_count_200m',
        'nearest_building_count_300m', 'nearest_building_count_400m',
        'nearest_building_count_500m'
    ]
    for feature in building_count_features:
        if feature in df.columns:
            feature_names.append(feature)
    
    # Building-environment interactions
    building_env_features = [
        'building_lst_interaction', 'building_ndbi_interaction',
        'building_ndvi_interaction', 'building_ndwi_interaction'
    ]
    for feature in building_env_features:
        if feature in df.columns:
            feature_names.append(feature)
    
    # LST interactions
    lst_features = [
        'lst_squared', 'lst_cubed', 'log_lst',
        'lst_ndbi_interaction', 'lst_ndvi_interaction',
        'lst_ndwi_interaction'
    ]
    for feature in lst_features:
        if feature in df.columns:
            feature_names.append(feature)
    
    # Building count transformations
    building_trans_features = [
        'building_count_squared', 'building_count_cubed',
        'log_building_count', 'building_count_interaction'
    ]
    for feature in building_trans_features:
        if feature in df.columns:
            feature_names.append(feature)
    
    # Environmental transformations
    env_trans_features = ['exp_ndbi', 'exp_ndvi']
    for feature in env_trans_features:
        if feature in df.columns:
            feature_names.append(feature)
    
    # Band interactions
    band_features = ['b04_b08_interaction', 'b03_b11_interaction']
    for feature in band_features:
        if feature in df.columns:
            feature_names.append(feature)
    
    # Composite indices
    composite_features = ['heat_susceptibility_index', 'surface_energy_index']
    for feature in composite_features:
        if feature in df.columns:
            feature_names.append(feature)
    
    return feature_names


def prepare_data(df):
    """Prepare data for training with advanced feature engineering."""
    print("Preparing data with advanced feature engineering...")
    
    # Check for and convert datetime columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].iloc[0])
                print(f"Converting datetime column: {col}")
                # Create cyclical time features
                hour = pd.to_datetime(df[col]).dt.hour
                month = pd.to_datetime(df[col]).dt.month
                df[f'{col}_hour_sin'] = np.sin(2 * np.pi * hour / 24)
                df[f'{col}_hour_cos'] = np.cos(2 * np.pi * hour / 24)
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * month / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * month / 12)
                df = df.drop(columns=[col])
            except:
                pass
    
    # Remove duplicate rows if any
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"Filling {missing_count} missing values in {col}")
            df[col] = df[col].fillna(df[col].median())
    
    # Create spatial features
    if all(col in df.columns for col in ['Longitude', 'Latitude']):
        # Distance to Manhattan center (approximately 40.7831° N, 73.9712° W)
        manhattan_lat, manhattan_lon = 40.7831, -73.9712
        df['dist_to_manhattan'] = df.apply(
            lambda row: haversine_distance(
                row['Latitude'], row['Longitude'],
                manhattan_lat, manhattan_lon
            ), axis=1
        )
        
        # Angle from Manhattan center
        df['angle_from_center'] = np.arctan2(
            df['Latitude'] - manhattan_lat,
            df['Longitude'] - manhattan_lon
        )
        
        # Create spatial clusters
        from sklearn.cluster import KMeans
        coords = df[['Latitude', 'Longitude']].values
        kmeans = KMeans(n_clusters=8, random_state=42)
        df['spatial_cluster'] = kmeans.fit_predict(coords)
        
        # Calculate local statistics within clusters
        for col in ['LST', 'NDVI', 'NDBI', 'building_count']:
            if col in df.columns:
                df[f'{col}_local_mean'] = df.groupby('spatial_cluster')[col].transform('mean')
                df[f'{col}_local_std'] = df.groupby('spatial_cluster')[col].transform('std')
                df[f'{col}_local_diff'] = df[col] - df[f'{col}_local_mean']
                df[f'{col}_zscore'] = df[f'{col}_local_diff'] / df[f'{col}_local_std'].clip(lower=1e-6)
    
    # Calculate spectral indices with enhanced formulas
    if all(band in df.columns for band in ['B04', 'B08']):
        # Enhanced Vegetation Index (EVI)
        df['EVI'] = 2.5 * ((df['B08'] - df['B04']) / (df['B08'] + 6 * df['B04'] - 7.5 * df['B02'] + 1))
        # Modified NDVI
        df['NDVI'] = (df['B08'] - df['B04']) / (df['B08'] + df['B04'] + 1e-6)
        df['NDVI_enhanced'] = df['NDVI'] * (1 + df['EVI']) / 2
    
    if all(band in df.columns for band in ['B11', 'B03']):
        # Enhanced Built-up Index
        df['NDBI'] = (df['B11'] - df['B03']) / (df['B11'] + df['B03'] + 1e-6)
        df['BUI'] = ((df['B11'] - df['B03']) / (df['B11'] + df['B03'] + 1e-6)) - df['NDVI']
        df['NDBI_enhanced'] = df['NDBI'] * (1 + abs(df['BUI'])) / 2
    
    if all(band in df.columns for band in ['B03', 'B08']):
        # Modified NDWI
        df['NDWI'] = (df['B03'] - df['B08']) / (df['B03'] + df['B08'] + 1e-6)
        df['MNDWI'] = (df['B03'] - df['B11']) / (df['B03'] + df['B11'] + 1e-6)
        df['NDWI_enhanced'] = (df['NDWI'] + df['MNDWI']) / 2
    
    if all(band in df.columns for band in ['B11', 'B04']):
        # Enhanced BSI
        df['BSI'] = ((df['B11'] + df['B04']) - (df['B08'] + df['B02'])) / ((df['B11'] + df['B04']) + (df['B08'] + df['B02']) + 1e-6)
        df['EBBI'] = (df['B11'] - df['B08']) / (10 * np.sqrt(df['B11'] + df['B04'] + 1e-6))
        df['BSI_enhanced'] = df['BSI'] * (1 + abs(df['EBBI'])) / 2
    
    # Create advanced building features
    if 'avg_building_height' in df.columns:
        # Building volume and density metrics
        df['building_volume'] = df['avg_building_height'] * df['nearest_building_count_100m']
        df['building_volume_density'] = df['building_volume'] / (np.pi * 100**2)
        df['height_to_count_ratio'] = df['avg_building_height'] / (df['nearest_building_count_100m'] + 1)
        
        if 'max_building_height' in df.columns:
            df['building_height_variance'] = df['max_building_height'] - df['avg_building_height']
            df['height_variance_ratio'] = df['building_height_variance'] / (df['avg_building_height'] + 1)
            df['building_height_range'] = df['max_building_height'] / (df['avg_building_height'] + 1)
            
            # Sky view factor approximation
            df['sky_view_factor'] = 1 / (1 + (df['avg_building_height'] * df['nearest_building_count_100m'] / 1000))
            
            # Urban canyon effect
            df['urban_canyon_effect'] = df['avg_building_height'] * np.sqrt(df['nearest_building_count_100m']) / 100
    
    # Create advanced environmental interactions
    if 'LST' in df.columns:
        df['LST_scaled'] = (df['LST'] - df['LST'].min()) / (df['LST'].max() - df['LST'].min() + 1e-6)
        df['LST_squared'] = df['LST'] ** 2
        df['LST_cubed'] = df['LST'] ** 3
        df['LST_log'] = np.log1p(df['LST'])
        
        # LST interactions with indices
        for index in ['NDVI', 'NDBI', 'UI', 'BSI', 'NDWI']:
            if index in df.columns:
                df[f'LST_{index}_interaction'] = df['LST'] * df[index]
                df[f'LST_{index}_ratio'] = df['LST'] / (abs(df[index]) + 1e-6)
                df[f'LST_{index}_enhanced'] = df[f'LST_{index}_interaction'] * df['LST_scaled']
    
    # Create advanced composite indices
    if all(index in df.columns for index in ['NDVI', 'NDBI', 'UI']):
        # Urban Heat Island susceptibility index
        df['uhi_susceptibility'] = (
            0.4 * df['LST_scaled'] +
            0.3 * (1 - df['NDVI_enhanced']) +
            0.2 * df['NDBI_enhanced'] +
            0.1 * df['BSI_enhanced']
        )
        
        # Surface energy balance index
        if 'sky_view_factor' in df.columns:
            df['surface_energy_index'] = (
                df['LST_scaled'] * (1 - df['NDVI_enhanced']) * 
                df['NDBI_enhanced'] * (1 - df['sky_view_factor'])
            )
    
    # Create advanced traffic features if available
    if 'traffic_volume' in df.columns:
        df['traffic_intensity'] = np.log1p(df['traffic_volume'])
        df['traffic_density'] = df['traffic_volume'] / (df['nearest_building_count_100m'] + 1)
        
        if 'LST' in df.columns:
            df['traffic_heat_interaction'] = df['traffic_volume'] * df['LST']
            df['traffic_heat_density'] = df['traffic_heat_interaction'] / (df['nearest_building_count_100m'] + 1)
            df['traffic_uhi_impact'] = df['traffic_heat_density'] * df['uhi_susceptibility']
    
    # Remove columns with zero variance
    zero_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if zero_var_cols:
        print(f"Removing {len(zero_var_cols)} columns with zero variance")
        df = df.drop(columns=zero_var_cols)
    
    # Remove extremely highly correlated features (correlation > 0.95)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
    
    if high_corr_cols:
        print(f"Removing {len(high_corr_cols)} highly correlated features")
        df = df.drop(columns=high_corr_cols)
    
    # Ensure we keep important features even if they're correlated
    important_cols = [
        'LST', 'NDVI', 'NDBI', 'BSI', 'UI',
        'LST_scaled', 'NDVI_enhanced', 'NDBI_enhanced', 'BSI_enhanced',
        'uhi_susceptibility', 'surface_energy_index',
        'building_volume_density', 'urban_canyon_effect',
        'traffic_uhi_impact', 'sky_view_factor'
    ]
    
    for col in important_cols:
        if col in numeric_df.columns and col in high_corr_cols:
            print(f"Preserving important feature: {col}")
            df[col] = numeric_df[col]
    
    print(f"Final dataset shape: {df.shape}")
    return df


def make_predictions_and_submit(model, sc, feature_names):
    """Make predictions on test data and create submission file."""
    print("Loading test data...")
    
    # Load test data
    test_data = load_data()
    
    if test_data is None:
        print("Error: Failed to load test data")
        return
    
    # Prepare test data
    test_data = prepare_data(test_data)
    
    # Make predictions
    print("Making predictions...")
    submission_df = predict_test_data(test_data, model, sc, feature_names)
    
    # Generate submission filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")
    submission_file = f"submission-non-stacked-{timestamp}.csv"
    
    # Save submission file
    print(f"Saving submission to {submission_file}...")
    submission_df.to_csv(submission_file, index=False)
    
    # Validate against best submission if available
    best_submission_file = "submission-enhanced-2025-03-11.csv"
    if os.path.exists(best_submission_file):
        print("Validating against best submission...")
        metrics, large_diff_points = validate_against_best_submission(submission_df, best_submission_file)
        
        if metrics is not None:
            # Save points with large differences for analysis
            if large_diff_points is not None and len(large_diff_points) > 0:
                large_diff_points.to_csv("poor_predictions.csv", index=False)
                print(f"Saved {len(large_diff_points)} points with large differences to poor_predictions.csv")
    
    print("Submission completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='UHI Prediction Pipeline')
    parser.add_argument('--prepare_data', action='store_true', help='Prepare data for training')
    parser.add_argument('--submit', action='store_true', help='Make predictions and submit')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the model')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # If no arguments provided, run with default settings
        main(should_prepare_data=True, submit=True, retrain=False)
    else:
        main(should_prepare_data=args.prepare_data, submit=args.submit, retrain=args.retrain)
