# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import required dependencies
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from shapely.geometry import box
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def load_and_prepare_data(landsat_path, building_footprint_path):
    """
    Load and prepare the Landsat and building footprint data
    """
    # Load Landsat data
    with rasterio.open(landsat_path) as src:
        landsat_data = src.read(1)  # Assuming single band
        transform = src.transform
        crs = src.crs
    
    # Load building footprints
    building_gdf = gpd.read_file(building_footprint_path)
    return landsat_data, building_gdf, transform, crs

def calculate_building_metrics(building_gdf, landsat_data, transform):
    """
    Calculate building metrics including temperature statistics
    """
    # Initialize lists to store results
    mean_temps = []
    building_areas = []
    
    for idx, building in building_gdf.iterrows():
        # Get building geometry
        geom = building.geometry
        
        # Calculate building area
        area = geom.area
        
        # Extract temperature values for the building
        coords = getattr(geom, 'coords', None)
        if coords:
            # Convert geometry to pixel coordinates
            rows, cols = rasterio.transform.rowcol(transform, *zip(*coords))
            temp_values = landsat_data[rows, cols]
            
            # Calculate statistics
            mean_temp = np.mean(temp_values)
            
            mean_temps.append(mean_temp)
            building_areas.append(area)
    
    # Add metrics to GeoDataFrame
    building_gdf['mean_temperature'] = mean_temps
    building_gdf['building_area'] = building_areas
    
    return building_gdf

def train_model(data):
    """
    Train XGBoost model to predict temperature based on building characteristics
    """
    # Prepare features and target
    X = data[['building_area']]
    y = data['mean_temperature']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

def main():
    # Define file paths
    landsat_path = "path/to/landsat_data.tif"
    building_footprint_path = "path/to/building_footprints.geojson"
    
    # Load and prepare data
    landsat_data, building_gdf, transform, crs = load_and_prepare_data(
        landsat_path, building_footprint_path
    )
    
    # Calculate building metrics
    processed_data = calculate_building_metrics(building_gdf, landsat_data, transform)
    
    # Train and evaluate model
    model, mse, r2 = train_model(processed_data)
    
    print(f"Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()