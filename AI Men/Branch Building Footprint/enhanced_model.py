import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

class EnhancedUHIPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        
    def prepare_data(self):
        # Load both poor predictions and original data
        poor_pred_df = pd.read_csv('poor_predictions.csv')
        
        # Extract features and target
        feature_cols = ['feature_0', 'feature_1', 'feature_2']
        X = poor_pred_df[feature_cols]
        y = poor_pred_df['actual_UHI']
        
        # Create polynomial features
        X_poly = self.poly.fit_transform(X)
        
        # Add interaction terms
        X_interact = pd.DataFrame(X_poly, columns=[f'feature_{i}' for i in range(X_poly.shape[1])])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_interact)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def train_rf_model(self, X_train, y_train):
        # Random Forest with hyperparameter tuning
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def train_gb_model(self, X_train, y_train):
        # Gradient Boosting with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"{model_name} R² Score: {r2:.4f}")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            importance_df = importance_df.sort_values('importance', ascending=False)
            print(f"\nTop 5 important features for {model_name}:")
            print(importance_df.head())
        
        return r2
    
    def train_and_evaluate(self):
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Train models
        print("Training Random Forest...")
        rf_model = self.train_rf_model(X_train, y_train)
        
        print("\nTraining Gradient Boosting...")
        gb_model = self.train_gb_model(X_train, y_train)
        
        # Evaluate models
        print("\nModel Evaluation:")
        rf_score = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        gb_score = self.evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
        
        # Weighted ensemble prediction
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        
        # Dynamic weighting based on model performance
        total_score = abs(rf_score) + abs(gb_score)
        rf_weight = abs(rf_score) / total_score
        gb_weight = abs(gb_score) / total_score
        
        ensemble_pred = (rf_weight * rf_pred + gb_weight * gb_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        print(f"\nWeighted Ensemble R² Score: {ensemble_r2:.4f}")

if __name__ == "__main__":
    predictor = EnhancedUHIPredictor()
    predictor.train_and_evaluate()
