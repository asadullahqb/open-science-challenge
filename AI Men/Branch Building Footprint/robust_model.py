import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.covariance import EllipticEnvelope

class RobustUHIPredictor:
    def __init__(self):
        self.robust_scaler = RobustScaler()
        self.outlier_detector = EllipticEnvelope(contamination=0.1, random_state=42)
        
    def prepare_data(self):
        # Load data
        df = pd.read_csv('poor_predictions.csv')
        
        # Extract base features
        X = df[['feature_0', 'feature_1', 'feature_2']]
        y = df['actual_UHI']
        
        # Scale features using robust scaler (less sensitive to outliers)
        X_scaled = self.robust_scaler.fit_transform(X)
        
        # Detect and remove outliers
        outlier_labels = self.outlier_detector.fit_predict(X_scaled)
        X_clean = X_scaled[outlier_labels == 1]
        y_clean = y[outlier_labels == 1]
        
        print(f"Removed {len(y) - len(y_clean)} outlier samples")
        
        return train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    
    def create_ensemble(self):
        # Create base models with focused hyperparameters
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
        
        return [('rf', rf), ('gb', gb)]
    
    def cross_validate_models(self, X, y, models):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {name: [] for name, _ in models}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_preds = {}
            print(f"\nFold {fold}:")
            
            for name, model in models:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                val_pred = model.predict(X_val)
                fold_preds[name] = val_pred
                
                # Calculate score
                fold_score = r2_score(y_val, val_pred)
                cv_scores[name].append(fold_score)
                print(f"{name} R² Score: {fold_score:.4f}")
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, imp in enumerate(importances):
                        print(f"Feature {i} importance: {imp:.4f}")
            
            # Ensemble prediction for this fold
            ensemble_pred = np.mean([pred for pred in fold_preds.values()], axis=0)
            ensemble_score = r2_score(y_val, ensemble_pred)
            print(f"Ensemble R² Score: {ensemble_score:.4f}")
        
        # Print average scores
        print("\nAverage CV Scores:")
        for name in cv_scores:
            avg_score = np.mean(cv_scores[name])
            std_score = np.std(cv_scores[name])
            print(f"{name}: {avg_score:.4f} (±{std_score:.4f})")
    
    def train_and_evaluate(self):
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Create and evaluate models
        models = self.create_ensemble()
        print("\nPerforming 5-fold cross-validation:")
        self.cross_validate_models(X_train, y_train, models)

if __name__ == "__main__":
    predictor = RobustUHIPredictor()
    predictor.train_and_evaluate()
