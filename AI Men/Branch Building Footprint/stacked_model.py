import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_predict

class ClippedRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, random_state=42):
        self.rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.feature_ranges = {
            0: (-3, 1),    # Feature 0 clipping range
            1: (-1, 7),    # Feature 1 clipping range
            2: (-2, 2.5)   # Feature 2 clipping range
        }
    
    def clip_features(self, X):
        X_clipped = X.copy()
        for feat_idx, (min_val, max_val) in self.feature_ranges.items():
            if feat_idx < X.shape[1]:  # Check if feature exists
                X_clipped[:, feat_idx] = np.clip(X[:, feat_idx], min_val, max_val)
        return X_clipped
    
    def fit(self, X, y):
        X_clipped = self.clip_features(X)
        self.rf.fit(X_clipped, y)
        return self
    
    def predict(self, X):
        X_clipped = self.clip_features(X)
        return self.rf.predict(X_clipped)

class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, cv=5, random_state=42):
        self.n_estimators = n_estimators
        self.cv = cv
        self.random_state = random_state
        
        # Level 1 models
        self.main_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.clipped_model = ClippedRandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        # Level 2 (meta) model
        self.meta_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
    
    def fit(self, X, y):
        # Generate out-of-fold predictions for training meta-model
        main_preds = cross_val_predict(self.main_model, X, y, cv=self.cv)
        clipped_preds = cross_val_predict(self.clipped_model, X, y, cv=self.cv)
        
        # Train level 1 models on full training data
        self.main_model.fit(X, y)
        self.clipped_model.fit(X, y)
        
        # Create meta-features and train meta-model
        meta_features = np.column_stack([main_preds, clipped_preds])
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        # Get predictions from level 1 models
        main_preds = self.main_model.predict(X)
        clipped_preds = self.clipped_model.predict(X)
        
        # Create meta-features and get final predictions
        meta_features = np.column_stack([main_preds, clipped_preds])
        return self.meta_model.predict(meta_features)

def train_and_evaluate(X_train, y_train, X_val, y_val):
    """
    Train and evaluate the stacking model on training and validation data
    """
    # Initialize and train stacking model
    stacking_model = StackingRegressor()
    stacking_model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = stacking_model.predict(X_train)
    val_preds = stacking_model.predict(X_val)
    
    return stacking_model, train_preds, val_preds 