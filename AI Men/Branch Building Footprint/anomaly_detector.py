import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Initialize multiple anomaly detection models
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        self.elliptic_envelope = EllipticEnvelope(
            contamination=0.1,
            random_state=42
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1
        )
        
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            nu=0.1
        )
    
    def load_and_prepare_data(self):
        # Load the poor predictions dataset
        df = pd.read_csv('poor_predictions.csv')
        
        # Extract features and target
        features = ['feature_0', 'feature_1', 'feature_2']
        X = df[features]
        y = df['actual_UHI']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, df
    
    def detect_anomalies(self, X):
        # Get predictions from each model (-1 for anomalies, 1 for normal samples)
        if_pred = self.isolation_forest.fit_predict(X)
        ee_pred = self.elliptic_envelope.fit_predict(X)
        lof_pred = self.lof.fit_predict(X)
        svm_pred = self.one_class_svm.fit_predict(X)
        
        # Convert predictions to binary (0 for anomalies, 1 for normal)
        predictions = {
            'IsolationForest': [1 if x == 1 else 0 for x in if_pred],
            'EllipticEnvelope': [1 if x == 1 else 0 for x in ee_pred],
            'LocalOutlierFactor': [1 if x == 1 else 0 for x in lof_pred],
            'OneClassSVM': [1 if x == 1 else 0 for x in svm_pred]
        }
        
        return predictions
    
    def analyze_anomalies(self):
        # Load and prepare data
        X_scaled, y, original_df = self.load_and_prepare_data()
        
        # Detect anomalies using multiple methods
        predictions = self.detect_anomalies(X_scaled)
        
        # Create consensus voting
        consensus = np.zeros(len(X_scaled))
        for pred_array in predictions.values():
            consensus += pred_array
        
        # Consider a point anomalous if majority of detectors agree
        final_anomalies = consensus <= 2  # anomaly if more than 2 detectors mark it
        
        # Add results to original dataframe
        results_df = original_df.copy()
        for method, preds in predictions.items():
            results_df[f'{method}_anomaly'] = [1 if x == 0 else 0 for x in preds]
        results_df['consensus_anomaly'] = final_anomalies.astype(int)
        
        # Print summary statistics
        print("\nAnomaly Detection Summary:")
        print("-" * 50)
        for method in predictions.keys():
            anomaly_count = sum(results_df[f'{method}_anomaly'])
            print(f"{method}: {anomaly_count} anomalies detected ({anomaly_count/len(results_df)*100:.1f}%)")
        
        consensus_count = sum(results_df['consensus_anomaly'])
        print(f"\nConsensus Anomalies: {consensus_count} ({consensus_count/len(results_df)*100:.1f}%)")
        
        # Analyze model predictions for anomalies vs normal samples
        print("\nPrediction Analysis for Consensus Anomalies vs Normal Samples:")
        print("-" * 50)
        model_cols = [col for col in results_df.columns if '_prediction' in col]
        
        for model_col in model_cols:
            normal_r2 = r2_score(
                results_df[~final_anomalies]['actual_UHI'],
                results_df[~final_anomalies][model_col]
            )
            anomaly_r2 = r2_score(
                results_df[final_anomalies]['actual_UHI'],
                results_df[final_anomalies][model_col]
            )
            print(f"\n{model_col}:")
            print(f"Normal samples R² score: {normal_r2:.4f}")
            print(f"Anomaly samples R² score: {anomaly_r2:.4f}")
        
        # Save filtered dataset (non-anomalous samples)
        normal_samples = results_df[~final_anomalies]
        normal_samples.to_csv('filtered_predictions.csv', index=False)
        print(f"\nSaved {len(normal_samples)} normal samples to 'filtered_predictions.csv'")
        
        # Plot feature distributions for normal vs anomalous samples
        self.plot_distributions(results_df, final_anomalies)
        
        return results_df
    
    def plot_distributions(self, df, anomalies):
        features = ['feature_0', 'feature_1', 'feature_2']
        fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
        
        for i, feature in enumerate(features):
            axes[i].hist(df[~anomalies][feature], bins=20, alpha=0.5, label='Normal', density=True)
            axes[i].hist(df[anomalies][feature], bins=20, alpha=0.5, label='Anomaly', density=True)
            axes[i].set_title(feature)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        plt.close()

if __name__ == "__main__":
    from sklearn.metrics import r2_score
    detector = AnomalyDetector()
    results = detector.analyze_anomalies()
