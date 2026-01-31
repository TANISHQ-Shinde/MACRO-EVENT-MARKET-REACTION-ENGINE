"""
Model Module
ML models for predicting market impact
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class MarketImpactModel:
    def __init__(self):
        self.logistic_model = None
        self.rf_model = None
        self.feature_names = None
        
    def train_logistic(self, X, y):
        """Train logistic regression model"""
        self.logistic_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.logistic_model.fit(X, y)
        self.feature_names = X.columns.tolist()
        
        # Cross-validation score
        cv_scores = cross_val_score(self.logistic_model, X, y, cv=3)
        print(f"Logistic Regression CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return self.logistic_model
    
    def train_random_forest(self, X, y):
        """Train random forest model"""
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        self.rf_model.fit(X, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.rf_model, X, y, cv=3)
        print(f"Random Forest CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return self.rf_model
    
    def predict_impact(self, X, model_type='logistic'):
        """Predict market impact probability"""
        if model_type == 'logistic' and self.logistic_model:
            proba = self.logistic_model.predict_proba(X)[:, 1]
            pred = self.logistic_model.predict(X)
        elif model_type == 'rf' and self.rf_model:
            proba = self.rf_model.predict_proba(X)[:, 1]
            pred = self.rf_model.predict(X)
        else:
            raise ValueError("Model not trained yet")
        
        return pred, proba
    
    def get_feature_importance(self, model_type='rf'):
        """Get feature importance"""
        if model_type == 'logistic' and self.logistic_model:
            importance = np.abs(self.logistic_model.coef_[0])
            features = self.feature_names
        elif model_type == 'rf' and self.rf_model:
            importance = self.rf_model.feature_importances_
            features = self.feature_names
        else:
            return None
        
        return pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def evaluate(self, X, y, model_type='logistic'):
        """Evaluate model performance"""
        pred, proba = self.predict_impact(X, model_type)
        
        print(f"\n{model_type.upper()} MODEL EVALUATION")
        print("=" * 50)
        print("\nClassification Report:")
        print(classification_report(y, pred, target_names=['No Impact', 'Negative Impact']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, pred))
        
        return {
            'predictions': pred,
            'probabilities': proba,
            'accuracy': (pred == y).mean()
        }
    
    def score_event_severity(self, event_features):
        """
        Score event severity based on predicted impact probability
        Returns risk score 0-100
        """
        if self.logistic_model is None:
            raise ValueError("Model not trained")
        
        _, proba = self.predict_impact(event_features, 'logistic')
        risk_score = proba * 100
        
        return risk_score
    
    def save_models(self, logistic_path='models/logistic_model.pkl', 
                   rf_path='models/rf_model.pkl'):
        """Save trained models"""
        if self.logistic_model:
            joblib.dump(self.logistic_model, logistic_path)
        if self.rf_model:
            joblib.dump(self.rf_model, rf_path)
        print("Models saved successfully")
    
    def load_models(self, logistic_path='models/logistic_model.pkl',
                   rf_path='models/rf_model.pkl'):
        """Load trained models"""
        try:
            self.logistic_model = joblib.load(logistic_path)
            self.rf_model = joblib.load(rf_path)
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")