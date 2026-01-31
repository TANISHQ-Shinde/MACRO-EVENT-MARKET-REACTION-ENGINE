"""
Feature Engineering Module
Extracts quantitative features from events and market data
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def calculate_returns(self, prices, periods=[1, 3, 5, 10]):
        """Calculate forward returns after event"""
        returns = {}
        for period in periods:
            if len(prices) > period:
                ret = (prices.iloc[period] - prices.iloc[0]) / prices.iloc[0]
                returns[f'return_{period}d'] = ret * 100
            else:
                returns[f'return_{period}d'] = np.nan
        return returns
    
    def calculate_volatility(self, prices, window=5):
        """Calculate realized volatility"""
        if len(prices) < 2:
            return np.nan
        returns = prices.pct_change().dropna()
        return returns.std() * np.sqrt(252) * 100
    
    def detect_volatility_spike(self, prices_before, prices_after):
        """Detect if volatility spiked after event"""
        if len(prices_before) < 2 or len(prices_after) < 2:
            return 0
        
        vol_before = prices_before.pct_change().std()
        vol_after = prices_after.pct_change().std()
        
        if vol_before == 0 or pd.isna(vol_before):
            return 0
        
        spike_ratio = vol_after / vol_before
        return 1 if spike_ratio > 1.5 else 0
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown in window"""
        if len(prices) < 2:
            return 0
        
        cummax = prices.expanding().max()
        drawdown = (prices - cummax) / cummax
        return abs(drawdown.min()) * 100
    
    def encode_category(self, category):
        """One-hot encode event category"""
        categories = ['monetary_policy', 'inflation', 'geopolitical', 
                     'financial_crisis', 'economic', 'commodity', 'pandemic']
        return {f'cat_{cat}': 1 if cat == category else 0 for cat in categories}
    
    def extract_features(self, event_windows):
        """Extract all features for modeling"""
        features_list = []
        
        for window in event_windows:
            window_data = window['window_data']
            
            if len(window_data) < 2:
                continue
            
            # Split into before/after event
            event_idx = window_data[window_data['date'] == window['event_date']].index
            if len(event_idx) == 0:
                event_idx = len(window_data) // 2
            else:
                event_idx = event_idx[0]
            
            prices_before = window_data.loc[:event_idx, 'close']
            prices_after = window_data.loc[event_idx:, 'close']
            
            # Calculate features
            features = {
                'event': window['event'],
                'date': window['event_date'],
                'category': window['category'],
                'severity': window['severity'],
            }
            
            # Category encoding
            features.update(self.encode_category(window['category']))
            
            # Return features
            returns = self.calculate_returns(prices_after)
            features.update(returns)
            
            # Volatility features
            features['vol_before'] = self.calculate_volatility(prices_before)
            features['vol_after'] = self.calculate_volatility(prices_after)
            features['vol_spike'] = self.detect_volatility_spike(prices_before, prices_after)
            
            # Drawdown
            features['max_drawdown'] = self.calculate_max_drawdown(prices_after)
            
            # Price momentum before event
            if len(prices_before) >= 5:
                features['momentum_5d'] = (prices_before.iloc[-1] / prices_before.iloc[0] - 1) * 100
            else:
                features['momentum_5d'] = 0
            
            # Target variable: did market drop significantly in next 3 days?
            if 'return_3d' in features and not pd.isna(features['return_3d']):
                features['negative_impact'] = 1 if features['return_3d'] < -1 else 0
            else:
                features['negative_impact'] = 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def prepare_ml_features(self, df):
        """Prepare features for ML model"""
        feature_cols = ['severity', 'cat_monetary_policy', 'cat_inflation', 
                       'cat_geopolitical', 'cat_financial_crisis', 'cat_economic',
                       'cat_commodity', 'cat_pandemic', 'vol_before', 'momentum_5d']
        
        # Handle missing values
        X = df[feature_cols].fillna(0)
        
        # Target: negative market impact
        y = df['negative_impact']
        
        return X, y, feature_cols