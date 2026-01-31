"""
Data Loader Module
Handles loading historical events and market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self):
        self.events_df = None
        self.market_df = None
    
    def create_sample_events(self):
        """Create sample macro events dataset"""
        events = [
            # Fed Rate Events
            {"date": "2022-03-16", "event": "Fed Rate Hike 25bps", "category": "monetary_policy", "severity": 6},
            {"date": "2022-05-04", "event": "Fed Rate Hike 50bps", "category": "monetary_policy", "severity": 8},
            {"date": "2022-11-02", "event": "Fed Rate Hike 75bps", "category": "monetary_policy", "severity": 9},
            {"date": "2023-07-26", "event": "Fed Rate Hike 25bps", "category": "monetary_policy", "severity": 6},
            
            # Inflation Reports
            {"date": "2022-06-10", "event": "CPI 8.6% YoY", "category": "inflation", "severity": 9},
            {"date": "2022-09-13", "event": "CPI 8.3% YoY (higher than expected)", "category": "inflation", "severity": 8},
            {"date": "2023-01-12", "event": "CPI 6.5% YoY", "category": "inflation", "severity": 7},
            {"date": "2023-11-14", "event": "CPI 3.2% YoY", "category": "inflation", "severity": 5},
            
            # Geopolitical Events
            {"date": "2022-02-24", "event": "Russia Invades Ukraine", "category": "geopolitical", "severity": 10},
            {"date": "2023-10-07", "event": "Israel-Hamas Conflict Begins", "category": "geopolitical", "severity": 8},
            {"date": "2024-04-13", "event": "Iran-Israel Tensions Escalate", "category": "geopolitical", "severity": 7},
            
            # Banking Crisis
            {"date": "2023-03-10", "event": "Silicon Valley Bank Collapse", "category": "financial_crisis", "severity": 10},
            {"date": "2023-03-12", "event": "Signature Bank Seized", "category": "financial_crisis", "severity": 9},
            {"date": "2023-03-19", "event": "Credit Suisse Emergency Takeover", "category": "financial_crisis", "severity": 9},
            
            # Economic Data
            {"date": "2022-07-28", "event": "GDP Contraction Q2 (Recession fears)", "category": "economic", "severity": 8},
            {"date": "2023-05-05", "event": "Strong Jobs Report (512k)", "category": "economic", "severity": 6},
            {"date": "2024-01-26", "event": "GDP Growth 3.3%", "category": "economic", "severity": 5},
            
            # Oil Shocks
            {"date": "2022-03-07", "event": "Oil Hits $130 (Russia sanctions)", "category": "commodity", "severity": 9},
            {"date": "2023-10-09", "event": "Oil Spikes on Middle East Conflict", "category": "commodity", "severity": 7},
            
            # COVID Related
            {"date": "2022-01-03", "event": "Omicron Variant Surge", "category": "pandemic", "severity": 6},
        ]
        
        self.events_df = pd.DataFrame(events)
        self.events_df['date'] = pd.to_datetime(self.events_df['date'])
        return self.events_df
    
    def load_market_data(self, ticker='SPY', start_date='2022-01-01', end_date='2024-12-31'):
        """Load market data - using synthetic data for demo"""
        print("⚠️ Using synthetic market data for demo")
        return self._create_synthetic_market_data(start_date, end_date)
    
    def _create_synthetic_market_data(self, start_date, end_date):
        """Fallback: create synthetic market data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic S&P 500-like price movement
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = 400 * np.exp(np.cumsum(returns))
        
        self.market_df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(50000000, 150000000, len(dates))
        })
        return self.market_df
    
    def merge_events_market(self):
        """Merge events with market data"""
        if self.events_df is None or self.market_df is None:
            raise ValueError("Load events and market data first")
        
        # Merge on date
        merged = pd.merge_asof(
            self.events_df.sort_values('date'),
            self.market_df.sort_values('date'),
            on='date',
            direction='backward'
        )
        return merged
    
    def get_event_windows(self, window_before=5, window_after=10):
        """Get price data around each event"""
        results = []
        
        for _, event in self.events_df.iterrows():
            event_date = event['date']
            
            # Get market data window
            window_start = event_date - timedelta(days=window_before)
            window_end = event_date + timedelta(days=window_after)
            
            window_data = self.market_df[
                (self.market_df['date'] >= window_start) & 
                (self.market_df['date'] <= window_end)
            ].copy()
            
            if len(window_data) > 0:
                results.append({
                    'event': event['event'],
                    'event_date': event_date,
                    'category': event['category'],
                    'severity': event['severity'],
                    'window_data': window_data
                })
        
        return results