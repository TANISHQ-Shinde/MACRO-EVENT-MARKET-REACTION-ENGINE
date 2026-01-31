"""
Visualizer Module
Create charts and visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

class Visualizer:
    def __init__(self):
        sns.set_style('darkgrid')
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_event_impact(self, event_windows, event_name):
        """Plot price movement around specific event"""
        for window in event_windows:
            if window['event'] == event_name:
                data = window['window_data'].copy()
                event_date = window['event_date']
                
                # Make sure event_date is a Timestamp
                if isinstance(event_date, str):
                    event_date = pd.to_datetime(event_date)
                
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data['close'],
                    mode='lines',
                    name='S&P 500',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Add event marker as a scatter point
                event_price = data[data['date'] == event_date]['close']
                if len(event_price) > 0:
                    fig.add_trace(go.Scatter(
                        x=[event_date],
                        y=[event_price.iloc[0]],
                        mode='markers+text',
                        name='Event',
                        marker=dict(size=15, color='red', symbol='diamond'),
                        text=['📍 Event'],
                        textposition='top center',
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title=f"Market Reaction: {event_name}",
                    xaxis_title="Date",
                    yaxis_title="S&P 500 Price ($)",
                    template="plotly_white",
                    height=400,
                    hovermode='x unified'
                )
                
                return fig
        return None
    
    def plot_severity_distribution(self, features_df):
        """Plot distribution of event severity"""
        fig = px.histogram(
            features_df,
            x='severity',
            color='category',
            title='Event Severity Distribution by Category',
            labels={'severity': 'Severity Score', 'count': 'Number of Events'},
            barmode='stack'
        )
        fig.update_layout(template="plotly_white", height=400)
        return fig
    
    def plot_returns_by_category(self, features_df):
        """Plot average returns by event category"""
        avg_returns = features_df.groupby('category')['return_3d'].mean().sort_values()
        
        fig = px.bar(
            x=avg_returns.values,
            y=avg_returns.index,
            orientation='h',
            title='Average 3-Day Returns by Event Category',
            labels={'x': '3-Day Return (%)', 'y': 'Category'},
            color=avg_returns.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        return fig
    
    def plot_feature_importance(self, importance_df):
        """Plot feature importance"""
        top_features = importance_df.head(10)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importance',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        fig.update_layout(template="plotly_white", height=400)
        return fig
    
    def plot_volatility_spike(self, features_df):
        """Plot volatility before vs after events"""
        # Remove any NaN values
        plot_df = features_df.dropna(subset=['vol_before', 'vol_after'])
        
        if len(plot_df) == 0:
            # Return empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for volatility analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title='Volatility Before vs After Events',
                template="plotly_white",
                height=500
            )
            return fig
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_df['vol_before'],
            y=plot_df['vol_after'],
            mode='markers',
            marker=dict(
                size=plot_df['severity'] * 2,
                color=plot_df['severity'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Severity")
            ),
            text=plot_df['event'],
            hovertemplate='<b>%{text}</b><br>Vol Before: %{x:.2f}%<br>Vol After: %{y:.2f}%<extra></extra>'
        ))
        
        # Add diagonal line
        max_vol = max(plot_df['vol_before'].max(), plot_df['vol_after'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_vol],
            y=[0, max_vol],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title='Volatility Before vs After Events',
            xaxis_title='Volatility Before (%)',
            yaxis_title='Volatility After (%)',
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def plot_prediction_confidence(self, features_df, predictions, probabilities):
        """Plot prediction confidence"""
        result_df = features_df.copy()
        result_df['prediction'] = predictions
        result_df['probability'] = probabilities
        
        # Remove NaN values
        result_df = result_df.dropna(subset=['severity', 'probability', 'vol_after'])
        
        if len(result_df) == 0:
            # Return empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for prediction confidence",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title='Prediction Confidence vs Event Severity',
                template="plotly_white",
                height=500
            )
            return fig
        
        fig = px.scatter(
            result_df,
            x='severity',
            y='probability',
            color='category',
            size='vol_after',
            hover_data=['event'],
            title='Prediction Confidence vs Event Severity',
            labels={
                'severity': 'Event Severity',
                'probability': 'Negative Impact Probability',
                'vol_after': 'Volatility After'
            }
        )
        fig.update_layout(template="plotly_white", height=500)
        return fig
    
    def create_summary_dashboard(self, features_df):
        """Create summary statistics"""
        # Calculate stats safely with NaN handling
        total_events = len(features_df)
        
        avg_return = features_df['return_3d'].mean()
        avg_return_str = f"{avg_return:.2f}%" if not pd.isna(avg_return) else "N/A"
        
        negative_count = features_df['negative_impact'].sum()
        negative_pct = (negative_count / total_events * 100) if total_events > 0 else 0
        negative_str = f"{int(negative_count)} ({negative_pct:.1f}%)"
        
        vol_spike = features_df['vol_spike'].mean() * 100
        vol_spike_str = f"{vol_spike:.1f}%" if not pd.isna(vol_spike) else "N/A"
        
        max_dd = features_df['max_drawdown'].max()
        max_dd_str = f"{max_dd:.2f}%" if not pd.isna(max_dd) else "N/A"
        
        stats = {
            'Total Events': total_events,
            'Avg 3d Return': avg_return_str,
            'Negative Impact Events': negative_str,
            'Avg Volatility Spike': vol_spike_str,
            'Max Drawdown': max_dd_str
        }
        return stats