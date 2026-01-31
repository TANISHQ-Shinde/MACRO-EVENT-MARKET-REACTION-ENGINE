"""
Streamlit Web App
Macro-Event Market Reaction Engine
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import modules
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import MarketImpactModel
from src.visualizer import Visualizer
import pandas as pd

# Page config
st.set_page_config(
    page_title="Macro-Event Market Engine",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process data"""
    loader = DataLoader()
    events = loader.create_sample_events()
    market = loader.load_market_data()
    event_windows = loader.get_event_windows()
    
    engineer = FeatureEngineer()
    features = engineer.extract_features(event_windows)
    X, y, feature_cols = engineer.prepare_ml_features(features)
    
    return loader, features, X, y, feature_cols, event_windows

@st.cache_resource
def train_models(X, y):
    """Train ML models"""
    model = MarketImpactModel()
    model.train_logistic(X, y)
    model.train_random_forest(X, y)
    return model

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🔥 Macro-Event Market Reaction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantifying News-Driven Market Risk</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data and training models...'):
        loader, features, X, y, feature_cols, event_windows = load_data()
        model = train_models(X, y)
        viz = Visualizer()
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Event Analysis", "Model Performance", "Live Prediction"])
    
    if page == "Dashboard":
        show_dashboard(features, viz)
    elif page == "Event Analysis":
        show_event_analysis(features, event_windows, viz)
    elif page == "Model Performance":
        show_model_performance(model, X, y, viz)
    else:
        show_live_prediction(model, feature_cols)

def show_dashboard(features, viz):
    """Dashboard page"""
    st.header("📊 Dashboard Overview")
    
    # Summary metrics
    stats = viz.create_summary_dashboard(features)
    cols = st.columns(5)
    for i, (key, value) in enumerate(stats.items()):
        with cols[i]:
            st.metric(key, value)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(viz.plot_severity_distribution(features), use_container_width=True)
    
    with col2:
        st.plotly_chart(viz.plot_returns_by_category(features), use_container_width=True)
    
    st.plotly_chart(viz.plot_volatility_spike(features), use_container_width=True)
    
    # Recent events table
    st.subheader("📅 Recent Events")
    display_cols = ['date', 'event', 'category', 'severity', 'return_3d', 'vol_spike']
    st.dataframe(
        features[display_cols].sort_values('date', ascending=False).head(10),
        use_container_width=True
    )

def show_event_analysis(features, event_windows, viz):
    """Event analysis page"""
    st.header("🔍 Event Deep Dive")
    
    # Select event
    event_list = features['event'].tolist()
    selected_event = st.selectbox("Select Event", event_list)
    
    # Event details
    event_data = features[features['event'] == selected_event].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Category", event_data['category'])
    with col2:
        st.metric("Severity", f"{event_data['severity']}/10")
    with col3:
        st.metric("3-Day Return", f"{event_data['return_3d']:.2f}%")
    with col4:
        st.metric("Vol Spike", "Yes" if event_data['vol_spike'] == 1 else "No")
    
    # Price chart
    fig = viz.plot_event_impact(event_windows, selected_event)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📈 Detailed Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['1-Day Return', '3-Day Return', '5-Day Return', '10-Day Return', 
                   'Max Drawdown', 'Vol Before', 'Vol After'],
        'Value': [
            f"{event_data.get('return_1d', 0):.2f}%",
            f"{event_data.get('return_3d', 0):.2f}%",
            f"{event_data.get('return_5d', 0):.2f}%",
            f"{event_data.get('return_10d', 0):.2f}%",
            f"{event_data.get('max_drawdown', 0):.2f}%",
            f"{event_data.get('vol_before', 0):.2f}%",
            f"{event_data.get('vol_after', 0):.2f}%"
        ]
    })
    st.table(metrics_df)

def show_model_performance(model, X, y, viz):
    """Model performance page"""
    st.header("🤖 Model Performance")
    
    # Model selection
    model_type = st.radio("Select Model", ["Logistic Regression", "Random Forest"])
    model_key = 'logistic' if model_type == "Logistic Regression" else 'rf'
    
    # Predictions
    results = model.evaluate(X, y, model_key)
    
    st.success(f"Model Accuracy: {results['accuracy']*100:.1f}%")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    importance = model.get_feature_importance(model_key)
    st.plotly_chart(viz.plot_feature_importance(importance), use_container_width=True)
    
    # Prediction confidence
    st.subheader("📊 Prediction Confidence Analysis")
    # This requires features dataframe - we'll need to pass it
    # For now, just show importance table
    st.dataframe(importance, use_container_width=True)

def show_live_prediction(model, feature_cols):
    """Live prediction page"""
    st.header("🎯 Predict Event Impact")
    
    st.markdown("### Input Event Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        severity = st.slider("Event Severity (1-10)", 1, 10, 5)
        category = st.selectbox("Category", [
            'monetary_policy', 'inflation', 'geopolitical',
            'financial_crisis', 'economic', 'commodity', 'pandemic'
        ])
        vol_before = st.number_input("Volatility Before (%)", 0.0, 100.0, 15.0)
    
    with col2:
        momentum_5d = st.number_input("5-Day Momentum (%)", -10.0, 10.0, 0.0)
    
    if st.button("🚀 Predict Impact", type="primary"):
        # Create feature vector
        features = pd.DataFrame([[
            severity,
            1 if category == 'monetary_policy' else 0,
            1 if category == 'inflation' else 0,
            1 if category == 'geopolitical' else 0,
            1 if category == 'financial_crisis' else 0,
            1 if category == 'economic' else 0,
            1 if category == 'commodity' else 0,
            1 if category == 'pandemic' else 0,
            vol_before,
            momentum_5d
        ]], columns=feature_cols)
        
        # Predict
        pred, proba = model.predict_impact(features, 'logistic')
        risk_score = proba[0] * 100
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score:.1f}/100")
        
        with col2:
            impact = "⚠️ NEGATIVE" if pred[0] == 1 else "✅ NEUTRAL"
            st.metric("Expected Impact", impact)
        
        with col3:
            confidence = "High" if abs(proba[0] - 0.5) > 0.3 else "Medium"
            st.metric("Confidence", confidence)
        
        # Risk interpretation
        st.markdown("### 📝 Interpretation")
        if risk_score > 70:
            st.error("🔴 **HIGH RISK**: This event is likely to cause significant negative market impact. Consider defensive positioning.")
        elif risk_score > 40:
            st.warning("🟡 **MODERATE RISK**: This event may cause some market volatility. Monitor closely.")
        else:
            st.success("🟢 **LOW RISK**: This event is unlikely to cause major market disruption.")

if __name__ == "__main__":
    main()