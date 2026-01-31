# 🔥 Macro-Event Market Reaction Engine

A quantitative finance project that analyzes how financial markets react to major macroeconomic events such as central bank decisions, inflation releases, and geopolitical shocks.

This project focuses on **event-driven market analysis**, combining historical data, feature engineering, and machine learning to study price direction, volatility changes, and risk intensity around macro events.

---

## 📌 What This Project Does

The Macro-Event Market Reaction Engine studies historical macroeconomic events and evaluates how markets behaved **before and after** those events.

For each event, the engine analyzes:

- **Market Direction** – Did the market move up or down after the event?
- **Volatility Spike** – Did volatility increase significantly?
- **Risk Score** – A 0–100 score representing the relative market impact of the event

This is designed as a **research and learning tool**, not a live trading system.

---

## 🧠 Quant Skills Demonstrated

- **Event Study Methodology** – Measuring market reactions around discrete events  
- **Feature Engineering** – Transforming raw price data into predictive signals  
- **Regime Analysis** – Identifying volatility expansion and contraction  
- **Machine Learning** – Logistic Regression and Random Forest classification  
- **Model Evaluation** – Cross-validation and classification metrics  
- **Data Visualization** – Interactive dashboards using Streamlit and Plotly  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/macro-event-market-reaction-engine.git
cd macro-event-market-reaction-engine

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
The app will open in your browser at:
Copy code

http://localhost:8501
🗂️ Project Structure
Copy code

macro-event-engine/
│
├── data/                     # Data storage (user-provided)
│
├── src/
│   ├── data_loader.py        # Load market and event data
│   ├── feature_engineering.py# Feature creation
│   ├── model.py              # ML models
│   └── visualizer.py         # Charts & plots
│
├── app/
│   └── main.py               # Streamlit web interface
│
├── requirements.txt          # Dependencies
└── README.md                 # Documentation

📊 Features
1️⃣ Dashboard Overview
Summary statistics of macro events
Event category distribution
Average post-event returns
Volatility spike analysis.
2️⃣ Event Deep Dive
Individual event analysis
Price charts with event markers
Forward returns (1d, 3d, 5d, 10d)
Drawdown analysis.
3️⃣ Model Performance
Logistic Regression & Random Forest models
Feature importance analysis
Cross-validation scores
Classification metrics.
4️⃣ Scenario-Based Analysis (Prototype)
Input custom event parameters
Generate estimated risk scores
Confidence bands for uncertainty interpretation.

📚 Data Sources
Market Data:
S&P 500 index (SPY) via yfinance
Macro Events:
Placeholder structure for major macroeconomic events such as:
Federal Reserve rate decisions
CPI / inflation reports
Employment data (NFP)
Geopolitical and financial system shocks.

⚠️ No proprietary or paid datasets are included.
Users can plug in their own event calendars or public macro datasets.

🧮 Features Engineered
Event severity score (1–10)
Event category encoding
Pre-event volatility
Pre-event momentum
Post-event returns (1d, 3d, 5d, 10d)
Volatility spike indicator
Maximum drawdown.

🤖 Models Used
Logistic Regression
Baseline linear classifier with L2 regularization
Random Forest
Non-linear ensemble model capturing complex interactions
Evaluation
3-fold cross-validation with classification reports.

📈 Sample Results (Illustrative)
Event Type
Avg 3D Return
Vol Spike Rate
Risk Score
Financial Crisis
-3.2%
85%
87 / 100
Fed Rate Hike
-0.8%
45%
62 / 100
Geopolitical
-1.5%
60%
71 / 100
Strong Jobs Data
+0.5%
25%
38 / 100
⚠️ These results are illustrative examples used to demonstrate the framework.
Actual results depend on the dataset and configuration.

💼 Use Cases
Portfolio Risk Management – Quantifying event-driven risk exposure
Trading Research – Identifying high-impact macro events
Market Analysis – Studying historical event reactions
Interview / Resume Project – Demonstrating quant + ML skills.

🛠️ Tech Stack
Python – Core language
Pandas / NumPy – Data manipulation
Scikit-learn – Machine learning
Streamlit – Web interface
Plotly – Interactive visualizations
yfinance – Market data

⚠️ Limitations
Uses historical data; does not predict future market behavior
Event definitions are simplified abstractions
Results depend heavily on event window and feature selection
Not intended for live trading or financial advice.

🔮 Future Enhancements
Adding earnings and election events
Integrate sentiment analysis from news headlines
Expand to FX, commodities, and bonds
Real-time event monitoring via APIs
Portfolio stress-testing module.

🤝 Contributing
Contributions are welcome:
Fork the repository
Create a feature branch
Submit a pull request.

📄 License
MIT License — see LICENSE file for details.
⭐ If you find this project useful, feel free to star the repository!
Built as a learning-focused quantitative finance project.
