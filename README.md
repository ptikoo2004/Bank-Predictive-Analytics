# 🏦 Bank Predictive Analytics Dashboard

A comprehensive Streamlit dashboard for banking analytics with machine learning-powered predictions for deposits, advances, profitability, and key financial ratios.

## 🚀 Features

- **📈 Deposits Analysis**: Historical trends and future predictions for all deposit types
- **💰 Advances Analysis**: Comprehensive analysis of retail, agriculture, MSME, and corporate advances  
- **📊 Profitability Analysis**: Income, expenditure, and profit forecasting
- **📋 Key Ratios Analysis**: Financial ratios tracking and prediction
- **🔮 Complete Forecast Report**: 5-year comprehensive business projections

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models
- **Python 3.8+**: Programming language

## 📦 Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bank-analytics-dashboard.git
cd bank-analytics-dashboard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Set the main file path to `app.py`
6. Click "Deploy"

### Deploy to Heroku

1. Install Heroku CLI
2. Login to Heroku: `heroku login`
3. Create a new app: `heroku create your-app-name`
4. Push to Heroku: `git push heroku main`

### Deploy to Railway

1. Connect your GitHub repository to [Railway](https://railway.app)
2. Select this repository
3. Railway will automatically detect and deploy the Streamlit app

## 📊 Data Overview

The dashboard uses historical banking data including:

- **Deposits**: Current Account, Savings Bank, CASA, Term Deposits
- **Advances**: Retail, Agriculture, MSME, Corporate advances
- **Profitability**: Interest income/expenditure, operating metrics, net profit
- **Key Ratios**: ROA, ROE, NIM, Cost-to-Income, CRAR

## 🤖 Machine Learning Models

The application employs multiple ML models for predictions:

- **Random Forest Regressor**: Ensemble method for robust predictions
- **Gradient Boosting Regressor**: Sequential learning for trend capture
- **Linear Regression**: Baseline linear trend analysis

## 📈 Key Metrics Predicted

- Total Deposits & Components (5-year forecast)
- Total Advances & Segments (5-year forecast) 
- Profitability Metrics (Net Profit, Operating Profit)
- Financial Ratios (ROA, ROE, NIM, etc.)
- CAGR and Growth Trajectories

## 🎯 Use Cases

- **Bank Management**: Strategic planning and target setting
- **Investors**: Performance analysis and future projections  
- **Analysts**: Comprehensive financial modeling
- **Regulators**: Risk assessment and compliance monitoring

## 📁 Project Structure

```
bank-analytics-dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .streamlit/           # Streamlit configuration
│   └── config.toml
├── Procfile              # Heroku deployment
├── setup.sh              # Heroku setup script
└── .gitignore           # Git ignore file
```

## 🔧 Configuration

The app uses Streamlit's built-in configuration. You can customize:

- Page layout and styling
- Chart themes and colors
- Model parameters
- Prediction horizons

## 📊 Dashboard Sections

### 1. Deposits Analysis
- Historical deposit trends
- Component-wise analysis (Current, Savings, Term)
- CASA ratio tracking
- 5-year deposits forecast

### 2. Advances Analysis  
- Segment-wise advances breakdown
- Retail vs Corporate lending trends
- Risk-adjusted growth projections
- Portfolio composition analysis

### 3. Profitability Analysis
- Income vs expenditure trends
- Operating efficiency metrics
- Net profit predictions
- Margin analysis

### 4. Key Ratios Analysis
- Return ratios (ROA, ROE)
- Efficiency ratios (NIM, Cost-to-Income)
- Capital adequacy (CRAR)
- Trend analysis and forecasting

### 5. Complete Forecast Report
- Executive summary dashboard
- 5-year comprehensive projections
- Risk assessment
- Growth trajectory analysis

## 🚨 Disclaimer

This dashboard is for educational and analytical purposes. The predictions are based on historical data and should not be used as the sole basis for financial decisions. Always consult with financial professionals for investment advice.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

⭐ Star this repository if you found it helpful!
