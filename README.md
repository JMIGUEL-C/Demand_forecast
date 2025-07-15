# Energy Demand Forecasting with LSTM

This project aims to predict daily electricity demand in the city of **Cali, Colombia**, using a **LSTM (Long Short-Term Memory)** recurrent neural network model.

Historical data is obtained from the **XM** public API, operator of the National Interconnected System and administrator of Colombia's Wholesale Energy Market.

The complete workflow covers:

- Data extraction from the XM API
- Preprocessing and feature creation
- Model training and evaluation
- Results visualization through an **interactive dashboard**

---

## Project Structure

```
Pronostico_De_Demanda/
│
├── data/
│   └── xm_api_data.csv                  # Raw data downloaded from XM API
│
├── notebooks/
│   └── LSTM_Energy_Demand_Forecasting.ipynb  # Complete experimentation process
│
├── src/
│   ├── data_loader.py                  # Script to download data
│   ├── processing.py                   # Cleaning and preprocessing functions
│   ├── lstm_model.py                   # LSTM model architecture
│   ├── train.py                        # Model training
│   ├── evaluate.py                     # Model evaluation
│   └── utils.py                        # Utility functions
│
├── dashboard/
│   └── app.py                          # Interactive dashboard with Streamlit
│
├── results/
│   ├── predicciones_vs_reales.png     # Results plot
│   └── metrics.txt                    # Metrics (RMSE, MAE, R²)
│
├── venv/                               # Virtual environment (optional)
├── requirements.txt                    # Project dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Jupyter Notebook
- (Optional) Virtualenv or conda

### Installation

```bash
git clone <repository-URL>
cd Pronostico_De_Demanda

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Usage

### 1. Get the data

You can download data from the XM API:

- Using the notebook: `notebooks/LSTM_Energy_Demand_Forecasting.ipynb`
- Or running directly:

```bash
python src/data_loader.py
```

### 2. Model Training

Follow the notebook steps:

- Data cleaning
- Feature engineering (lags, holidays, etc.)
- Scaling
- Model training and evaluation
- Save results

### 3. Interactive Dashboard

Run the app to visualize results:

```bash
streamlit run dashboard/app.py
```

You will be able to:

- View predictions vs actual values
- Get a prediction for a specific day
- Explore model metrics

---

## Results

After training, in the `results/` folder you will find:

- `metrics.txt`: Metrics such as RMSE, MAE, and R².
- `predicciones_lstm`: Graphical comparison between actual and predicted demand.

---

## Main Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical calculations
- `torch`: LSTM neural network
- `scikit-learn`: Scaling and metrics
- `streamlit`: Interactive dashboard
- `plotly`: Interactive plots
- `holidays`: Calculation of holidays in Colombia

---

## Author

**Miguel Correa**  
Intelligent Energy Engineering – Universidad Icesi  
GitHub: [@JMIGUEL-C](https://github.com/JMIGUEL-C)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.