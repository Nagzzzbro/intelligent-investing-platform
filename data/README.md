# 📊 Dataset Setup Guide

This directory stores the raw stock market data used by the **NIFTY-50 Investment Intelligence Platform**.

> **Note:** The application ships with built-in demo data and will work out of the box *without* downloading these datasets. Download them only if you want to train models on the full historical data.

---

## Datasets Overview

| # | Dataset | Source | Period | Description |
|---|---------|--------|--------|-------------|
| 1 | **NIFTY-50 Stock Market Data** | [Kaggle – Rohan Rao](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data) | Jan 2000 – Apr 2021 | Individual CSV files for each NIFTY-50 constituent stock |
| 2 | **India Stock Data (NSE)** | [Kaggle – StoicStatic](https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020) | 1990 – 2020 | Broader NSE stock data covering a wider universe of listed securities |

---

## Dataset 1: NIFTY-50 Stock Market Data (Primary)

### About

This is the **primary dataset** used for model training and analysis. It contains one CSV file per NIFTY-50 stock with daily trading data.

**Columns:**

| Column | Description |
|--------|-------------|
| `Date` | Trading date (YYYY-MM-DD) |
| `Symbol` | NSE stock ticker symbol (e.g., RELIANCE, TCS, INFY) |
| `Prev Close` | Previous day's closing price |
| `Open` | Opening price |
| `High` | Intraday high price |
| `Low` | Intraday low price |
| `Last` | Last traded price |
| `Close` | Closing price |
| `VWAP` | Volume-Weighted Average Price |
| `Volume` | Total traded volume (number of shares) |
| `Turnover` | Total turnover in ₹ |
| `Trades` | Number of trades executed |
| `Deliverable Volume` | Volume delivered (settled) |
| `%Deliverble` | Percentage of deliverable volume to total volume |

**Date Range:** January 1, 2000 – April 30, 2021

### Download — Option A: Kaggle CLI (Recommended)

1. **Install the Kaggle CLI** (if not already installed):

   ```bash
   pip install kaggle
   ```

2. **Set up API credentials:**
   - Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
   - This downloads a `kaggle.json` file
   - Place it in the appropriate location:
     - **Linux/macOS:** `~/.kaggle/kaggle.json`
     - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Set permissions (Linux/macOS only):
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download and extract the dataset:**

   ```bash
   # Navigate to the data directory
   cd data/

   # Download the dataset
   kaggle datasets download -d rohanrao/nifty50-stock-market-data

   # Extract the ZIP file
   unzip nifty50-stock-market-data.zip -d nifty50-stock-market-data/

   # Clean up the ZIP
   rm nifty50-stock-market-data.zip
   ```

### Download — Option B: Manual Download

1. Visit: [https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
2. Sign in to your Kaggle account (create one if needed)
3. Click the **Download** button (top-right)
4. Extract the downloaded ZIP file into this `data/` directory
5. Ensure the CSV files are inside `data/nifty50-stock-market-data/`

---

## Dataset 2: India Stock Data (NSE) — Supplementary

### About

A broader dataset covering NSE-listed stocks from 1990–2020. Used for supplementary analysis, additional backtesting, and cross-validation.

### Download — Option A: Kaggle CLI

```bash
# Navigate to the data directory
cd data/

# Download the dataset
kaggle datasets download -d stoicstatic/india-stock-data-nse-1990-2020

# Extract the ZIP file
unzip india-stock-data-nse-1990-2020.zip -d india-stock-data-nse/

# Clean up
rm india-stock-data-nse-1990-2020.zip
```

### Download — Option B: Manual Download

1. Visit: [https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020](https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020)
2. Sign in and click **Download**
3. Extract the ZIP into `data/india-stock-data-nse/`

---

## Expected Directory Structure

After downloading and extracting both datasets, your `data/` directory should look like this:

```
data/
├── README.md                          ← You are here
│
├── nifty50-stock-market-data/         ← Primary dataset
│   ├── ADANIPORTS.csv
│   ├── ASIANPAINT.csv
│   ├── AXISBANK.csv
│   ├── BAJAJ-AUTO.csv
│   ├── BAJAJFINSV.csv
│   ├── BAJFINANCE.csv
│   ├── BHARTIARTL.csv
│   ├── BPCL.csv
│   ├── BRITANNIA.csv
│   ├── CIPLA.csv
│   ├── COALINDIA.csv
│   ├── DIVISLAB.csv
│   ├── DRREDDY.csv
│   ├── EICHERMOT.csv
│   ├── GRASIM.csv
│   ├── HCLTECH.csv
│   ├── HDFC.csv
│   ├── HDFCBANK.csv
│   ├── HDFCLIFE.csv
│   ├── HEROMOTOCO.csv
│   ├── HINDALCO.csv
│   ├── HINDUNILVR.csv
│   ├── ICICIBANK.csv
│   ├── INDUSINDBK.csv
│   ├── INFY.csv
│   ├── IOC.csv
│   ├── ITC.csv
│   ├── JSWSTEEL.csv
│   ├── KOTAKBANK.csv
│   ├── LT.csv
│   ├── M&M.csv
│   ├── MARUTI.csv
│   ├── NESTLEIND.csv
│   ├── NTPC.csv
│   ├── ONGC.csv
│   ├── POWERGRID.csv
│   ├── RELIANCE.csv
│   ├── SBILIFE.csv
│   ├── SBIN.csv
│   ├── SHREECEM.csv
│   ├── SUNPHARMA.csv
│   ├── TATACONSUM.csv
│   ├── TATAMOTORS.csv
│   ├── TATASTEEL.csv
│   ├── TCS.csv
│   ├── TECHM.csv
│   ├── TITAN.csv
│   ├── ULTRACEMCO.csv
│   ├── UPL.csv
│   └── WIPRO.csv
│
└── india-stock-data-nse/              ← Supplementary dataset
    └── ...                            (broader NSE stock files)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `kaggle: command not found` | Run `pip install kaggle` and ensure your Python Scripts folder is in your PATH |
| `403 - Forbidden` | Accept the dataset's terms on the Kaggle webpage, then retry the CLI download |
| `kaggle.json not found` | Generate an API token from [Kaggle Settings](https://www.kaggle.com/settings) and place it in `~/.kaggle/` |
| CSV files not loading in the app | Verify the CSVs are inside `data/nifty50-stock-market-data/`, not nested in an extra subdirectory |

---

## Using Demo Data

If you prefer not to download the full datasets, the platform includes a **demo data generator** that creates synthetic stock data for testing purposes. Simply run the application without any datasets in this directory, and it will automatically use the built-in demo data.

```bash
# Run the app directly — demo data is used if no datasets are found
streamlit run app.py
```

---

*For more information, see the [main project README](../README.md).*
