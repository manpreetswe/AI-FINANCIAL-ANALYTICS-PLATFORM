# File: data_collection/stock_downloader.py

import os
from datetime import datetime
import yfinance as yf
import pandas as pd

# Create output directory
OUTPUT_DIR = "data/stock_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STRONG / BULLISH STOCKS
# ============================================================
BULLISH_TICKERS = [

    # Big Tech
    "META",    # Meta
    "NVDA",    # Nvidia
    "NFLX",    # Netflix
    "ADBE",    # Adobe
    "CRM",     # Salesforce
    "ORCL"] # Oracle
#============================================================
# MEDIUM / STABLE STOCKS
# ============================================================

MEDIUM_TICKERS = [

    # Stable Tech
    "IBM",
    "CSCO",
    "HPQ",

    # Banking
    "BAC",
    "C",

    # Telecom
    "VZ",
    "T",

    # Retail
    "EBAY",
    "ETSY",

    # Travel
    "UAL",

    # Consumer
    "SBUX",
    "KO",

    # Industrials
    "GE",
    "F"
]

BEARISH_TICKERS = [

    # EV / Speculative
    "RIVN",    # Rivian
    "LCID",    # Lucid
    "NIO",     # NIO

    # Weak Social / Media
    "SNAP",    # Snap
    "PINS",    # Pinterest
    "WBD"]  # Warner Bros Discovery
# ============================================================
# COMBINED TICKERS
# ============================================================

TICKERS = (
    BULLISH_TICKERS
    + MEDIUM_TICKERS
    + BEARISH_TICKERS
)
# Date range
START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def download_stock_data(ticker):
    """
    Download historical stock data from Yahoo Finance
    and save it as a CSV file.
    """

    try:
        print(f"Downloading data for {ticker}...")

        # Download stock data
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False
        )

        # Check if dataframe is empty
        if df.empty:
            print(f"No data found for {ticker}")
            return

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        # Fix multi-level column issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only required columns
        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        df = df[required_columns]

        # Save CSV
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{ticker}_stock_data.csv"
        )

        df.to_csv(output_path, index=False)

        print(f"Saved: {output_path} ({len(df)} rows)")

    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")


def main():
    """
    Download stock data for all tickers.
    """

    for ticker in TICKERS:
        download_stock_data(ticker)

    print("\nData collection complete.")


if __name__ == "__main__":
    main()