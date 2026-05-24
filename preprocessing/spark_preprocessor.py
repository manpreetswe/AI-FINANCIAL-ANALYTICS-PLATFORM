# preprocessing/spark_preprocessor.py

import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    lag,
    when,
    stddev,
    lit,
    round,
    last,
    to_date,
    input_file_name,
    regexp_extract,
)
from pyspark.sql.window import Window


# ------------------------------------------------------------------
# Bullish Stocks
# ------------------------------------------------------------------
BULLISH_TICKERS = [

    # Big Tech
    "META",    # Meta
    "NVDA",    # Nvidia
    "NFLX",    # Netflix
    "ADBE",    # Adobe
    "CRM",     # Salesforce
    "ORCL"     # Oracle
]


# ------------------------------------------------------------------
# Medium / Stable Stocks
# ------------------------------------------------------------------
MEDIUM_TICKERS = [

    # Stable Tech
    "IBM",     # IBM
    "CSCO",    # Cisco
    "HPQ",     # HP

    # Banking
    "BAC",     # Bank of America
    "C",       # Citigroup

    # Telecom
    "VZ",      # Verizon
    "T",       # AT&T

    # Retail
    "EBAY",    # eBay
    "ETSY",    # Etsy

    # Travel
    "UAL",     # United Airlines

    # Consumer
    "SBUX",    # Starbucks
    "KO",      # Coca-Cola

    # Industrials
    "GE",      # General Electric
    "F"        # Ford
]


# ------------------------------------------------------------------
# Bearish Stocks
# ------------------------------------------------------------------
BEARISH_TICKERS = [

    # EV / Speculative
    "RIVN",    # Rivian
    "LCID",    # Lucid
    "NIO",     # NIO

    # Weak Social / Media
    "SNAP",    # Snap
    "PINS",    # Pinterest
    "WBD"      # Warner Bros Discovery
]


# ------------------------------------------------------------------
# Combined Tickers
# ------------------------------------------------------------------
TICKERS = (
    BULLISH_TICKERS
    + MEDIUM_TICKERS
    + BEARISH_TICKERS
)


# ------------------------------------------------------------------
# Extract Ticker Function
# ------------------------------------------------------------------
def extract_ticker(query):
    """
    Extract stock ticker from user query.
    """

    company_map = {

        # Bullish Stocks
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "netflix": "NFLX",
        "adobe": "ADBE",
        "salesforce": "CRM",
        "oracle": "ORCL",

        # Medium Stocks
        "ibm": "IBM",
        "cisco": "CSCO",
        "hp": "HPQ",
        "hewlett packard": "HPQ",

        "bank of america": "BAC",
        "bofa": "BAC",

        "citigroup": "C",
        "citi": "C",

        "verizon": "VZ",
        "at&t": "T",
        "att": "T",

        "ebay": "EBAY",
        "etsy": "ETSY",

        "united airlines": "UAL",

        "starbucks": "SBUX",
        "coca cola": "KO",
        "coke": "KO",

        "general electric": "GE",
        "ford": "F",

        # Bearish Stocks
        "rivian": "RIVN",
        "lucid": "LCID",
        "nio": "NIO",

        "snap": "SNAP",
        "snapchat": "SNAP",

        "pinterest": "PINS",

        "warner bros": "WBD",
        "warner bros discovery": "WBD",
    }

    query_lower = query.lower()

    # --------------------------------------------------------------
    # Check company names
    # --------------------------------------------------------------
    for company, ticker in company_map.items():
        if company in query_lower:
            return ticker

    # --------------------------------------------------------------
    # Extract ticker symbols
    # --------------------------------------------------------------
    matches = re.findall(r'\b[A-Z]{1,5}\b', query.upper())

    invalid_words = {
        "SHOW",
        "PRICE",
        "OPEN",
        "CLOSE",
        "DATA",
        "NEXT",
        "DAYS",
        "STOCK",
        "PREDICT",
        "WHAT",
        "TODAY",
        "TOMORROW",
    }

    valid_tickers = set(BULLISH_TICKERS + BEARISH_TICKERS)

    for match in matches:

        if (
            match not in invalid_words
            and match in valid_tickers
        ):
            return match

    return None


# ------------------------------------------------------------------
# Create Spark Session
# ------------------------------------------------------------------
def create_spark_session():
    return (
        SparkSession.builder
        .appName("StockDataPreprocessing")
        .getOrCreate()
    )


# ------------------------------------------------------------------
# Load CSV Data
# ------------------------------------------------------------------
def load_csv_data(spark, input_path):
    """
    Load CSV stock data into Spark DataFrame.
    """

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(input_path)
    )

    # Convert Date column to proper date type
    df = df.withColumn("Date", to_date(col("Date")))

    # Extract ticker name from filename
    # Example:
    # AAPL_stock_data.csv -> AAPL
    df = df.withColumn(
        "Ticker",
        regexp_extract(
            input_file_name(),
            r'([^/]+)_stock_data\.csv',
            1
        )
    )

    return df


# ------------------------------------------------------------------
# Feature Engineering & Preprocessing
# ------------------------------------------------------------------
def preprocess_stock_data(df):
    """
    Perform feature engineering and preprocessing.
    """

    # Window specification
    window_spec = Window.partitionBy("Ticker").orderBy("Date")

    # Rolling windows
    window_7 = window_spec.rowsBetween(-6, 0)
    window_30 = window_spec.rowsBetween(-29, 0)
    window_90 = window_spec.rowsBetween(-89, 0)
    window_14 = window_spec.rowsBetween(-13, 0)

    # --------------------------------------------------------------
    # Daily Return
    # --------------------------------------------------------------
    df = df.withColumn(
        "Prev_Close",
        lag("Close").over(window_spec)
    )

    df = df.withColumn(
        "Daily_Return",
        when(
            col("Prev_Close").isNotNull(),
            (col("Close") - col("Prev_Close")) / col("Prev_Close")
        ).otherwise(lit(0.0))
    )

    # --------------------------------------------------------------
    # Moving Averages
    # --------------------------------------------------------------
    df = df.withColumn(
        "MA_7",
        avg("Close").over(window_7)
    )

    df = df.withColumn(
        "MA_30",
        avg("Close").over(window_30)
    )

    df = df.withColumn(
        "MA_90",
        avg("Close").over(window_90)
    )

    # --------------------------------------------------------------
    # RSI (14-Day)
    # RSI = 100 - (100 / (1 + RS))
    # --------------------------------------------------------------
    df = df.withColumn(
        "Price_Change",
        col("Close") - col("Prev_Close")
    )

    df = df.withColumn(
        "Gain",
        when(col("Price_Change") > 0, col("Price_Change")).otherwise(0)
    )

    df = df.withColumn(
        "Loss",
        when(col("Price_Change") < 0, -col("Price_Change")).otherwise(0)
    )

    df = df.withColumn(
        "Avg_Gain",
        avg("Gain").over(window_14)
    )

    df = df.withColumn(
        "Avg_Loss",
        avg("Loss").over(window_14)
    )

    df = df.withColumn(
        "RS",
        when(
            col("Avg_Loss") != 0,
            col("Avg_Gain") / col("Avg_Loss")
        ).otherwise(lit(None))
    )

    df = df.withColumn(
        "RSI",
        when(
            col("RS").isNotNull(),
            100 - (100 / (1 + col("RS")))
        ).otherwise(lit(None))
    )

    # --------------------------------------------------------------
    # Volatility (30-day rolling std dev)
    # --------------------------------------------------------------
    df = df.withColumn(
        "Volatility",
        stddev("Daily_Return").over(window_30)
    )

    # --------------------------------------------------------------
    # Sharpe Ratio
    # --------------------------------------------------------------
    df = df.withColumn(
        "Mean_Return",
        avg("Daily_Return").over(window_30)
    )

    df = df.withColumn(
        "Std_Return",
        stddev("Daily_Return").over(window_30)
    )

    df = df.withColumn(
        "Sharpe_Ratio",
        when(
            col("Std_Return") != 0,
            col("Mean_Return") / col("Std_Return")
        ).otherwise(lit(0.0))
    )

    # --------------------------------------------------------------
    # Handle Missing Values (Forward Fill)
    # --------------------------------------------------------------
    fill_window = (
        Window.partitionBy("Ticker")
        .orderBy("Date")
        .rowsBetween(Window.unboundedPreceding, 0)
    )

    feature_cols = [
        "MA_7",
        "MA_30",
        "MA_90",
        "RSI",
        "Volatility",
        "Daily_Return",
        "Sharpe_Ratio"
    ]

    for feature in feature_cols:
        df = df.withColumn(
            feature,
            last(feature, ignorenulls=True).over(fill_window)
        )

    # --------------------------------------------------------------
    # Drop Helper Columns
    # --------------------------------------------------------------
    drop_cols = [
        "Prev_Close",
        "Price_Change",
        "Gain",
        "Loss",
        "Avg_Gain",
        "Avg_Loss",
        "RS",
        "Mean_Return",
        "Std_Return",
    ]

    df = df.drop(*drop_cols)

    # --------------------------------------------------------------
    # Round Numerical Features
    # --------------------------------------------------------------
    numeric_cols = [
        "MA_7",
        "MA_30",
        "MA_90",
        "RSI",
        "Volatility",
        "Daily_Return",
        "Sharpe_Ratio"
    ]

    for c in numeric_cols:
        df = df.withColumn(c, round(col(c), 4))

    # Sort final dataframe
    df = df.orderBy("Ticker", "Date")

    return df


# ------------------------------------------------------------------
# Save Data as Parquet
# ------------------------------------------------------------------
def save_as_parquet(df, output_path):

    (
        df.write
        .mode("overwrite")
        .parquet(output_path)
    )


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
def main():

    spark = create_spark_session()

    # Reduce Spark log noise
    spark.sparkContext.setLogLevel("ERROR")

    # Input CSV files
    input_path = "data/stock_data/*.csv"

    # Output parquet path
    output_path = "data/processed_stocks.parquet"

    # Load stock data
    df = load_csv_data(spark, input_path)

    # Show generated tickers
    print("\nGenerated Tickers:")
    df.select("Ticker").distinct().show(truncate=False)

    # Preprocess and engineer features
    processed_df = preprocess_stock_data(df)

    # Save processed data
    save_as_parquet(processed_df, output_path)

    print("\nPreprocessing completed successfully!")
    print(f"Saved processed data to: {output_path}")

    processed_df.show(5)

    spark.stop()


if __name__ == "__main__":
    main()