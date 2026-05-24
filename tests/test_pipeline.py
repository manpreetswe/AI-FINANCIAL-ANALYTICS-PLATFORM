# ============================================================
# FILE: tests/test_pipeline.py
# ============================================================

import os
import sys
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ============================================================
# ADD PROJECT ROOT TO PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from data_collection.stock_downloader import TICKERS
from preprocessing.spark_preprocessor import create_spark_session
from sql_interface.database_manager import DatabaseManager


# ============================================================
# SAMPLE TEST DATA
# ============================================================

@pytest.fixture
def sample_stock_dataframe():
    """
    Create sample stock dataframe for testing.
    """

    data = {
        "Date": pd.date_range(start="2024-01-01", periods=10),
        "Open": np.random.uniform(100, 200, 10),
        "High": np.random.uniform(200, 300, 10),
        "Low": np.random.uniform(90, 150, 10),
        "Close": np.random.uniform(100, 250, 10),
        "Volume": np.random.randint(1000000, 5000000, 10),
        "Ticker": ["AAPL"] * 10,
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def spark_session():
    """
    Create Spark session for testing.
    """

    spark = create_spark_session()

    yield spark

    spark.stop()


# ============================================================
# TASK 8.1 - DATA COLLECTION TESTS
# ============================================================

class TestDataCollection:
    """
    Test stock data collection functionality.
    """

    def test_tickers_exist(self):
        """
        Test ticker list is not empty.
        """

        assert len(TICKERS) > 0
        assert isinstance(TICKERS, list)

    def test_expected_tickers_present(self):
        """
        Test expected stock tickers exist.
        """

        expected = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        for ticker in expected:
            assert ticker in TICKERS

    def test_dataframe_structure(self, sample_stock_dataframe):
        """
        Test dataframe contains required columns.
        """

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Ticker",
        ]

        for col in required_columns:
            assert col in sample_stock_dataframe.columns

    def test_dataframe_not_empty(self, sample_stock_dataframe):
        """
        Test dataframe is not empty.
        """

        assert len(sample_stock_dataframe) > 0

    def test_no_null_ticker_values(self, sample_stock_dataframe):
        """
        Test ticker column has no null values.
        """

        assert sample_stock_dataframe["Ticker"].isnull().sum() == 0


# ============================================================
# TASK 8.2 - FEATURE ENGINEERING TESTS
# ============================================================

class TestFeatureEngineering:
    """
    Test feature engineering and preprocessing.
    """

    def test_spark_session_creation(self, spark_session):
        """
        Test Spark session initializes correctly.
        """

        assert spark_session is not None
        assert spark_session.sparkContext.appName == "StockDataPreprocessing"

    def test_spark_dataframe_creation(self, spark_session, sample_stock_dataframe):
        """
        Test conversion to Spark dataframe.
        """

        spark_df = spark_session.createDataFrame(sample_stock_dataframe)

        assert spark_df.count() == len(sample_stock_dataframe)
        assert len(spark_df.columns) == len(sample_stock_dataframe.columns)

    def test_moving_average_feature(self, sample_stock_dataframe):
        """
        Test moving average calculation.
        """

        df = sample_stock_dataframe.copy()

        df["MA_3"] = df["Close"].rolling(window=3).mean()

        assert "MA_3" in df.columns

        valid_values = df["MA_3"].dropna()

        assert len(valid_values) > 0

    def test_daily_return_feature(self, sample_stock_dataframe):
        """
        Test daily return calculation.
        """

        df = sample_stock_dataframe.copy()

        df["Daily_Return"] = df["Close"].pct_change()

        assert "Daily_Return" in df.columns

    def test_volatility_feature(self, sample_stock_dataframe):
        """
        Test rolling volatility feature.
        """

        df = sample_stock_dataframe.copy()

        df["Volatility"] = (
            df["Close"]
            .pct_change()
            .rolling(window=3)
            .std()
        )

        assert "Volatility" in df.columns

    def test_no_duplicate_rows(self, sample_stock_dataframe):
        """
        Test duplicate removal.
        """

        duplicates = sample_stock_dataframe.duplicated().sum()

        assert duplicates == 0


# ============================================================
# TASK 8.3 - DATABASE TESTS
# ============================================================

class TestDatabaseOperations:
    """
    Test SQLite database operations.
    """

    def test_database_creation(self):
        """
        Test SQLite database file creation.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_financial.db")
            parquet_path = os.path.join(temp_dir, "test.parquet")

            db_manager = DatabaseManager(
                db_path=db_path,
                parquet_path=parquet_path,
            )

            assert os.path.exists(db_path)
            assert db_manager is not None

    def test_database_connection(self):
        """
        Test database connection.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_financial.db")

            conn = sqlite3.connect(db_path)

            assert conn is not None

            conn.close()

    def test_table_creation(self):
        """
        Test table creation in SQLite.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_financial.db")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    close REAL
                )
                """
            )

            conn.commit()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'"
            )

            result = cursor.fetchone()

            assert result is not None
            assert result[0] == "stocks"

            conn.close()

    def test_insert_and_fetch_data(self):
        """
        Test insert and fetch operations.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_financial.db")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE stocks (
                    ticker TEXT,
                    close REAL
                )
                """
            )

            cursor.execute(
                "INSERT INTO stocks VALUES (?, ?)",
                ("AAPL", 200.50),
            )

            conn.commit()

            cursor.execute("SELECT * FROM stocks")

            rows = cursor.fetchall()

            assert len(rows) == 1
            assert rows[0][0] == "AAPL"

            conn.close()


# ============================================================
# TASK 8.4 - ML MODEL TESTS
# ============================================================

class TestMachineLearning:
    """
    Test machine learning model functionality.
    """

    def test_train_test_split(self, sample_stock_dataframe):
        """
        Test train/test split logic.
        """

        from sklearn.model_selection import train_test_split

        X = sample_stock_dataframe[["Open", "High", "Low", "Volume"]]
        y = sample_stock_dataframe["Close"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0

    def test_linear_regression_prediction(self, sample_stock_dataframe):
        """
        Test regression model predictions.
        """

        from sklearn.linear_model import LinearRegression

        X = sample_stock_dataframe[["Open", "High", "Low", "Volume"]]
        y = sample_stock_dataframe["Close"]

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert isinstance(predictions[0], np.float64)

    def test_prediction_values_valid(self, sample_stock_dataframe):
        """
        Test predictions contain valid numeric values.
        """

        from sklearn.ensemble import RandomForestRegressor

        X = sample_stock_dataframe[["Open", "High", "Low", "Volume"]]
        y = sample_stock_dataframe["Close"]

        model = RandomForestRegressor(
            n_estimators=10,
            random_state=42,
        )

        model.fit(X, y)

        predictions = model.predict(X)

        assert np.isnan(predictions).sum() == 0
        assert np.isfinite(predictions).all()

    def test_model_score_range(self, sample_stock_dataframe):
        """
        Test model score range.
        """

        from sklearn.linear_model import LinearRegression

        X = sample_stock_dataframe[["Open", "High", "Low", "Volume"]]
        y = sample_stock_dataframe["Close"]

        model = LinearRegression()
        model.fit(X, y)

        score = model.score(X, y)

        assert -1 <= score <= 1


# ============================================================
# TASK 8.5 - INTEGRATION TEST
# ============================================================

class TestFullPipeline:
    """
    Integration test for full AI pipeline.
    """

    def test_complete_pipeline(
        self,
        spark_session,
        sample_stock_dataframe,
    ):
        """
        Test end-to-end stock prediction pipeline.
        """

        # ----------------------------------------------------
        # STEP 1 - RAW DATA VALIDATION
        # ----------------------------------------------------

        assert not sample_stock_dataframe.empty

        # ----------------------------------------------------
        # STEP 2 - FEATURE ENGINEERING
        # ----------------------------------------------------

        df = sample_stock_dataframe.copy()

        df["MA_3"] = df["Close"].rolling(window=3).mean()
        df["Daily_Return"] = df["Close"].pct_change()

        df = df.dropna()

        assert "MA_3" in df.columns
        assert "Daily_Return" in df.columns

        # ----------------------------------------------------
        # STEP 3 - SPARK CONVERSION
        # ----------------------------------------------------

        spark_df = spark_session.createDataFrame(df)

        assert spark_df.count() > 0

        # ----------------------------------------------------
        # STEP 4 - MODEL TRAINING
        # ----------------------------------------------------

        from sklearn.linear_model import LinearRegression

        feature_columns = [
            "Open",
            "High",
            "Low",
            "Volume",
            "MA_3",
        ]

        X = df[feature_columns]
        y = df["Close"]

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        # ----------------------------------------------------
        # STEP 5 - PREDICTION VALIDATION
        # ----------------------------------------------------

        assert len(predictions) == len(df)
        assert np.isfinite(predictions).all()

        # ----------------------------------------------------
        # STEP 6 - DATABASE STORAGE TEST
        # ----------------------------------------------------

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "pipeline_test.db")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE predictions (
                    prediction REAL
                )
                """
            )

            for pred in predictions:
                cursor.execute(
                    "INSERT INTO predictions VALUES (?)",
                    (float(pred),),
                )

            conn.commit()

            cursor.execute("SELECT COUNT(*) FROM predictions")

            count = cursor.fetchone()[0]

            assert count == len(predictions)

            conn.close()


# ============================================================
# PYTEST ENTRY CHECK
# ============================================================

if __name__ == "__main__":
    pytest.main(["-v"])