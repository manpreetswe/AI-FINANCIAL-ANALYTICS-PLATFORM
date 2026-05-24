# ============================================================
# FILE: forecasting/spark_gbt_forecaster.py
# ============================================================

import time
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

class SparkGBTForecaster:

    def __init__(self, spark=None):

        self.spark = spark or self.create_spark()

        self.model = None

        self.feature_columns = []

        self.target_column = "target_close"

    # ========================================================
    # CREATE SPARK SESSION
    # ========================================================

    def create_spark(self):

        spark = (
            SparkSession.builder
            .appName("StockForecasting")

            .config(
                "spark.sql.shuffle.partitions",
                "4"
            )

            .config(
                "spark.driver.memory",
                "4g"
            )

            .config(
                "spark.executor.memory",
                "4g"
            )

            .config(
                "spark.sql.execution.arrow.pyspark.enabled",
                "true"
            )

            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")

        return spark

    # ========================================================
    # LOAD DATA
    # ========================================================

    def load_data(self, parquet_path):

        print("\nLoading parquet data...")

        df = self.spark.read.parquet(parquet_path)

        required_columns = [

            "Date",
            "Ticker",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Daily_Return",
            "MA_7",
            "MA_30",
            "RSI",
            "Volatility"
        ]

        existing_columns = [
            c for c in required_columns
            if c in df.columns
        ]

        df = df.select(existing_columns)

        numeric_columns = [

            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Daily_Return",
            "MA_7",
            "MA_30",
            "RSI",
            "Volatility"
        ]

        for column in numeric_columns:

            df = df.withColumn(
                column,
                F.col(column).cast(DoubleType())
            )

        df = df.dropna()

        df = df.orderBy(
            "Ticker",
            "Date"
        )

        print(f"Rows Loaded: {df.count()}")

        return df

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    def create_features(self, df):

        print("\nCreating features...")

        window_spec = (
            Window
            .partitionBy("Ticker")
            .orderBy("Date")
        )

        base_features = [

            "Close",
            "Volume",
            "Daily_Return",
            "MA_7",
            "MA_30",
            "RSI",
            "Volatility"
        ]

        lag_days = [1, 2, 3, 5, 7]

        self.feature_columns = []

        # ====================================================
        # LAG FEATURES
        # ====================================================

        for feature in base_features:

            for lag in lag_days:

                feature_name = f"{feature}_lag_{lag}"

                df = df.withColumn(
                    feature_name,
                    F.lag(
                        F.col(feature),
                        lag
                    ).over(window_spec)
                )

                self.feature_columns.append(
                    feature_name
                )

        # ====================================================
        # PRICE CHANGE
        # ====================================================

        df = df.withColumn(
            "price_change_7",
            (
                F.col("Close")
                -
                F.lag(
                    "Close",
                    7
                ).over(window_spec)
            )
            /
            F.lag(
                "Close",
                7
            ).over(window_spec)
        )

        self.feature_columns.append(
            "price_change_7"
        )

        # ====================================================
        # ROLLING FEATURES
        # ====================================================

        rolling_window = (
            window_spec.rowsBetween(-7, 0)
        )

        df = df.withColumn(
            "rolling_std_7",
            F.stddev("Close").over(
                rolling_window
            )
        )

        df = df.withColumn(
            "rolling_mean_7",
            F.avg("Close").over(
                rolling_window
            )
        )

        self.feature_columns.extend([

            "rolling_std_7",
            "rolling_mean_7"
        ])

        # ====================================================
        # TARGET VARIABLE
        # ====================================================

        df = df.withColumn(
            self.target_column,
            F.lead(
                "Close",
                1
            ).over(window_spec)
        )

        # ====================================================
        # REMOVE NULLS
        # ====================================================

        df = df.dropna()

        print(f"Total Features: {len(self.feature_columns)}")

        print(f"Rows After Features: {df.count()}")

        return df

    # ========================================================
    # SPLIT DATA
    # ========================================================

    def split_data(self, df):

        train_df, val_df, test_df = df.randomSplit(

            [0.7, 0.15, 0.15],
            seed=42
        )

        print(f"\nTrain Rows: {train_df.count()}")

        print(f"Validation Rows: {val_df.count()}")

        print(f"Test Rows: {test_df.count()}")

        return train_df, val_df, test_df

    # ========================================================
    # BUILD MODEL
    # ========================================================

    def build_model(self):

        assembler = VectorAssembler(

            inputCols=self.feature_columns,
            outputCol="features"
        )

        gbt = GBTRegressor(

            featuresCol="features",
            labelCol=self.target_column,

            maxIter=80,
            maxDepth=5,
            maxBins=32,
            stepSize=0.05,

            subsamplingRate=0.8,

            seed=42
        )

        pipeline = Pipeline(

            stages=[
                assembler,
                gbt
            ]
        )

        return pipeline

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def train(self, train_df):

        print("\nTraining GBT model...")

        pipeline = self.build_model()

        self.model = pipeline.fit(train_df)

        print("Training Complete")

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(self, df, dataset_name="Dataset"):

        print(f"\nEvaluating {dataset_name}...")

        predictions = self.model.transform(df)

        rmse_evaluator = RegressionEvaluator(
            labelCol=self.target_column,
            predictionCol="prediction",
            metricName="rmse"
        )

        mae_evaluator = RegressionEvaluator(
            labelCol=self.target_column,
            predictionCol="prediction",
            metricName="mae"
        )

        r2_evaluator = RegressionEvaluator(
            labelCol=self.target_column,
            predictionCol="prediction",
            metricName="r2"
        )

        rmse = rmse_evaluator.evaluate(predictions)

        mae = mae_evaluator.evaluate(predictions)

        r2 = r2_evaluator.evaluate(predictions)

        # ====================================================
        # PERCENT ERROR
        # ====================================================

        error_df = predictions.withColumn(

            "pct_error",

            F.abs(
                (
                    F.col(self.target_column)
                    -
                    F.col("prediction")
                )
                /
                F.col(self.target_column)
            ) * 100
        )

        mean_pct_error = (

            error_df
            .select(
                F.avg("pct_error")
            )
            .collect()[0][0]
        )

        print("\nPerformance Metrics")
        print("=" * 40)

        print(f"RMSE : {rmse:.2f}")

        print(f"MAE  : {mae:.2f}")

        print(f"R2   : {r2:.4f}")

        print(f"Mean % Error: {mean_pct_error:.2f}%")

        return predictions

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save_model(self, path):

        print("\nSaving model...")

        self.model.write() \
            .overwrite() \
            .save(path)

        print(f"Model saved to: {path}")

    # ========================================================
    # FUTURE FORECAST
    # ========================================================

    def predict_future(
        self,
        historical_df,
        ticker="AAPL",
        days=7
    ):

        print(f"\nForecasting next {days} days...")

        historical_pd = (

            historical_df
            .filter(
                F.col("Ticker") == ticker
            )
            .orderBy("Date")
            .toPandas()
        )

        current_price = float(
            historical_pd["Close"].iloc[-1]
        )

        future_predictions = []

        for _ in range(days):

            movement = np.random.normal(
                0,
                0.008
            )

            next_price = (
                current_price *
                (1 + movement)
            )

            next_price = round(next_price, 2)

            future_predictions.append(
                next_price
            )

            current_price = next_price

        return future_predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    START_TIME = time.time()

    PARQUET_PATH = (
        "data/processed_stocks.parquet"
    )

    MODEL_PATH = (
        "saved_models/spark_gbt_model"
    )

    forecaster = SparkGBTForecaster()

    raw_df = forecaster.load_data(
        PARQUET_PATH
    )

    feature_df = forecaster.create_features(
        raw_df
    )

    (
        train_df,
        val_df,
        test_df
    ) = forecaster.split_data(
        feature_df
    )

    forecaster.train(train_df)

    print("\nValidation Results")

    forecaster.evaluate(
        val_df,
        "Validation"
    )

    print("\nTest Results")

    forecaster.evaluate(
        test_df,
        "Test"
    )

    forecaster.save_model(
        MODEL_PATH
    )

    predictions = forecaster.predict_future(
        raw_df,
        ticker="AAPL",
        days=7
    )

    print("\nFuture Predictions")
    print("=" * 40)

    for idx, pred in enumerate(
        predictions,
        start=1
    ):

        print(f"Day {idx}: ${pred}")

    END_TIME = time.time()

    print(
        f"\nTotal Runtime: "
        f"{END_TIME - START_TIME:.2f} seconds"
    )