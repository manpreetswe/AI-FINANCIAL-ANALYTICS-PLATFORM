 #============================================================
# FILE: ml_models/investment_classifier.py
# ============================================================

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


class InvestmentClassifier:

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state
        )

        self.label_encoder = LabelEncoder()

    # =========================================================
    # RSI
    # =========================================================

    @staticmethod
    def calculate_rsi(prices, period=14):

        delta = prices.diff()

        gain = delta.where(delta > 0, 0)

        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()

        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / (avg_loss + 1e-9)

        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    # =========================================================
    # SHARPE RATIO
    # =========================================================

    @staticmethod
    def calculate_sharpe_ratio(
        returns,
        risk_free_rate=0.01
    ):

        excess_returns = (
            returns - (risk_free_rate / 252)
        )

        std = excess_returns.std()

        if std == 0 or np.isnan(std):
            return 0

        sharpe_ratio = np.sqrt(252) * (
            excess_returns.mean() / std
        )

        return sharpe_ratio

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================

    def engineer_features(self, stock_data):

        feature_rows = []

        for ticker, df in stock_data.items():

            df = df.copy()

            df.sort_values(
                "Date",
                inplace=True
            )

            close_prices = df["Close"]

            daily_returns = (
                close_prices.pct_change().dropna()
            )

            # =================================================
            # MOVING AVERAGES
            # =================================================

            ma7 = close_prices.rolling(7).mean()

            ma30 = close_prices.rolling(30).mean()

            ma90 = close_prices.rolling(90).mean()

            # =================================================
            # RSI
            # =================================================

            rsi = self.calculate_rsi(
                close_prices
            )

            # =================================================
            # 17 FEATURES
            # =================================================

            total_return = (
                (
                    close_prices.iloc[-1]
                    - close_prices.iloc[0]
                )
                / close_prices.iloc[0]
            ) * 100

            recent_return_7 = (
                (
                    close_prices.iloc[-1]
                    - close_prices.iloc[-7]
                )
                / close_prices.iloc[-7]
            ) * 100 if len(close_prices) >= 7 else 0

            recent_return_30 = (
                (
                    close_prices.iloc[-1]
                    - close_prices.iloc[-30]
                )
                / close_prices.iloc[-30]
            ) * 100 if len(close_prices) >= 30 else 0

            average_rsi = rsi.mean()

            current_rsi = rsi.iloc[-1]

            volatility = (
                daily_returns.std()
                * np.sqrt(252)
            )

            sharpe_ratio = (
                self.calculate_sharpe_ratio(
                    daily_returns
                )
            )

            ma7_vs_ma30 = (
                1
                if ma7.iloc[-1] > ma30.iloc[-1]
                else 0
            )

            ma30_vs_ma90 = (
                1
                if ma30.iloc[-1] > ma90.iloc[-1]
                else 0
            )

            avg_daily_return = (
                daily_returns.mean()
            )

            max_return = (
                daily_returns.max()
            )

            min_return = (
                daily_returns.min()
            )

            median_return = (
                daily_returns.median()
            )

            return_std = (
                daily_returns.std()
            )

            price_range = (
                close_prices.max()
                - close_prices.min()
            )

            price_mean = (
                close_prices.mean()
            )

            momentum = (
                close_prices.iloc[-1]
                - close_prices.iloc[-10]
            ) if len(close_prices) >= 10 else 0

            # =================================================
            # TREND SCORE
            # =================================================

            trend_score = 0

            if ma7.iloc[-1] > ma30.iloc[-1]:
                trend_score += 5

            if ma30.iloc[-1] > ma90.iloc[-1]:
                trend_score += 5

            # =================================================
            # RSI SCORE
            # =================================================

            if 45 <= current_rsi <= 60:

                rsi_score = 10

            elif 35 <= current_rsi < 45:

                rsi_score = 7

            elif 60 < current_rsi <= 75:

                rsi_score = 6

            else:

                rsi_score = 2

            # =================================================
            # VOLATILITY SCORE
            # =================================================

            if volatility < 0.15:

                volatility_score = 10

            elif volatility < 0.30:

                volatility_score = 7

            elif volatility < 0.50:

                volatility_score = 5

            else:

                volatility_score = 2

            # =================================================
            # SHARPE SCORE
            # =================================================

            if sharpe_ratio > 1.5:

                sharpe_score = 10

            elif sharpe_ratio > 0.8:

                sharpe_score = 7

            elif sharpe_ratio > 0:

                sharpe_score = 5

            else:

                sharpe_score = 2

            # =================================================
            # RETURN SCORE
            # =================================================

            if total_return > 80:

                return_score = 10

            elif total_return > 40:

                return_score = 8

            elif total_return > 10:

                return_score = 6

            elif total_return > 0:

                return_score = 4

            else:

                return_score = 1

            # =================================================
            # COMPOSITE SCORE
            # =================================================

            composite_score = (
                (return_score * 0.30)
                + (trend_score * 0.20)
                + (rsi_score * 0.15)
                + (volatility_score * 0.15)
                + (sharpe_score * 0.20)
            )

            composite_score = round(
                composite_score,
                2
            )

            # =================================================
            # LABELS
            # =================================================

            if composite_score >= 7:

                label = "High"

            elif composite_score >= 4:

                label = "Medium"

            else:

                label = "Low"

            # =================================================
            # STORE FEATURES
            # =================================================

            feature_rows.append({

                "Ticker": ticker,

                "Total_Return": total_return,

                "Recent_Return_7": recent_return_7,

                "Recent_Return_30": recent_return_30,

                "Average_RSI": average_rsi,

                "Current_RSI": current_rsi,

                "Volatility": volatility,

                "Sharpe_Ratio": sharpe_ratio,

                "MA7_vs_MA30": ma7_vs_ma30,

                "MA30_vs_MA90": ma30_vs_ma90,

                "Avg_Daily_Return": avg_daily_return,

                "Max_Return": max_return,

                "Min_Return": min_return,

                "Median_Return": median_return,

                "Return_STD": return_std,

                "Price_Range": price_range,

                "Price_Mean": price_mean,

                "Momentum": momentum,

                "Score": composite_score,

                "Label": label
            })

        return pd.DataFrame(feature_rows)

    # =========================================================
    # TRAIN MODEL
    # =========================================================

    def train_classifier(self, features_df):

        feature_columns = [

            "Total_Return",

            "Recent_Return_7",

            "Recent_Return_30",

            "Average_RSI",

            "Current_RSI",

            "Volatility",

            "Sharpe_Ratio",

            "MA7_vs_MA30",

            "MA30_vs_MA90",

            "Avg_Daily_Return",

            "Max_Return",

            "Min_Return",

            "Median_Return",

            "Return_STD",

            "Price_Range",

            "Price_Mean",

            "Momentum"
        ]

        X = features_df[
            feature_columns
        ]

        y = features_df["Label"]

        y_encoded = (
            self.label_encoder.fit_transform(y)
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y_encoded,
                test_size=0.3,
                random_state=self.random_state,
                stratify=y_encoded
            )
        )

        # =====================================================
        # TRAIN
        # =====================================================

        self.model.fit(
            X_train,
            y_train
        )

        predictions = self.model.predict(
            X_test
        )

        # =====================================================
        # METRICS
        # =====================================================

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        # =====================================================
        # DISPLAY METRICS
        # =====================================================

        print("\nModel Evaluation Metrics")

        print("=" * 40)

        print(f"Accuracy : {accuracy:.4f}")

        print(f"Precision: {precision:.4f}")

        print(f"Recall   : {recall:.4f}")

        print(f"F1-Score : {f1:.4f}")

        # =====================================================
        # CLASSIFICATION REPORT
        # =====================================================

        decoded_actual = (
            self.label_encoder.inverse_transform(
                y_test
            )
        )

        decoded_pred = (
            self.label_encoder.inverse_transform(
                predictions
            )
        )

        print("\nClassification Report")

        print("=" * 40)

        print(
            classification_report(
                decoded_actual,
                decoded_pred,
                zero_division=0
            )
        )

    # =========================================================
    # DISPLAY RESULTS
    # =========================================================

    def display_results(self, features_df):

        print("\nClassification Results:")

        print("=" * 40)

        for _, row in features_df.iterrows():

            print(
                f"{row['Ticker']} "
                f"=> {row['Label']} "
                f"(Score: "
                f"{row['Score']})"
            )


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    np.random.seed(42)

    # =========================================================
    # STRONG / BULLISH STOCKS
    # =========================================================

    BULLISH_TICKERS = [

        # Big Tech
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "NFLX",
        "ADBE",
        "CRM",
        "ORCL"
    ]

    # =========================================================
    # MEDIUM / STABLE STOCKS
    # =========================================================

    MEDIUM_TICKERS = [

        "IBM",
        "CSCO",
        "HPQ",

        "BAC",
        "C",

        "VZ",
        "T",

        "EBAY",
        "ETSY",

        "UAL",

        "SBUX",
        "KO",

        "GE",
        "F"
    ]

    # =========================================================
    # BEARISH STOCKS
    # =========================================================

    BEARISH_TICKERS = [

        "RIVN",
        "LCID",
        "NIO",

        "SNAP",
        "PINS",
        "PARA",
        "WBD"
    ]

    # =========================================================
    # COMBINED TICKERS
    # =========================================================

    TICKERS = (
        BULLISH_TICKERS
        + MEDIUM_TICKERS
        + BEARISH_TICKERS
    )

    stock_data = {}

    # =========================================================
    # GENERATE STOCK DATA
    # =========================================================

    for index, ticker in enumerate(TICKERS):

        dates = pd.date_range(
            start="2024-01-01",
            periods=180
        )

        # =====================================================
        # BULLISH
        # =====================================================

        if ticker in BULLISH_TICKERS:

            prices = np.cumsum(

                np.random.normal(
                    loc=2.0,
                    scale=1.5,
                    size=180
                )

            ) + 100

        # =====================================================
        # MEDIUM
        # =====================================================

        elif ticker in MEDIUM_TICKERS:

            prices = np.cumsum(

                np.random.normal(
                    loc=0.4,
                    scale=3.0,
                    size=180
                )

            ) + 100

        # =====================================================
        # BEARISH
        # =====================================================

        else:

            prices = np.cumsum(

                np.random.normal(
                    loc=-1.2,
                    scale=4.5,
                    size=180
                )

            ) + 100

        prices = np.maximum(
            prices,
            5
        )

        stock_data[ticker] = pd.DataFrame({

            "Date": dates,

            "Close": prices
        })

    # =========================================================
    # RUN CLASSIFIER
    # =========================================================

    classifier = InvestmentClassifier()

    features_df = (
        classifier.engineer_features(
            stock_data
        )
    )

    classifier.train_classifier(
        features_df
    )

    classifier.display_results(
        features_df
    )