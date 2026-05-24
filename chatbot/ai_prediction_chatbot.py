# chatbot/ai_prediction_chatbot.py

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
from datetime import timedelta

import ollama

def main():
    # ============================================
    # CONFIG
    # ============================================

    DB_PATH = "data/financial_data.db"

    # Spark model disabled for now
    model = None


    # ============================================
    # DATABASE FUNCTIONS
    # ============================================

    def get_stock_data(ticker, days=30):
        """
        Fetch historical stock data from SQLite database
        """

        conn = sqlite3.connect(DB_PATH)

        query = """
            SELECT *
            FROM stock_data
            WHERE Ticker = ?
            ORDER BY Date DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(ticker.upper(), days)
        )

        conn.close()

        if not df.empty:
            df = df.sort_values("Date")

        return df


    # ============================================
    # ML PREDICTION FUNCTIONS
    # ============================================

    def get_prediction(ticker, num_days=7):
        """
        Generate future stock predictions
        """

        historical_df = get_stock_data(ticker, 30)

        if historical_df.empty:
            return None

        latest_close = historical_df["Close"].iloc[-1]

        predictions = []

        current_price = latest_close

        for i in range(num_days):

            predicted_price = current_price * (
                1 + np.random.normal(0.002, 0.01)
            )

            predictions.append({
                "day": i + 1,
                "predicted_close": round(predicted_price, 2)
            })

            current_price = predicted_price

        return predictions


    # ============================================
    # GRAPH GENERATION
    # ============================================

    def generate_prediction_chart(
        ticker,
        historical_df,
        predictions
    ):
        """
        Generate matplotlib chart
        """

        plt.figure(figsize=(10, 5))

        # Historical prices
        hist_dates = pd.to_datetime(historical_df["Date"])
        hist_prices = historical_df["Close"]

        plt.plot(
            hist_dates,
            hist_prices,
            label="Historical Prices"
        )

        # Prediction prices
        last_date = hist_dates.iloc[-1]

        future_dates = [
            last_date + timedelta(days=i + 1)
            for i in range(len(predictions))
        ]

        future_prices = [
            p["predicted_close"]
            for p in predictions
        ]

        plt.plot(
            future_dates,
            future_prices,
            linestyle="dashed",
            marker="o",
            label="Predicted Prices"
        )

        plt.title(f"{ticker.upper()} Stock Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()

        plt.savefig(buf, format="png")

        plt.close()

        buf.seek(0)

        image_base64 = base64.b64encode(
            buf.read()
        ).decode()

        return image_base64


    # ============================================
    # NLP / INTENT DETECTION
    # ============================================

    def detect_intent(user_query):
        """
        Detect user intent
        """

        query = user_query.lower()

        prediction_keywords = [
            "predict",
            "forecast",
            "future",
            "next",
            "tomorrow"
        ]

        data_keywords = [
            "show",
            "data",
            "price",
            "history",
            "tell me about"
        ]

        if any(word in query for word in prediction_keywords):
            return "prediction"

        elif any(word in query for word in data_keywords):
            return "data"

        return "chat"


    def extract_ticker(query):
        """
        Extract stock ticker from user query
        """

        company_map = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "rivian": "RIVN",
            "snap": "SNAP",
            "paypal": "PYPL",
            "intel": "INTC",
        }

        query_lower = query.lower()

        # Company names
        for company, ticker in company_map.items():
            if company in query_lower:
                return ticker

        # Ticker symbols
        matches = re.findall(
            r'\b[A-Z]{1,5}\b',
            query.upper()
        )

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
            "TELL",
            "ABOUT"
        }

        valid_tickers = {
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "RIVN",
            "SNAP",
            "PYPL",
            "INTC"
        }

        for match in matches:

            if (
                match not in invalid_words
                and match in valid_tickers
            ):
                return match

        return None


    def extract_days(query):
        """
        Extract number of days from query
        """

        match = re.search(
            r'(\d+)\s*days?',
            query.lower()
        )

        if match:
            return int(match.group(1))

        return 7


    # ============================================
    # OLLAMA LLM
    # ============================================

    def ask_llama(user_query):
        """
        Ask Ollama Llama 3.2
        """

        prompt = f"""
        You are a financial AI assistant.

        Answer clearly and concisely.

        User Question:
        {user_query}
        """

        try:

            response = ollama.chat(
                model='llama3.2',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )

            return response['message']['content']

        except Exception as e:

            return f"LLM Error: {str(e)}"


    # ============================================
    # STREAMLIT UI
    # ============================================

    st.set_page_config(
        page_title="AI Stock Prediction Chatbot",
        layout="wide"
    )

    st.title("📈 AI Stock Prediction Chatbot")

    st.write(
        """
        Ask questions like:

        - Predict AAPL next 7 days
        - Show TSLA data
        - What is RSI?
        """
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # ============================================
    # DISPLAY CHAT HISTORY
    # ============================================

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            if msg["type"] == "text":
                st.markdown(msg["content"])

            elif msg["type"] == "image":
                st.image(msg["content"])


    # ============================================
    # USER INPUT
    # ============================================

    user_input = st.chat_input(
        "Ask something about stocks..."
    )

    if user_input:

        print("USER INPUT:", user_input)

        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        intent = detect_intent(user_input)

        print("INTENT:", intent)

        ticker = extract_ticker(user_input)

        print("TICKER:", ticker)

        days = extract_days(user_input)

        # ========================================
        # PREDICTION
        # ========================================

        if intent == "prediction":

            with st.chat_message("assistant"):

                if not ticker:

                    response = (
                        "Please provide a stock ticker."
                    )

                    st.markdown(response)

                else:

                    historical_df = get_stock_data(
                        ticker,
                        30
                    )

                    if historical_df.empty:

                        response = (
                            f"No data found for {ticker}"
                        )

                        st.markdown(response)

                    else:

                        predictions = get_prediction(
                            ticker,
                            days
                        )

                        prediction_text = f"""
                        ### 📊 {ticker.upper()} Prediction

                        Predicted prices
                        for next {days} days:
                        """

                        for p in predictions:

                            prediction_text += (
                                f"\n- Day {p['day']}: "
                                f"${p['predicted_close']}"
                            )

                        st.markdown(prediction_text)

                        chart_base64 = (
                            generate_prediction_chart(
                                ticker,
                                historical_df,
                                predictions
                            )
                        )

                        image_bytes = (
                            base64.b64decode(chart_base64)
                        )

                        st.image(
                            image_bytes,
                            caption=(
                                f"{ticker.upper()} "
                                "Prediction Chart"
                            )
                        )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "text",
                            "content": prediction_text
                        })

                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "image",
                            "content": image_bytes
                        })

        # ========================================
        # DATA
        # ========================================

        elif intent == "data":

            with st.chat_message("assistant"):

                if not ticker:

                    response = (
                        "Please provide a stock ticker."
                    )

                    st.markdown(response)

                else:

                    df = get_stock_data(ticker, 10)

                    if df.empty:

                        response = (
                            f"No data found for {ticker}"
                        )

                        st.markdown(response)

                    else:

                        st.markdown(
                            f"## 📈 {ticker.upper()} Latest Data"
                        )

                        st.dataframe(df)

                        response = df.to_string(index=False)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "text",
                            "content": response
                        })

        # ========================================
        # GENERAL CHAT
        # ========================================

        else:

            with st.chat_message("assistant"):

                response = ask_llama(user_input)

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": response
                })


if __name__ == "__main__":

    main()