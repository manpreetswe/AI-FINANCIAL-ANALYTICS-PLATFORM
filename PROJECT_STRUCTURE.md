# Project Structure - Student Template

## 📁 Complete File Structure

```
project_template/
│
├── README.md                          ⭐ START HERE - Project overview and tasks
├── SETUP.md                           ⭐ Setup guide (Docker, Ollama, dependencies)
├── PROJECT_STRUCTURE.md               📄 This file
├── requirements.txt                   📦 Python dependencies
├── main.py                            🚀 Main pipeline orchestrator
│
├── config/
│   ├── __init__.py
│   └── config.py                      ⚙️ Configuration settings (paths, hyperparameters)
│
├── data_collection/
│   ├── __init__.py
│   ├── stock_downloader.py            📊 Task 1 (20 pts): Download stock data from Yahoo Finance
│   └── sec_downloader.py              📄 Optional: Download 10-K filings
│
├── preprocessing/
│   ├── __init__.py
│   └── spark_preprocessor.py          🔧 Task 2 (30 pts): PySpark preprocessing + feature engineering
│
├── sql_interface/
│   ├── __init__.py
│   └── database_manager.py            🗄️ Task 3 (15 pts): SQLite database setup and queries
│
├── ml_models/
│   ├── __init__.py
│   ├── spark_gbt_forecaster.py        🤖 Task 4 (35 pts): Time series forecasting (Spark GBT)
│   └── investment_classifier.py       🎯 Task 5 (20 pts): Investment classification (High/Med/Low)
│
├── chatbot/
│   ├── __init__.py
│   ├── investment_chatbot.py          💬 Basic chatbot template
│   └── ai_prediction_chatbot.py       🤖 Task 6 (30 pts): Advanced chatbot with ML + graphs
│
├── dashboard/
│   ├── __init__.py
│   └── dashboard_app.py               📊 Task 7 (20 pts): Streamlit dashboard with visualizations
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py               ✅ Task 8 (10 pts): Unit tests
```

---

## 📚 File Descriptions

### Core Files (Must Complete)

#### 1. **README.md** ⭐
- **Purpose**: Project overview, task descriptions, grading rubric
- **What Students Learn**: Project requirements, deliverables, resources
- **Action**: Read this first to understand the assignment

#### 2. **SETUP.md** ⭐
- **Purpose**: Step-by-step setup instructions
- **What Students Learn**: Docker, Ollama, LLM setup, troubleshooting
- **Action**: Follow all setup steps before coding

#### 3. **requirements.txt**
- **Purpose**: List of Python dependencies
- **Action**: Run `pip install -r requirements.txt`

#### 4. **main.py** 🚀
- **Purpose**: Main pipeline orchestrator with menu
- **What Students Learn**: Pipeline design, orchestration
- **Action**: Complete menu logic to run all components

---

### Configuration

#### 5. **config/config.py** ⚙️
- **Purpose**: Central configuration (paths, hyperparameters, settings)
- **What Students Learn**: Configuration management, best practices
- **Task**: Review and modify settings for your system
- **Key Variables**:
  - `DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']`
  - `MA_WINDOWS = [7, 30, 90]`
  - `LOOKBACK_DAYS = 30`
  - `GBT_MAX_ITER = 100`

---

### Data Collection (Task 1: 20 points)

#### 6. **data_collection/stock_downloader.py** 📊
- **Purpose**: Download historical stock data from Yahoo Finance
- **What Students Learn**: API integration, data validation, error handling
- **Key Functions to Implement**:
  - `download_stock_data(ticker)` - Download data for one ticker
  - `download_all()` - Download all tickers
  - `save_to_csv()` - Save data to CSV files
  - `validate_data()` - Data quality checks

- **Expected Output**:
  ```
  data/stock_data/
  ├── AAPL_stock_data.csv (1254 rows)
  ├── MSFT_stock_data.csv (1254 rows)
  ├── GOOGL_stock_data.csv (1254 rows)
  ├── AMZN_stock_data.csv (1254 rows)
  └── TSLA_stock_data.csv (1254 rows)
  ```

- **Testing**: `python data_collection/stock_downloader.py`

---

### Preprocessing (Task 2: 30 points)

#### 7. **preprocessing/spark_preprocessor.py** 🔧
- **Purpose**: Clean data and engineer features using PySpark
- **What Students Learn**: PySpark DataFrames, Window functions, feature engineering
- **Key Functions to Implement**:
  - `load_csv_files()` - Load CSVs into Spark DataFrame
  - `calculate_moving_averages()` - MA_7, MA_30, MA_90 (8 pts)
  - `calculate_rsi()` - Relative Strength Index (10 pts)
  - `calculate_volatility()` - Rolling std dev (3 pts)
  - `calculate_returns()` - Daily returns + Sharpe ratio (4 pts)
  - `handle_missing_values()` - Handle NaN values

- **Expected Output**:
  ```
  data/processed_stocks.parquet/
  Columns: Ticker, Date, Open, High, Low, Close, Volume,
           MA_7, MA_30, MA_90, RSI, Volatility, Daily_Return, Sharpe_Ratio
  Rows: ~6,270
  ```

- **Testing**: `python preprocessing/spark_preprocessor.py`

---

### Database (Task 3: 15 points)

#### 8. **sql_interface/database_manager.py** 🗄️
- **Purpose**: SQLite database operations
- **What Students Learn**: SQL, database design, ORM
- **Key Functions to Implement**:
  - `create_tables()` - Create schema
  - `load_from_parquet()` - Load processed data into SQLite
  - `get_stock_data(ticker)` - Query by ticker
  - `get_latest_prices()` - Get most recent prices

- **Schema**:
  ```sql
  CREATE TABLE stock_data (
      ticker TEXT NOT NULL,
      date DATETIME NOT NULL,
      open REAL,
      high REAL,
      low REAL,
      close REAL,
      volume INTEGER,
      ma_7 REAL,
      ma_30 REAL,
      ma_90 REAL,
      rsi REAL,
      volatility REAL,
      daily_return REAL,
      sharpe_ratio REAL,
      PRIMARY KEY (ticker, date)
  );
  ```

- **Testing**: `python sql_interface/database_manager.py`

---

### Machine Learning

#### 9. **ml_models/spark_gbt_forecaster.py** 🤖 (Task 4: 35 points)
- **Purpose**: Time series forecasting using Spark MLlib GBT
- **What Students Learn**: Time series ML, Spark MLlib, lagged features
- **Key Functions to Implement**:
  - `create_lagged_features()` - Create 150 lagged features (15 pts)
  - `train()` - Train GBT model (10 pts)
  - `predict_future()` - Forecast N days ahead (10 pts)

- **Model Details**:
  - **Input**: 150 features (Close_lag_1-30, Open_lag_1-30, etc.)
  - **Output**: Stock price 7 days in future
  - **Algorithm**: Gradient Boosted Trees (GBTRegressor)
  - **Hyperparameters**: 100 trees, max_depth=6, learning_rate=0.1

- **Expected Performance**:
  - Test R² Score: 0.90+ (90%+ accuracy)
  - Test RMSE: $25-40
  - Mean % Error: < 10%

- **Testing**: `python ml_models/spark_gbt_forecaster.py`

#### 10. **ml_models/investment_classifier.py** 🎯 (Task 5: 20 points)
- **Purpose**: Classify stocks as High/Medium/Low investment potential
- **What Students Learn**: Classification, feature engineering, composite scoring
- **Key Functions to Implement**:
  - `engineer_features()` - Create 17 aggregate features
  - `calculate_composite_score()` - Multi-criteria scoring
  - `train()` - Train Random Forest Classifier

- **Features** (17 total):
  - Returns: Total, 7-day, 30-day
  - Technical: RSI, Volatility, Sharpe Ratio
  - Trends: MA7 vs MA30, MA30 vs MA90

- **Classification**:
  - High: Score ≥ 7 (Strong buy)
  - Medium: 4 ≤ Score < 7 (Hold)
  - Low: Score < 4 (Avoid)

- **Testing**: `python ml_models/investment_classifier.py`

---

### Chatbot (Task 6: 30 points)

#### 11. **chatbot/investment_chatbot.py** 💬
- **Purpose**: Basic chatbot template using Llama
- **What Students Learn**: LLM integration, chatbot basics

#### 12. **chatbot/ai_prediction_chatbot.py** 🤖 (Main Task)
- **Purpose**: Advanced chatbot with database, ML, and graph generation
- **What Students Learn**: NLP, intent detection, API integration, image generation
- **Key Components**:
  - Database queries (SQLite)
  - ML predictions (Spark GBT model)
  - Graph generation (Matplotlib → base64 PNG)
  - Intent detection (natural language)
  - LLM integration (Ollama + Llama 3.2)

- **Features to Implement**:
  - `query_database()` - Query SQLite
  - `get_prediction()` - Get ML forecasts
  - `generate_prediction_graph()` - Create chart image
  - `detect_intent()` - NLP intent detection
  - `get_response()` - Main response handler

- **Supported Queries**:
  - "Predict AAPL next 7 days" → ML prediction + graph
  - "Show TSLA data" → Historical data table
  - "What is RSI?" → LLM explanation

- **Testing**: `streamlit run chatbot/ai_prediction_chatbot.py --server.port 8502`

---

### Dashboard (Task 7: 20 points)

#### 13. **dashboard/dashboard_app.py** 📊
- **Purpose**: Interactive Streamlit dashboard for visualization
- **What Students Learn**: Web development, Streamlit, data visualization
- **Required Tabs**:
  1. **Stock Data Viewer** - Display historical data
  2. **Technical Indicators** - Plot MA, RSI, Volatility
  3. **ML Predictions** - Show forecasts with charts
  4. **Investment Classification** - Display classification results
  5. **Model Explanations** - Explain features and model

- **Key Features**:
  - Dropdown selectors for tickers
  - Interactive charts (Plotly/Matplotlib)
  - Data tables
  - Model performance metrics

- **Testing**: `streamlit run dashboard/dashboard_app.py --server.port 8501`

---

### Testing (Task 8: 10 points)

#### 14. **tests/test_pipeline.py** ✅
- **Purpose**: Unit tests for all components
- **What Students Learn**: Testing, pytest, edge cases
- **Tests to Implement**:
  - Test data collection (API errors, validation)
  - Test preprocessing (feature calculations)
  - Test database (queries, integrity)
  - Test ML models (predictions, accuracy)
  - Integration test (full pipeline)

- **Testing**: `pytest tests/test_pipeline.py -v`

---

## 🎯 Recommended Order

### Phase 1: Setup (Day 1)
1. Read **README.md** fully
2. Follow **SETUP.md** step-by-step
3. Install Docker, Ollama, dependencies
4. Test Ollama with Llama 3.2
5. Verify all imports work

### Phase 2: Data Pipeline (Days 2-3)
1. **Task 1**: Implement `stock_downloader.py`
2. **Task 2**: Implement `spark_preprocessor.py`
3. **Task 3**: Implement `database_manager.py`
4. Test: Run data collection → preprocessing → database

### Phase 3: Machine Learning (Days 4-5)
1. **Task 4**: Implement `spark_gbt_forecaster.py`
2. **Task 5**: Implement `investment_classifier.py`
3. Train models and evaluate performance
4. Save trained models

### Phase 4: Applications (Days 6-7)
1. **Task 6**: Implement `ai_prediction_chatbot.py`
2. **Task 7**: Implement `dashboard_app.py`
3. Test chatbot queries
4. Test dashboard visualizations

### Phase 5: Testing & Documentation (Day 8)
1. **Task 8**: Write unit tests
2. Complete `main.py` menu
3. Write documentation
4. Create demo video
5. Prepare submission

---

## 🔑 Key Files for Students

| File | Priority | Difficulty | Points |
|------|----------|-----------|--------|
| **SETUP.md** | ⭐⭐⭐ | Easy | 0 (required) |
| **README.md** | ⭐⭐⭐ | Easy | 0 (required) |
| **stock_downloader.py** | ⭐⭐⭐ | Medium | 20 |
| **spark_preprocessor.py** | ⭐⭐⭐ | Hard | 30 |
| **spark_gbt_forecaster.py** | ⭐⭐⭐ | Very Hard | 35 |
| **ai_prediction_chatbot.py** | ⭐⭐ | Hard | 30 |
| **dashboard_app.py** | ⭐⭐ | Medium | 20 |
| **database_manager.py** | ⭐ | Easy | 15 |
| **investment_classifier.py** | ⭐ | Medium | 20 |
| **test_pipeline.py** | ⭐ | Medium | 10 |

---

## 📝 What's Already Done

**Template provides**:
- ✅ Project structure
- ✅ Configuration system
- ✅ Detailed TODOs and hints
- ✅ Function signatures
- ✅ Documentation
- ✅ Setup instructions
- ✅ Example code patterns

**Students must implement**:
- 🔲 All function bodies (marked with TODO)
- 🔲 ML model training logic
- 🔲 Feature engineering algorithms
- 🔲 Chatbot intent detection
- 🔲 Dashboard visualizations
- 🔲 Testing suite

---

## 🏁 Success Criteria

Your project is complete when:
- ✅ All 8 tasks implemented
- ✅ Pipeline runs end-to-end without errors
- ✅ ML model achieves R² ≥ 0.90
- ✅ Chatbot responds to all query types
- ✅ Dashboard shows all visualizations
- ✅ Tests pass with good coverage
- ✅ Code is documented and clean
- ✅ README explains how to run everything

---

**Good luck with your assignment! 🚀**
