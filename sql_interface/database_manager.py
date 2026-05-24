# sql_interface/database_manager.py

import sqlite3
from pathlib import Path
import pandas as pd


class DatabaseManager:
    """
    SQLite database manager for financial stock data.
    """

    def __init__(
        self,
        db_path: str = "data/financial_data.db",
        parquet_path: str = "data/processed_stocks.parquet"
    ):
        """
        Initialize database connection and create schema.

        Args:
            db_path: SQLite database file path
            parquet_path: Path to parquet dataset
        """

        self.db_path = Path(db_path)
        self.parquet_path = Path(parquet_path)

        # Create data directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to SQLite database
        self.conn = sqlite3.connect(self.db_path)

        # Improve SQLite performance
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        # Create schema
        self.create_table()

    def create_table(self):
        """
        Create stock_data table and indexes.
        """

        create_query = """
        CREATE TABLE IF NOT EXISTS stock_data (
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
        """

        self.conn.execute(create_query)

        # Create indexes for faster queries
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker
        ON stock_data(ticker);
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_date
        ON stock_data(date);
        """)

        self.conn.commit()

    def load_parquet_to_sqlite(self):
        """
        Load parquet data into SQLite database.
        """

        if not self.parquet_path.exists():

            available_files = list(Path("data").rglob("*.parquet"))

            raise FileNotFoundError(
                f"""
Parquet file not found: {self.parquet_path}

Available parquet files:
{available_files}
"""
            )

        print(f"Loading parquet file: {self.parquet_path}")

        # Read parquet file
        df = pd.read_parquet(self.parquet_path)

        # Convert date column properly
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Store in SQLite
        df.to_sql(
            "stock_data",
            self.conn,
            if_exists="replace",
            index=False
        )

        # Recreate indexes because replace drops them
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker
        ON stock_data(ticker);
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_date
        ON stock_data(date);
        """)

        self.conn.commit()

        print(f"Successfully loaded {len(df)} rows.")

    def get_stock_data(self, ticker: str):
        """
        Get all data for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            pandas.DataFrame
        """

        query = """
        SELECT *
        FROM stock_data
        WHERE ticker = ?
        ORDER BY date ASC;
        """

        return pd.read_sql_query(
            query,
            self.conn,
            params=(ticker,)
        )

    def get_latest_prices(self):
        """
        Get latest closing price for all tickers.

        Returns:
            pandas.DataFrame
        """

        query = """
        SELECT s1.ticker,
               s1.date,
               s1.close
        FROM stock_data s1
        INNER JOIN (
            SELECT ticker,
                   MAX(date) AS latest_date
            FROM stock_data
            GROUP BY ticker
        ) s2
        ON s1.ticker = s2.ticker
        AND s1.date = s2.latest_date
        ORDER BY s1.ticker;
        """

        return pd.read_sql_query(query, self.conn)

    def get_row_count(self):
        """
        Return total row count.
        """

        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM stock_data;"
        )

        return cursor.fetchone()[0]

    def close(self):
        """
        Close SQLite connection.
        """

        if self.conn:
            self.conn.close()


if __name__ == "__main__":

    # Initialize database manager
    db = DatabaseManager()

    # Load parquet into SQLite
    db.load_parquet_to_sqlite()

    # Print total rows
    print("\nTotal rows in database:")
    print(db.get_row_count())

    # Show latest prices
    print("\nLatest Prices:")
    print(db.get_latest_prices().head())

    # Example stock query
    print("\nSample Stock Data (AAPL):")
    print(db.get_stock_data("AAPL").head())

    # Close connection
    db.close()