import sqlite3
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "market_cache.db")

def init_db():
    """
    Initializes the SQLite database cache and creates required tables.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            Date TEXT,
            Ticker TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL,
            PRIMARY KEY (Date, Ticker)
        )
    """)
    conn.commit()
    conn.close()
    logger.info("SQLite database cache initialized successfully.")

def cache_ticker_data(ticker: str, df: pd.DataFrame):
    """
    Saves historical ticker data to the SQLite database.
    """
    if df.empty:
        return
        
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df_to_save = df.copy().reset_index()
    
    # Standardize column structure
    df_to_save["Date"] = df_to_save["Date"].dt.strftime("%Y-%m-%d")
    df_to_save["Ticker"] = ticker
    
    records = df_to_save[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].values.tolist()
    
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO market_data (Date, Ticker, Open, High, Low, Close, Volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    conn.close()
    logger.info(f"Cached {len(df_to_save)} records to SQLite database for {ticker}.")

def get_cached_ticker_data(ticker: str) -> pd.DataFrame:
    """
    Retrieves historical price records from the SQLite database cache.
    """
    if not os.path.exists(DB_PATH):
        logger.warning(f"Database cache file not found. No local data for {ticker}.")
        return pd.DataFrame()
        
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT Date, Open, High, Low, Close, Volume FROM market_data WHERE Ticker = ? ORDER BY Date ASC"
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    
    if df.empty:
        logger.warning(f"No cached data found for {ticker}.")
        return df
        
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    logger.info(f"Retrieved {len(df)} cached records from SQLite database for {ticker}.")
    return df
