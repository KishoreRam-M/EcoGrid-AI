import logging
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fetch_ticker_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetches historical daily market data for a given ticker from Yahoo Finance.
    
    Parameters:
        ticker (str): Ticker symbol (e.g., 'ICLN', 'XLU', 'CEG').
        period (str): Historical data lookback period (default '5y' for deep history).
        
    Returns:
        pd.DataFrame: Cleaned historical data with 'Date' index and OHLCV columns.
    """
    logger.info(f"Fetching data for {ticker} with period {period}...")
    try:
        # Fetch data using yfinance
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker}. Verify if ticker symbol is correct or if API is available.")
            
        logger.info(f"Successfully fetched {len(df)} rows for {ticker}.")
        
        # Clean and validate the data
        df = clean_and_validate_data(df, ticker)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise e

def clean_and_validate_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Validates and cleans fetched market data.
    - Handles missing values via forward/backward fill.
    - Standardizes the index and columns.
    - Validates required columns are present.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure index is datetime and localized/timezone-naive for model consistency
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    df.index.name = "Date"
    
    # Validate required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' is missing in the fetched data for {ticker}.")
            
    # Handle missing values
    null_counts = df[required_cols].isnull().sum().sum()
    if null_counts > 0:
        logger.warning(f"Found {null_counts} null values in {ticker} dataset. Applying forward/backward fill.")
        df[required_cols] = df[required_cols].ffill().bfill()
        
    # Final check to verify no NaNs remain in required columns
    if df[required_cols].isnull().any().any():
        raise ValueError(f"Data for {ticker} contains unresolvable NaN values after cleaning.")
        
    return df
