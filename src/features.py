import numpy as np
import pandas as pd

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes baseline technical and financial features on raw historical data.
    Ensures stationary transformations for regression metrics.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing 'Close' price and DatetimeIndex.
        
    Returns:
        pd.DataFrame: Dataframe with basic features and stationary indicators.
    """
    df = df.copy()
    
    # 1. Daily Returns & Log Returns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # 2. Rolling Volatility (annualized, assuming 252 trading days/year)
    df["Roll_Vol_7d"] = df["Log_Return"].rolling(window=7).std() * np.sqrt(252)
    df["Roll_Vol_30d"] = df["Log_Return"].rolling(window=30).std() * np.sqrt(252)
    
    # 3. Simple Moving Averages & Stationary SMA Ratios
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    
    df["SMA_50_Ratio"] = (df["Close"] / df["SMA_50"]) - 1
    df["SMA_200_Ratio"] = (df["Close"] / df["SMA_200"]) - 1
    
    # 4. Drawdown calculation (Percentage drop from running peak)
    rolling_peak = df["Close"].cummax()
    df["Drawdown"] = (df["Close"] - rolling_peak) / rolling_peak
    
    return df

def compute_ml_features(df: pd.DataFrame) -> tuple:
    """
    Calculates stationary lag and rolling features for machine learning.
    Splits the data into a training dataset and a separate single-row inference vector.
    
    Parameters:
        df (pd.DataFrame): Dataframe with basic features computed.
        
    Returns:
        tuple: (train_df, inference_row)
    """
    df = df.copy()
    
    # 1. Stationary Lag features on Log Returns (No absolute Close price lags!)
    for lag in [1, 7, 14]:
        df[f"Return_Lag_{lag}"] = df["Log_Return"].shift(lag)
        
    # 2. Rolling statistics on log returns
    for window in [7, 14]:
        df[f"Return_Mean_{window}d"] = df["Log_Return"].rolling(window=window).mean()
        df[f"Return_Std_{window}d"] = df["Log_Return"].rolling(window=window).std()
        
    # 3. Relative Strength Index (RSI - 14 days, bounded and stationary)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # 4. Target variable: Next-day Log Return (Shifted forward by 1 step)
    df["Target_Log_Return"] = df["Log_Return"].shift(-1)
    
    feature_cols = [
        "Return_Lag_1", "Return_Lag_7", "Return_Lag_14",
        "Return_Mean_7d", "Return_Std_7d",
        "Return_Mean_14d", "Return_Std_14d",
        "RSI_14", "SMA_50_Ratio", "SMA_200_Ratio"
    ]
    
    # Extract inference row (The absolute latest row, today).
    # It contains all calculated features, but Target_Log_Return is NaN since tomorrow is unknown.
    inference_row = df.iloc[-1:].copy()
    
    # Create clean training set: drop rows with NaNs in training features or target labels
    train_df = df.dropna(subset=feature_cols + ["Target_Log_Return"])
    
    return train_df, inference_row

def compute_correlation_matrix(df_dict: dict) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix of daily log returns for a set of assets.
    """
    returns_dict = {}
    for ticker, df in df_dict.items():
        if "Log_Return" in df.columns:
            returns_dict[ticker] = df["Log_Return"]
        else:
            # Recompute on the fly
            returns_dict[ticker] = np.log(df["Close"] / df["Close"].shift(1))
            
    returns_df = pd.DataFrame(returns_dict)
    return returns_df.corr(method="pearson")
