import os
import pickle
import logging
import pandas as pd
import yfinance as yf

# Import project modules
from src.data_pipeline import fetch_ticker_data
from src.database import cache_ticker_data, get_cached_ticker_data
from src.features import compute_basic_features, compute_ml_features
from src.models import (
    train_and_predict_xgb,
    define_risk_regimes,
    train_risk_classifier
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TICKERS = ["ICLN", "XLU", "CEG"]
MODELS_DIR = "models"

def run_training_pipeline():
    logger.info("Initializing offline batch model training...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for ticker in TICKERS:
        logger.info(f"------------------ Processing {ticker} ------------------")
        
        # 1. Fetch data with database caching and fallback
        try:
            # Try fetching live data (5y period)
            df_raw = fetch_ticker_data(ticker, "5y")
            # Cache to database
            cache_ticker_data(ticker, df_raw)
        except Exception as e:
            logger.warning(f"Failed to fetch live data for {ticker}: {e}. Attempting database cache fallback.")
            df_raw = get_cached_ticker_data(ticker)
            if df_raw.empty:
                logger.error(f"Critical Error: No live or cached data available for {ticker}. Skipping training.")
                continue
                
        # 2. Compute basic features & Risk regimes
        df_basic = compute_basic_features(df_raw)
        df_labeled, risk_meta = define_risk_regimes(df_basic)
        
        # Save risk metadata thresholds
        with open(os.path.join(MODELS_DIR, f"{ticker}_risk_meta.pkl"), "wb") as f:
            pickle.dump(risk_meta, f)
            
        # 3. Compute ML features (split into training and inference subsets)
        train_df, inference_row = compute_ml_features(df_labeled)
        
        # Save inference row for dashboard quick lookup
        inference_row.to_pickle(os.path.join(MODELS_DIR, f"{ticker}_latest_features.pkl"))
        
        # 4. Train XGBoost model
        xgb_model, test_results, xgb_metrics, importance_df = train_and_predict_xgb(train_df)
        with open(os.path.join(MODELS_DIR, f"{ticker}_xgb.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
        logger.info(f"XGBoost model saved for {ticker}. Test MAE: {xgb_metrics['MAE']:.2f}")
        
        # Save XGBoost metrics for UI visualization
        with open(os.path.join(MODELS_DIR, f"{ticker}_xgb_metrics.pkl"), "wb") as f:
            pickle.dump(xgb_metrics, f)
        with open(os.path.join(MODELS_DIR, f"{ticker}_xgb_results.pkl"), "wb") as f:
            pickle.dump(test_results, f)
        with open(os.path.join(MODELS_DIR, f"{ticker}_xgb_importance.pkl"), "wb") as f:
            pickle.dump(importance_df, f)
            
        # 5. Train Risk Engine Classifier (Random Forest)
        rf_clf, le, rf_metrics = train_risk_classifier(df_labeled)
        with open(os.path.join(MODELS_DIR, f"{ticker}_risk_clf.pkl"), "wb") as f:
            pickle.dump(rf_clf, f)
        with open(os.path.join(MODELS_DIR, f"{ticker}_risk_le.pkl"), "wb") as f:
            pickle.dump(le, f)
        with open(os.path.join(MODELS_DIR, f"{ticker}_risk_metrics.pkl"), "wb") as f:
            pickle.dump(rf_metrics, f)
        logger.info(f"Risk Classifier saved for {ticker}. Test Accuracy: {rf_metrics['Accuracy']:.2%}")
        
    logger.info("Batch model training complete. All artifacts serialized successfully.")

if __name__ == "__main__":
    run_training_pipeline()
