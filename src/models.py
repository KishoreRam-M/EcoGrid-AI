import logging
import numpy as np
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ==========================================
# GARCH(1,1) VOLATILITY ESTIMATION ENGINE
# ==========================================

class GARCH11:
    """
    Custom lightweight GARCH(1,1) model using SciPy MLE optimization.
    """
    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta = None
        self.fitted = False

    def _negative_log_likelihood(self, params, residuals):
        omega, alpha, beta = params
        n = len(residuals)
        variance = np.zeros(n)
        variance[0] = np.var(residuals)  # Seed variance with sample variance
        
        for t in range(1, n):
            variance[t] = omega + alpha * (residuals[t-1] ** 2) + beta * variance[t-1]
            
        variance = np.clip(variance, 1e-8, None)
        nll = 0.5 * np.sum(np.log(variance) + (residuals ** 2) / variance)
        return nll

    def fit(self, residuals):
        var_init = np.var(residuals)
        init_params = [0.05 * var_init, 0.05, 0.90]
        bounds = ((1e-10, None), (1e-10, 0.999), (1e-10, 0.999))
        constraints = ({'type': 'ineq', 'fun': lambda x: 0.999 - (x[1] + x[2])})  # alpha + beta < 1 stability
        
        res = minimize(
            self._negative_log_likelihood, 
            init_params, 
            args=(residuals,), 
            bounds=bounds, 
            constraints=constraints, 
            method="SLSQP"
        )
        
        if res.success:
            self.omega, self.alpha, self.beta = res.x
            self.fitted = True
            logger.info(f"GARCH(1,1) parameters fitted successfully: omega={self.omega:.6f}, alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        else:
            # Fallback to standard historic parameters if optimization fails
            self.omega = 0.05 * var_init
            self.alpha = 0.05
            self.beta = 0.90
            self.fitted = True
            logger.warning("GARCH optimization failed to converge. Utilizing fallback parameters.")
            
        return self

    def forecast(self, residuals, horizon_days: int) -> np.ndarray:
        """
        Forecasts variance step-ahead.
        """
        if not self.fitted:
            raise ValueError("GARCH model must be fitted before running forecasts.")
            
        n = len(residuals)
        variance = np.zeros(n)
        variance[0] = np.var(residuals)
        
        for t in range(1, n):
            variance[t] = self.omega + self.alpha * (residuals[t-1] ** 2) + self.beta * variance[t-1]
            
        last_var = variance[-1]
        last_resid = residuals[-1]
        
        forecast_var = np.zeros(horizon_days)
        # First step ahead
        forecast_var[0] = self.omega + self.alpha * (last_resid ** 2) + self.beta * last_var
        
        # Succeeding steps
        persistence = self.alpha + self.beta
        unconditional_var = self.omega / (1.0 - persistence + 1e-9)
        
        for h in range(1, horizon_days):
            forecast_var[h] = self.omega + persistence * forecast_var[h-1]
            
        return np.sqrt(forecast_var)


# ==========================================
# 1. HYBRID PROPHET + GARCH FORECAST
# ==========================================

def fit_and_forecast_prophet(df: pd.DataFrame, forecast_days: int = 30) -> tuple:
    """
    Fits Prophet for macro trend and GARCH(1,1) on residuals to estimate dynamic volatility bands.
    
    Returns:
        tuple: (fitted_model, forecast_df, metrics_dict)
    """
    logger.info("Fitting Prophet + GARCH(1,1) hybrid model...")
    
    # Format data for Prophet
    prophet_df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=forecast_days, freq="D")
    forecast = model.predict(future)
    
    # 1. GARCH Sizing on residuals
    in_sample = forecast.iloc[:-forecast_days]
    actuals = prophet_df["y"].values
    yhats = in_sample["yhat"].values
    residuals = actuals - yhats
    
    # Fit GARCH on Prophet residuals
    garch = GARCH11().fit(residuals)
    garch_vol_forecast = garch.forecast(residuals, forecast_days)
    
    # Scale uncertainty bands dynamically using GARCH volatility
    mean_historical_vol = np.std(residuals)
    vol_scale_factor = garch_vol_forecast / (mean_historical_vol + 1e-9)
    
    # Apply dynamic scaling factors to the out-of-sample forecast intervals
    # Prophet's default width = yhat_upper - yhat_lower
    forecast_out_of_sample = forecast.iloc[-forecast_days:].copy()
    default_widths = (forecast_out_of_sample["yhat_upper"] - forecast_out_of_sample["yhat_lower"]).values / 2.0
    
    # Scale width using volatility scale factor
    scaled_widths = default_widths * vol_scale_factor
    
    forecast.loc[forecast.index[-forecast_days:], "yhat_lower"] = forecast_out_of_sample["yhat"] - scaled_widths
    forecast.loc[forecast.index[-forecast_days:], "yhat_upper"] = forecast_out_of_sample["yhat"] + scaled_widths
    
    # In-sample metrics
    mae = mean_absolute_error(actuals, yhats)
    rmse = np.sqrt(mean_squared_error(actuals, yhats))
    mape = np.mean(np.abs((actuals - yhats) / (actuals + 1e-9))) * 100
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
    
    return model, forecast, metrics


# ==========================================
# 2. STATIONARY ML REGRESSION (XGBOOST)
# ==========================================

def train_and_predict_xgb(train_df: pd.DataFrame) -> tuple:
    """
    Trains XGBoost to predict next-day log returns using stationary indicators.
    Reconstructs absolute prices on the validation set for visual feedback.
    """
    logger.info("Training stationary XGBoost Regressor on Log Returns...")
    
    feature_cols = [
        "Return_Lag_1", "Return_Lag_7", "Return_Lag_14",
        "Return_Mean_7d", "Return_Std_7d",
        "Return_Mean_14d", "Return_Std_14d",
        "RSI_14", "SMA_50_Ratio", "SMA_200_Ratio"
    ]
    
    X = train_df[feature_cols]
    y = train_df["Target_Log_Return"]
    
    split_idx = int(len(train_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = xgb.XGBRegressor(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)
    
    # Predict returns
    test_pred_returns = model.predict(X_test)
    
    # Reconstruct absolute prices on test set: P(t+1) = P(t) * exp(R(t+1))
    test_actual_prices = train_df["Close"].iloc[split_idx+1:].values
    prev_close_prices = train_df["Close"].iloc[split_idx:split_idx+len(test_pred_returns)-1].values
    
    # Exclude last index mismatch if split bounds shift
    min_len = min(len(test_actual_prices), len(test_pred_returns)-1)
    
    reconstructed_preds = prev_close_prices[:min_len] * np.exp(test_pred_returns[:min_len])
    actual_test_targets = test_actual_prices[:min_len]
    test_dates = train_df.index[split_idx+1:split_idx+1+min_len]
    
    # Evaluate return indicators
    mae = mean_absolute_error(actual_test_targets, reconstructed_preds)
    rmse = np.sqrt(mean_squared_error(actual_test_targets, reconstructed_preds))
    r2 = r2_score(actual_test_targets, reconstructed_preds)
    
    # Directional Accuracy: Sign alignment of returns
    directional_accuracy = np.mean(np.sign(y_test.values[:min_len]) == np.sign(test_pred_returns[:min_len])) * 100
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy
    }
    
    test_results = pd.DataFrame({
        "Actual": actual_test_targets,
        "Predicted": reconstructed_preds
    }, index=test_dates)
    
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    return model, test_results, metrics, importance_df

def predict_next_day_xgb(model, inference_row: pd.DataFrame, current_price: float) -> float:
    """
    Predicts tomorrow's price using today's features and current price scale.
    """
    feature_cols = [
        "Return_Lag_1", "Return_Lag_7", "Return_Lag_14",
        "Return_Mean_7d", "Return_Std_7d",
        "Return_Mean_14d", "Return_Std_14d",
        "RSI_14", "SMA_50_Ratio", "SMA_200_Ratio"
    ]
    latest_features = inference_row[feature_cols]
    pred_return = model.predict(latest_features)[0]
    predicted_price = current_price * np.exp(pred_return)
    return float(predicted_price)


# ==========================================
# 3. CLASSIFICATION MODEL (RISK REGIME ENGINE)
# ==========================================

def define_risk_regimes(df: pd.DataFrame) -> tuple:
    """
    Labels each day dynamically based on volatility distributions.
    """
    df = df.copy()
    temp_df = df.dropna(subset=["Roll_Vol_30d", "Drawdown"])
    
    vol_50 = temp_df["Roll_Vol_30d"].median()
    vol_75 = temp_df["Roll_Vol_30d"].quantile(0.75)
    
    def get_regime(row):
        vol = row["Roll_Vol_30d"]
        dd = row["Drawdown"]
        
        if pd.isna(vol) or pd.isna(dd):
            return "Medium"
            
        if vol > vol_75 or dd < -0.15:
            return "High"
        elif vol > vol_50 or dd < -0.05:
            return "Medium"
        else:
            return "Low"
            
    df["Risk_Regime"] = df.apply(get_regime, axis=1)
    return df, {"vol_50": vol_50, "vol_75": vol_75}

def train_risk_classifier(df_labeled: pd.DataFrame) -> tuple:
    """
    Trains a Random Forest Classifier to predict next-day Risk Regime.
    """
    logger.info("Training Risk Engine Classifier...")
    df = df_labeled.copy()
    
    feature_cols = [
        "Daily_Return", "Log_Return", "Roll_Vol_7d", "Roll_Vol_30d",
        "SMA_50_Ratio", "SMA_200_Ratio", "Drawdown"
    ]
    
    df = df.dropna(subset=feature_cols + ["Risk_Regime"])
    df["Target_Risk"] = df["Risk_Regime"].shift(-1)
    df = df.dropna(subset=["Target_Risk"])
    
    le = LabelEncoder()
    df["Target_Risk_Encoded"] = le.fit_transform(df["Target_Risk"])
    
    X = df[feature_cols]
    y = df["Target_Risk_Encoded"]
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    target_names = list(le.classes_)
    
    report = classification_report(y_test, preds, target_names=target_names, output_dict=True, zero_division=0)
    
    metrics = {
        "Accuracy": accuracy,
        "Report": report
    }
    
    return clf, le, metrics

def predict_next_day_risk(clf, le, latest_row: pd.DataFrame) -> str:
    """
    Predicts tomorrow's Risk Regime based on today's technical states.
    """
    feature_cols = [
        "Daily_Return", "Log_Return", "Roll_Vol_7d", "Roll_Vol_30d",
        "SMA_50_Ratio", "SMA_200_Ratio", "Drawdown"
    ]
    latest_features = latest_row[feature_cols]
    pred_encoded = clf.predict(latest_features)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    return pred_label
