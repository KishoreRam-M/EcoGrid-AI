def get_procurement_advice(
    ticker: str,
    current_price: float,
    rolling_vol_30d: float,
    current_drawdown: float,
    predicted_risk_regime: str,
    trend_state: str,
    forecast_uncertainty_ratio: float
) -> dict:
    """
    Translates model predictions and volatility risk levels into concrete, rule-based
    business recommendations for an energy procurement manager.
    
    Parameters:
        ticker (str): The asset ticker ('ICLN', 'XLU', 'CEG').
        current_price (float): Latest close price.
        rolling_vol_30d (float): Annualized 30-day rolling volatility.
        current_drawdown (float): Current peak-to-trough decline.
        predicted_risk_regime (str): Predicted risk regime ('Low', 'Medium', 'High').
        trend_state (str): Short-term trend description ('Bullish' or 'Bearish').
        forecast_uncertainty_ratio (float): Width of the Prophet uncertainty interval normalized by price.
        
    Returns:
        dict: A recommendation payload with fields 'action', 'badge_color', 'headline', 'justifications', 'tactics'.
    """
    
    # Initialize response structure
    advice = {
        "ticker": ticker,
        "action": "STANDARD PROCUREMENT RUN",
        "badge_color": "#10B981",  # Emerald Green
        "headline": "Stable market conditions. Maintain baseline procurement actions.",
        "justifications": [],
        "tactics": []
    }
    
    # 1. EVALUATE HIGH RISK / HEDGING SCENARIOS
    if predicted_risk_regime == "High" or rolling_vol_30d > 0.40 or current_drawdown < -0.15:
        if ticker == "XLU":
            advice["action"] = "OVERWEIGHT UTILITY HEDGE"
            advice["badge_color"] = "#F59E0B"  # Warm Amber
            advice["headline"] = "Clean energy volatility is spiking. Shift allocation to defensive utilities."
            advice["justifications"] = [
                f"Defensive utility index XLU exhibits lower relative risk under high volatility.",
                f"Current annualized 30-day volatility is high at {rolling_vol_30d:.1%}.",
                f"Peak drawdown has reached {current_drawdown:.1%}, signaling a risk regime shift."
            ]
            advice["tactics"] = [
                "Increase allocations to regulated utility contracts to establish a pricing floor.",
                "Suspend direct clean energy spot market exposure.",
                "Execute hedging derivative contracts or green capacity options."
            ]
        else:
            advice["action"] = "SUSPEND SPOT ACQUISITIONS / HEDGE EXPOSURE"
            advice["badge_color"] = "#EF4444"  # Coral Red
            advice["headline"] = f"CRITICAL VOLATILITY SPIKE: De-risk asset exposure for {ticker} immediately."
            advice["justifications"] = [
                f"Predicted next-day Risk Regime is HIGH risk.",
                f"Drawdown is at a severe {current_drawdown:.1%}.",
                f"Annualized volatility ({rolling_vol_30d:.1%}) has breached risk limits."
            ]
            advice["tactics"] = [
                "Immediately freeze spot market PPA (Power Purchase Agreement) purchases.",
                "Rely on existing battery energy storage systems (BESS) or grid buffer reserves.",
                "Temporarily source baseline power from utility contracts to insulate budgets from spot rate spikes."
            ]
            
    # 2. EVALUATE MEDIUM RISK / CONSERVATIVE BUYING
    elif predicted_risk_regime == "Medium" or current_drawdown < -0.05:
        if trend_state == "Bullish":
            advice["action"] = "DOLLAR-COST-AVERAGE / GRADUAL ACCUMULATION"
            advice["badge_color"] = "#3B82F6"  # Indigo Blue
            advice["headline"] = "Bullish momentum present amidst moderate risk. Buy incrementally."
            advice["justifications"] = [
                f"Asset has positive trend structure but medium volatility.",
                f"Uncertainty ratio is moderate ({forecast_uncertainty_ratio:.1%})."
            ]
            advice["tactics"] = [
                "Procure green energy blocks in monthly or bi-weekly installments.",
                "Utilize indexed-price contracts with caps to limit upside risk.",
                "Secure REC (Renewable Energy Certificate) volume requirements for the next quarter only."
            ]
        else:
            advice["action"] = "HOLD PROCUREMENT RUNS / MONITOR"
            advice["badge_color"] = "#EAB308"  # Yellow
            advice["headline"] = "Neutral or bearish trend with moderate volatility. Pause expansions."
            advice["justifications"] = [
                f"Asset is experiencing a bearish trend line under moderate risk.",
                f"Rolling 30-day volatility stands at {rolling_vol_30d:.1%}."
            ]
            advice["tactics"] = [
                "Maintain baseline compliance purchases but defer signing long-term PPAs.",
                "Re-negotiate index-linked contract clauses to capture lower prices.",
                "Audit internal data center loads to optimize demand-response programs."
            ]
            
    # 3. EVALUATE LOW RISK / OPPORTUNISTIC BUYING
    else:
        if trend_state == "Bullish" and forecast_uncertainty_ratio < 0.12:
            advice["action"] = "STRONG BUY / LOCK LONG-TERM PPAs"
            advice["badge_color"] = "#059669"  # Deep Emerald
            advice["headline"] = f"OPPORTUNISTIC WINDOW: Lock in long-term carbon-free energy supply."
            advice["justifications"] = [
                f"Volatility is low ({rolling_vol_30d:.1%}), and short-term price trend is highly bullish.",
                f"Prophet forecast uncertainty is low at {forecast_uncertainty_ratio:.1%}, showing high forecast confidence."
            ]
            advice["tactics"] = [
                "Lock in long-term fixed-price physical PPAs (e.g., 10-15 year solar/wind contracts).",
                "Pre-purchase Green RECs at current discount rates to fulfill annual ESG requirements.",
                "Minimize utility hedges to maximize carbon-free energy percentage match."
            ]
        else:
            advice["action"] = "EXECUTE STANDARD ACQUISITIONS"
            advice["badge_color"] = "#10B981"  # Standard Emerald
            advice["headline"] = "Standard market state. Continue steady procurement path."
            advice["justifications"] = [
                f"Asset volatility ({rolling_vol_30d:.1%}) is within standard tolerance ranges.",
                f"Trend structure remains stable."
            ]
            advice["tactics"] = [
                "Execute routine monthly carbon-free energy matching contracts.",
                "Continue standard grid utility offsets.",
                "Maintain existing portfolio allocation levels."
            ]
            
    return advice
