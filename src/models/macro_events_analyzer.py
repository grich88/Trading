"""
CPI / Macro Events Tracking module.

This module provides advanced analysis of macroeconomic events and their impact on crypto markets,
including CPI data, FOMC events, economic indicators, and combined signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import calendar

from src.utils import (
    get_logger,
    performance_monitor,
    ModelError
)

# Import configuration
from src.config import (
    MACRO_LOOKBACK_PERIOD,
    MACRO_SIGNAL_THRESHOLD,
    CPI_IMPACT_WINDOW,
    FOMC_IMPACT_WINDOW
)

logger = get_logger("MacroEventsAnalyzer")


class MacroEventsAnalyzer:
    """
    Advanced CPI / Macro Events analyzer for market data.
    
    This class provides methods for:
    - CPI data analysis and impact measurement
    - FOMC event tracking and analysis
    - Economic indicator monitoring
    - Event impact assessment
    - Combined signal generation
    """
    
    def __init__(self, 
                 lookback_period: int = MACRO_LOOKBACK_PERIOD,
                 signal_threshold: float = MACRO_SIGNAL_THRESHOLD,
                 cpi_window: int = CPI_IMPACT_WINDOW,
                 fomc_window: int = FOMC_IMPACT_WINDOW):
        """
        Initialize the Macro Events analyzer.
        
        Args:
            lookback_period: Lookback period for analysis
            signal_threshold: Threshold for signal generation
            cpi_window: Impact window for CPI events (days)
            fomc_window: Impact window for FOMC events (days)
        """
        self.lookback_period = lookback_period
        self.signal_threshold = signal_threshold
        self.cpi_window = cpi_window
        self.fomc_window = fomc_window
        
        logger.info(f"MacroEventsAnalyzer initialized with lookback period: {lookback_period}, "
                   f"CPI window: {cpi_window}, FOMC window: {fomc_window}")
    
    @performance_monitor()
    def analyze_cpi_impact(self, 
                          cpi_data: pd.DataFrame,
                          price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze CPI data impact on crypto prices.
        
        Args:
            cpi_data: DataFrame with CPI data (date, actual, forecast, previous)
            price_data: DataFrame with price data
            
        Returns:
            Dictionary with CPI impact analysis
        """
        required_cpi_cols = ["date", "actual", "forecast", "previous"]
        for col in required_cpi_cols:
            if col not in cpi_data.columns:
                raise ModelError(f"Required column not found in CPI data: {col}")
        
        if "close" not in price_data.columns or "date" not in price_data.columns:
            raise ModelError("Price data must contain 'date' and 'close' columns")
        
        # Calculate CPI surprises
        cpi_data = cpi_data.copy()
        cpi_data["surprise"] = cpi_data["actual"] - cpi_data["forecast"]
        cpi_data["surprise_pct"] = (cpi_data["surprise"] / cpi_data["forecast"]) * 100
        cpi_data["mom_change"] = cpi_data["actual"] - cpi_data["previous"]
        
        # Analyze price impact for each CPI release
        impact_results = []
        
        for _, cpi_event in cpi_data.iterrows():
            event_date = pd.to_datetime(cpi_event["date"])
            
            # Find price data around the event
            start_date = event_date - timedelta(days=self.cpi_window)
            end_date = event_date + timedelta(days=self.cpi_window)
            
            event_prices = price_data[
                (pd.to_datetime(price_data["date"]) >= start_date) &
                (pd.to_datetime(price_data["date"]) <= end_date)
            ].copy()
            
            if len(event_prices) > 0:
                # Calculate pre and post event returns
                event_idx = event_prices[pd.to_datetime(event_prices["date"]) >= event_date].index[0] if len(event_prices[pd.to_datetime(event_prices["date"]) >= event_date]) > 0 else None
                
                if event_idx is not None and event_idx > 0:
                    pre_event_price = event_prices.iloc[event_idx - 1]["close"]
                    post_event_prices = event_prices.iloc[event_idx:]
                    
                    # Calculate returns at different intervals
                    returns = {}
                    for days in [1, 3, 7]:
                        if len(post_event_prices) > days:
                            future_price = post_event_prices.iloc[days]["close"]
                            returns[f"{days}d_return"] = ((future_price - pre_event_price) / pre_event_price) * 100
                    
                    impact_results.append({
                        "date": event_date,
                        "cpi_actual": cpi_event["actual"],
                        "cpi_forecast": cpi_event["forecast"],
                        "cpi_surprise": cpi_event["surprise"],
                        "cpi_surprise_pct": cpi_event["surprise_pct"],
                        **returns
                    })
        
        if not impact_results:
            return {
                "avg_impact": 0,
                "surprise_correlation": 0,
                "positive_surprise_impact": 0,
                "negative_surprise_impact": 0,
                "impact_events": [],
                "current_regime": "neutral"
            }
        
        impact_df = pd.DataFrame(impact_results)
        
        # Calculate average impacts
        avg_impact = impact_df[["1d_return", "3d_return", "7d_return"]].mean().to_dict() if "1d_return" in impact_df.columns else {}
        
        # Calculate correlation between CPI surprise and returns
        correlations = {}
        for col in ["1d_return", "3d_return", "7d_return"]:
            if col in impact_df.columns:
                correlations[col] = impact_df["cpi_surprise_pct"].corr(impact_df[col])
        
        # Analyze positive vs negative surprises
        positive_surprises = impact_df[impact_df["cpi_surprise"] > 0]
        negative_surprises = impact_df[impact_df["cpi_surprise"] < 0]
        
        positive_impact = positive_surprises[["1d_return", "3d_return", "7d_return"]].mean().to_dict() if len(positive_surprises) > 0 and "1d_return" in positive_surprises.columns else {}
        negative_impact = negative_surprises[["1d_return", "3d_return", "7d_return"]].mean().to_dict() if len(negative_surprises) > 0 and "1d_return" in negative_surprises.columns else {}
        
        # Determine current regime based on recent CPI trends
        recent_cpi = cpi_data.tail(3)
        if len(recent_cpi) >= 3:
            if all(recent_cpi["mom_change"] > 0):
                current_regime = "inflationary"
            elif all(recent_cpi["mom_change"] < 0):
                current_regime = "deflationary"
            else:
                current_regime = "transitional"
        else:
            current_regime = "neutral"
        
        return {
            "avg_impact": avg_impact,
            "surprise_correlation": correlations,
            "positive_surprise_impact": positive_impact,
            "negative_surprise_impact": negative_impact,
            "impact_events": impact_results[-5:],  # Last 5 events
            "current_regime": current_regime
        }
    
    @performance_monitor()
    def analyze_fomc_impact(self, 
                           fomc_data: pd.DataFrame,
                           price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze FOMC event impact on crypto prices.
        
        Args:
            fomc_data: DataFrame with FOMC data (date, decision, statement_sentiment)
            price_data: DataFrame with price data
            
        Returns:
            Dictionary with FOMC impact analysis
        """
        required_fomc_cols = ["date", "decision", "statement_sentiment"]
        for col in required_fomc_cols:
            if col not in fomc_data.columns:
                raise ModelError(f"Required column not found in FOMC data: {col}")
        
        # Analyze price impact for each FOMC event
        impact_results = []
        
        for _, fomc_event in fomc_data.iterrows():
            event_date = pd.to_datetime(fomc_event["date"])
            
            # Find price data around the event
            start_date = event_date - timedelta(days=self.fomc_window)
            end_date = event_date + timedelta(days=self.fomc_window)
            
            event_prices = price_data[
                (pd.to_datetime(price_data["date"]) >= start_date) &
                (pd.to_datetime(price_data["date"]) <= end_date)
            ].copy()
            
            if len(event_prices) > 0:
                # Calculate volatility before and after event
                event_idx = event_prices[pd.to_datetime(event_prices["date"]) >= event_date].index[0] if len(event_prices[pd.to_datetime(event_prices["date"]) >= event_date]) > 0 else None
                
                if event_idx is not None and event_idx > self.fomc_window:
                    pre_event_prices = event_prices.iloc[:event_idx]
                    post_event_prices = event_prices.iloc[event_idx:]
                    
                    # Calculate volatility
                    pre_volatility = pre_event_prices["close"].pct_change().std() * np.sqrt(252)
                    post_volatility = post_event_prices["close"].pct_change().std() * np.sqrt(252) if len(post_event_prices) > 1 else 0
                    
                    # Calculate returns
                    pre_event_price = event_prices.iloc[event_idx - 1]["close"] if event_idx > 0 else event_prices.iloc[0]["close"]
                    
                    returns = {}
                    for days in [1, 3, 7]:
                        if len(post_event_prices) > days:
                            future_price = post_event_prices.iloc[days]["close"]
                            returns[f"{days}d_return"] = ((future_price - pre_event_price) / pre_event_price) * 100
                    
                    impact_results.append({
                        "date": event_date,
                        "decision": fomc_event["decision"],
                        "sentiment": fomc_event["statement_sentiment"],
                        "pre_volatility": pre_volatility,
                        "post_volatility": post_volatility,
                        "volatility_change": post_volatility - pre_volatility,
                        **returns
                    })
        
        if not impact_results:
            return {
                "avg_volatility_change": 0,
                "decision_impacts": {},
                "sentiment_impacts": {},
                "impact_events": [],
                "upcoming_event": None
            }
        
        impact_df = pd.DataFrame(impact_results)
        
        # Calculate average volatility change
        avg_volatility_change = impact_df["volatility_change"].mean()
        
        # Analyze impact by decision type
        decision_impacts = {}
        for decision in impact_df["decision"].unique():
            decision_data = impact_df[impact_df["decision"] == decision]
            if len(decision_data) > 0:
                decision_impacts[decision] = {
                    "avg_return": decision_data[["1d_return", "3d_return", "7d_return"]].mean().to_dict() if "1d_return" in decision_data.columns else {},
                    "avg_volatility_change": decision_data["volatility_change"].mean()
                }
        
        # Analyze impact by sentiment
        sentiment_impacts = {}
        for sentiment in impact_df["sentiment"].unique():
            sentiment_data = impact_df[impact_df["sentiment"] == sentiment]
            if len(sentiment_data) > 0:
                sentiment_impacts[sentiment] = {
                    "avg_return": sentiment_data[["1d_return", "3d_return", "7d_return"]].mean().to_dict() if "1d_return" in sentiment_data.columns else {},
                    "avg_volatility_change": sentiment_data["volatility_change"].mean()
                }
        
        # Find upcoming FOMC event
        upcoming_event = None
        future_events = fomc_data[pd.to_datetime(fomc_data["date"]) > datetime.now()]
        if len(future_events) > 0:
            next_event = future_events.iloc[0]
            upcoming_event = {
                "date": next_event["date"],
                "days_until": (pd.to_datetime(next_event["date"]) - datetime.now()).days
            }
        
        return {
            "avg_volatility_change": float(avg_volatility_change),
            "decision_impacts": decision_impacts,
            "sentiment_impacts": sentiment_impacts,
            "impact_events": impact_results[-5:],  # Last 5 events
            "upcoming_event": upcoming_event
        }
    
    @performance_monitor()
    def analyze_economic_calendar(self, 
                                 calendar_data: pd.DataFrame,
                                 lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze economic calendar for upcoming high-impact events.
        
        Args:
            calendar_data: DataFrame with economic calendar data
            lookback_days: Number of days to look back for pattern analysis
            
        Returns:
            Dictionary with economic calendar analysis
        """
        required_cols = ["date", "event", "impact", "currency"]
        for col in required_cols:
            if col not in calendar_data.columns:
                raise ModelError(f"Required column not found in calendar data: {col}")
        
        # Filter high-impact events
        high_impact_events = calendar_data[calendar_data["impact"] == "High"].copy()
        
        # Analyze upcoming events
        current_date = datetime.now()
        upcoming_week = current_date + timedelta(days=7)
        
        upcoming_events = high_impact_events[
            (pd.to_datetime(high_impact_events["date"]) >= current_date) &
            (pd.to_datetime(high_impact_events["date"]) <= upcoming_week)
        ]
        
        # Count events by day
        events_by_day = {}
        for i in range(7):
            day_date = current_date + timedelta(days=i)
            day_events = upcoming_events[
                pd.to_datetime(upcoming_events["date"]).dt.date == day_date.date()
            ]
            events_by_day[day_date.strftime("%Y-%m-%d")] = len(day_events)
        
        # Analyze event patterns
        lookback_date = current_date - timedelta(days=lookback_days)
        recent_events = high_impact_events[
            pd.to_datetime(high_impact_events["date"]) >= lookback_date
        ]
        
        # Count events by type
        event_counts = recent_events["event"].value_counts().to_dict()
        
        # Analyze event clustering
        event_dates = pd.to_datetime(recent_events["date"]).dt.date.unique()
        if len(event_dates) > 1:
            date_diffs = np.diff(sorted(event_dates))
            avg_days_between = np.mean([d.days for d in date_diffs])
            event_clustering = 1 / (1 + avg_days_between)  # Higher value = more clustered
        else:
            event_clustering = 0
        
        # Determine event density
        events_per_week = len(recent_events) / (lookback_days / 7)
        
        # Categorize upcoming week
        total_upcoming = sum(events_by_day.values())
        if total_upcoming >= 5:
            week_category = "high_impact"
        elif total_upcoming >= 2:
            week_category = "moderate_impact"
        else:
            week_category = "low_impact"
        
        return {
            "upcoming_events": upcoming_events.to_dict('records'),
            "events_by_day": events_by_day,
            "event_counts": event_counts,
            "event_clustering": float(event_clustering),
            "events_per_week": float(events_per_week),
            "week_category": week_category,
            "total_upcoming": int(total_upcoming)
        }
    
    @performance_monitor()
    def calculate_macro_sentiment(self, 
                                cpi_analysis: Dict[str, Any],
                                fomc_analysis: Dict[str, Any],
                                calendar_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall macroeconomic sentiment.
        
        Args:
            cpi_analysis: CPI impact analysis results
            fomc_analysis: FOMC impact analysis results
            calendar_analysis: Economic calendar analysis results
            
        Returns:
            Dictionary with macro sentiment analysis
        """
        sentiment_score = 0
        sentiment_factors = {}
        
        # CPI sentiment contribution
        cpi_score = 0
        if cpi_analysis["current_regime"] == "inflationary":
            cpi_score = -0.3  # Inflation typically negative for risk assets
        elif cpi_analysis["current_regime"] == "deflationary":
            cpi_score = 0.2   # Mild deflation can be positive
        
        # Adjust based on market reaction patterns
        if "1d_return" in cpi_analysis.get("avg_impact", {}):
            avg_return = cpi_analysis["avg_impact"]["1d_return"]
            if avg_return > 0:
                cpi_score += 0.1
            elif avg_return < 0:
                cpi_score -= 0.1
        
        sentiment_factors["cpi_contribution"] = cpi_score
        sentiment_score += cpi_score
        
        # FOMC sentiment contribution
        fomc_score = 0
        
        # Check upcoming event impact
        if fomc_analysis.get("upcoming_event"):
            days_until = fomc_analysis["upcoming_event"]["days_until"]
            if days_until <= 3:
                # Increase uncertainty near FOMC
                fomc_score -= 0.2
            elif days_until <= 7:
                fomc_score -= 0.1
        
        # Check volatility impact
        avg_vol_change = fomc_analysis.get("avg_volatility_change", 0)
        if avg_vol_change > 0.1:
            fomc_score -= 0.1  # Higher volatility is negative
        elif avg_vol_change < -0.05:
            fomc_score += 0.1  # Lower volatility is positive
        
        sentiment_factors["fomc_contribution"] = fomc_score
        sentiment_score += fomc_score
        
        # Calendar sentiment contribution
        calendar_score = 0
        
        week_category = calendar_analysis.get("week_category", "low_impact")
        if week_category == "high_impact":
            calendar_score = -0.2  # High event density increases uncertainty
        elif week_category == "moderate_impact":
            calendar_score = -0.1
        else:
            calendar_score = 0.1   # Low event density is positive
        
        # Adjust for event clustering
        if calendar_analysis.get("event_clustering", 0) > 0.5:
            calendar_score -= 0.1  # Clustered events increase volatility
        
        sentiment_factors["calendar_contribution"] = calendar_score
        sentiment_score += calendar_score
        
        # Determine overall sentiment
        if sentiment_score > 0.3:
            sentiment = "bullish"
        elif sentiment_score > 0.1:
            sentiment = "slightly_bullish"
        elif sentiment_score < -0.3:
            sentiment = "bearish"
        elif sentiment_score < -0.1:
            sentiment = "slightly_bearish"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "sentiment_score": float(sentiment_score),
            "sentiment_factors": sentiment_factors,
            "confidence": min(1.0, abs(sentiment_score) * 2)
        }
    
    @performance_monitor()
    def generate_signal(self, 
                      cpi_analysis: Dict[str, Any],
                      fomc_analysis: Dict[str, Any],
                      calendar_analysis: Dict[str, Any],
                      macro_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined macro events signal.
        
        Args:
            cpi_analysis: CPI impact analysis
            fomc_analysis: FOMC impact analysis
            calendar_analysis: Economic calendar analysis
            macro_sentiment: Macro sentiment analysis
            
        Returns:
            Dictionary with combined signal information
        """
        # Initialize signal components
        signal_components = {}
        
        # 1. CPI Component (range: -0.3 to 0.3)
        cpi_score = 0.0
        
        # Base score from regime
        if cpi_analysis["current_regime"] == "inflationary":
            cpi_score = -0.2
        elif cpi_analysis["current_regime"] == "deflationary":
            cpi_score = 0.1
        elif cpi_analysis["current_regime"] == "transitional":
            cpi_score = -0.05
        
        # Adjust for surprise correlations
        if "1d_return" in cpi_analysis.get("surprise_correlation", {}):
            correlation = cpi_analysis["surprise_correlation"]["1d_return"]
            if not pd.isna(correlation):
                if correlation > 0.5:
                    cpi_score += 0.1  # Positive correlation with surprises
                elif correlation < -0.5:
                    cpi_score -= 0.1  # Negative correlation with surprises
        
        signal_components["cpi"] = max(-0.3, min(0.3, cpi_score))
        
        # 2. FOMC Component (range: -0.3 to 0.3)
        fomc_score = 0.0
        
        # Check for upcoming event
        if fomc_analysis.get("upcoming_event"):
            days_until = fomc_analysis["upcoming_event"]["days_until"]
            if days_until <= 2:
                fomc_score = -0.2  # High uncertainty
            elif days_until <= 5:
                fomc_score = -0.1  # Moderate uncertainty
        
        # Adjust for average volatility change
        avg_vol_change = fomc_analysis.get("avg_volatility_change", 0)
        if avg_vol_change > 0.2:
            fomc_score -= 0.1
        elif avg_vol_change < -0.1:
            fomc_score += 0.1
        
        signal_components["fomc"] = max(-0.3, min(0.3, fomc_score))
        
        # 3. Calendar Component (range: -0.2 to 0.2)
        calendar_score = 0.0
        
        week_category = calendar_analysis.get("week_category", "low_impact")
        if week_category == "high_impact":
            calendar_score = -0.15
        elif week_category == "moderate_impact":
            calendar_score = -0.05
        else:
            calendar_score = 0.1
        
        # Adjust for event clustering
        event_clustering = calendar_analysis.get("event_clustering", 0)
        if event_clustering > 0.7:
            calendar_score -= 0.05
        
        signal_components["calendar"] = max(-0.2, min(0.2, calendar_score))
        
        # 4. Sentiment Component (range: -0.2 to 0.2)
        sentiment_score = macro_sentiment["sentiment_score"] * 0.4  # Scale down
        signal_components["sentiment"] = max(-0.2, min(0.2, sentiment_score))
        
        # Calculate final score
        final_score = sum(signal_components.values())
        
        # Apply dampening during high uncertainty periods
        if week_category == "high_impact" or (fomc_analysis.get("upcoming_event") and fomc_analysis["upcoming_event"]["days_until"] <= 3):
            final_score *= 0.7  # Reduce signal strength
        
        # Cap final score
        final_score = max(-1.0, min(1.0, final_score))
        
        # Determine signal
        signal = "NEUTRAL"
        if final_score > self.signal_threshold:
            signal = "BUY"
        elif final_score > self.signal_threshold * 2:
            signal = "STRONG BUY"
        elif final_score < -self.signal_threshold:
            signal = "SELL"
        elif final_score < -self.signal_threshold * 2:
            signal = "STRONG SELL"
        
        return {
            "signal": signal,
            "score": float(final_score),
            "components": signal_components,
            "confidence": min(1.0, abs(final_score) * 1.5),
            "risk_level": week_category
        }
    
    @performance_monitor()
    def analyze(self, 
               cpi_data: pd.DataFrame,
               fomc_data: pd.DataFrame,
               calendar_data: pd.DataFrame,
               price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete macro events analysis.
        
        Args:
            cpi_data: DataFrame with CPI data
            fomc_data: DataFrame with FOMC data
            calendar_data: DataFrame with economic calendar
            price_data: DataFrame with price history
            
        Returns:
            Dictionary with complete analysis results
        """
        # Analyze CPI impact
        cpi_analysis = self.analyze_cpi_impact(cpi_data, price_data)
        
        # Analyze FOMC impact
        fomc_analysis = self.analyze_fomc_impact(fomc_data, price_data)
        
        # Analyze economic calendar
        calendar_analysis = self.analyze_economic_calendar(calendar_data)
        
        # Calculate macro sentiment
        macro_sentiment = self.calculate_macro_sentiment(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis
        )
        
        # Generate signal
        signal = self.generate_signal(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis,
            macro_sentiment
        )
        
        # Combine all results
        result = {
            "cpi_analysis": cpi_analysis,
            "fomc_analysis": fomc_analysis,
            "calendar_analysis": calendar_analysis,
            "macro_sentiment": macro_sentiment,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "risk_level": signal["risk_level"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def analyze_cpi_impact(cpi_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze CPI impact.
    
    Args:
        cpi_data: DataFrame with CPI data
        price_data: DataFrame with price data
        
    Returns:
        Dictionary with CPI impact analysis
    """
    analyzer = MacroEventsAnalyzer()
    return analyzer.analyze_cpi_impact(cpi_data, price_data)


def analyze_macro_events(cpi_data: pd.DataFrame,
                        fomc_data: pd.DataFrame,
                        calendar_data: pd.DataFrame,
                        price_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze all macro events.
    
    Args:
        cpi_data: DataFrame with CPI data
        fomc_data: DataFrame with FOMC data
        calendar_data: DataFrame with economic calendar
        price_data: DataFrame with price data
        
    Returns:
        Dictionary with complete macro analysis
    """
    analyzer = MacroEventsAnalyzer()
    return analyzer.analyze(cpi_data, fomc_data, calendar_data, price_data)


# Test function
def test_macro_events_analyzer():
    """Test the Macro Events analyzer."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample CPI data
    cpi_dates = pd.date_range(end=datetime.now(), periods=12, freq='MS')
    cpi_data = pd.DataFrame({
        "date": cpi_dates,
        "actual": np.random.uniform(2.0, 4.0, 12),
        "forecast": np.random.uniform(2.0, 4.0, 12),
        "previous": np.random.uniform(2.0, 4.0, 12)
    })
    
    # Generate sample FOMC data
    fomc_dates = pd.date_range(end=datetime.now(), periods=8, freq='6W')
    fomc_data = pd.DataFrame({
        "date": fomc_dates,
        "decision": np.random.choice(["Hike", "Hold", "Cut"], 8),
        "statement_sentiment": np.random.choice(["Hawkish", "Neutral", "Dovish"], 8)
    })
    
    # Add future FOMC event
    future_fomc = pd.DataFrame({
        "date": [datetime.now() + timedelta(days=10)],
        "decision": ["Hold"],
        "statement_sentiment": ["Neutral"]
    })
    fomc_data = pd.concat([fomc_data, future_fomc], ignore_index=True)
    
    # Generate sample calendar data
    calendar_dates = []
    events = ["NFP", "GDP", "Retail Sales", "PMI", "Consumer Confidence"]
    for i in range(30):
        calendar_dates.append(datetime.now() - timedelta(days=15) + timedelta(days=i))
    
    calendar_data = pd.DataFrame({
        "date": calendar_dates,
        "event": np.random.choice(events, 30),
        "impact": np.random.choice(["High", "Medium", "Low"], 30, p=[0.3, 0.4, 0.3]),
        "currency": "USD"
    })
    
    # Generate sample price data
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    prices = 40000 + np.cumsum(np.random.normal(0, 500, 365))
    price_data = pd.DataFrame({
        "date": dates,
        "close": prices
    })
    
    # Create analyzer
    analyzer = MacroEventsAnalyzer()
    
    # Analyze data
    result = analyzer.analyze(cpi_data, fomc_data, calendar_data, price_data)
    
    # Print results
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Risk Level: {result['risk_level']}")
    
    print("\nSignal Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.3f}")
    
    print("\nMacro Sentiment:")
    print(f"  Sentiment: {result['macro_sentiment']['sentiment']}")
    print(f"  Score: {result['macro_sentiment']['sentiment_score']:.3f}")
    
    print("\nCPI Analysis:")
    print(f"  Current Regime: {result['cpi_analysis']['current_regime']}")
    print(f"  Average Impact: {result['cpi_analysis']['avg_impact']}")
    
    print("\nFOMC Analysis:")
    if result['fomc_analysis']['upcoming_event']:
        print(f"  Upcoming Event: {result['fomc_analysis']['upcoming_event']['days_until']} days")
    print(f"  Avg Volatility Change: {result['fomc_analysis']['avg_volatility_change']:.3f}")
    
    print("\nCalendar Analysis:")
    print(f"  Week Category: {result['calendar_analysis']['week_category']}")
    print(f"  Total Upcoming Events: {result['calendar_analysis']['total_upcoming']}")
    print(f"  Events per Week: {result['calendar_analysis']['events_per_week']:.2f}")
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Price with CPI events
    plt.subplot(3, 1, 1)
    plt.plot(price_data["date"], price_data["close"], 'b-', alpha=0.7)
    
    # Mark CPI events
    for _, event in cpi_data.iterrows():
        event_date = event["date"]
        if event_date in price_data["date"].values:
            idx = price_data[price_data["date"] == event_date].index[0]
            color = 'r' if event["actual"] > event["forecast"] else 'g'
            plt.scatter(event_date, price_data.iloc[idx]["close"], 
                       color=color, s=100, alpha=0.7, zorder=5)
    
    plt.title("Price Chart with CPI Events (Red: Above Forecast, Green: Below)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: CPI Data
    plt.subplot(3, 1, 2)
    plt.plot(cpi_data["date"], cpi_data["actual"], 'b-', label="Actual", marker='o')
    plt.plot(cpi_data["date"], cpi_data["forecast"], 'r--', label="Forecast", marker='s')
    plt.fill_between(cpi_data["date"], cpi_data["actual"], cpi_data["forecast"], 
                     alpha=0.3, color='gray')
    plt.title("CPI: Actual vs Forecast")
    plt.xlabel("Date")
    plt.ylabel("CPI %")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Event Calendar Heatmap
    plt.subplot(3, 1, 3)
    
    # Create weekly event counts
    weekly_counts = []
    weeks = []
    for i in range(4):
        week_start = datetime.now() - timedelta(days=7*(3-i))
        week_end = week_start + timedelta(days=7)
        week_events = calendar_data[
            (pd.to_datetime(calendar_data["date"]) >= week_start) &
            (pd.to_datetime(calendar_data["date"]) < week_end) &
            (calendar_data["impact"] == "High")
        ]
        weekly_counts.append(len(week_events))
        weeks.append(f"Week {i+1}")
    
    plt.bar(weeks, weekly_counts, color=['green', 'yellow', 'orange', 'red'])
    plt.title("High Impact Events by Week")
    plt.xlabel("Week")
    plt.ylabel("Number of Events")
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_macro_events_analyzer()
