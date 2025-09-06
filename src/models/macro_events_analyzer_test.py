"""
Tests for the CPI / Macro Events Tracking module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.macro_events_analyzer import (
    MacroEventsAnalyzer,
    analyze_cpi_impact,
    analyze_macro_events
)


class TestMacroEventsAnalyzer(unittest.TestCase):
    """Test cases for the MacroEventsAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = MacroEventsAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        
        # Generate CPI data
        cpi_dates = pd.date_range(end=datetime.now(), periods=12, freq='MS')
        self.cpi_data = pd.DataFrame({
            "date": cpi_dates,
            "actual": np.random.uniform(2.0, 4.0, 12),
            "forecast": np.random.uniform(2.0, 4.0, 12),
            "previous": np.random.uniform(2.0, 4.0, 12)
        })
        
        # Generate FOMC data
        fomc_dates = pd.date_range(end=datetime.now(), periods=8, freq='6W')
        self.fomc_data = pd.DataFrame({
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
        self.fomc_data = pd.concat([self.fomc_data, future_fomc], ignore_index=True)
        
        # Generate calendar data
        calendar_dates = []
        events = ["NFP", "GDP", "Retail Sales", "PMI", "Consumer Confidence"]
        for i in range(30):
            calendar_dates.append(datetime.now() - timedelta(days=15) + timedelta(days=i))
        
        self.calendar_data = pd.DataFrame({
            "date": calendar_dates,
            "event": np.random.choice(events, 30),
            "impact": np.random.choice(["High", "Medium", "Low"], 30, p=[0.3, 0.4, 0.3]),
            "currency": "USD"
        })
        
        # Generate price data
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        prices = 40000 + np.cumsum(np.random.normal(0, 500, 365))
        self.price_data = pd.DataFrame({
            "date": dates,
            "close": prices
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = MacroEventsAnalyzer()
        self.assertEqual(analyzer.lookback_period, 20)  # Default lookback period
        
        # Test custom initialization
        custom_analyzer = MacroEventsAnalyzer(
            lookback_period=15,
            signal_threshold=0.4,
            cpi_window=5,
            fomc_window=4
        )
        self.assertEqual(custom_analyzer.lookback_period, 15)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
        self.assertEqual(custom_analyzer.cpi_window, 5)
        self.assertEqual(custom_analyzer.fomc_window, 4)
    
    def test_analyze_cpi_impact(self):
        """Test CPI impact analysis."""
        # Analyze CPI impact
        cpi_impact = self.analyzer.analyze_cpi_impact(self.cpi_data, self.price_data)
        
        # Check result structure
        self.assertIn("avg_impact", cpi_impact)
        self.assertIn("surprise_correlation", cpi_impact)
        self.assertIn("positive_surprise_impact", cpi_impact)
        self.assertIn("negative_surprise_impact", cpi_impact)
        self.assertIn("impact_events", cpi_impact)
        self.assertIn("current_regime", cpi_impact)
        
        # Check result types
        self.assertIsInstance(cpi_impact["avg_impact"], dict)
        self.assertIsInstance(cpi_impact["surprise_correlation"], dict)
        self.assertIsInstance(cpi_impact["positive_surprise_impact"], dict)
        self.assertIsInstance(cpi_impact["negative_surprise_impact"], dict)
        self.assertIsInstance(cpi_impact["impact_events"], list)
        self.assertIn(cpi_impact["current_regime"], ["inflationary", "deflationary", "transitional", "neutral"])
        
        # Test with missing columns
        bad_df = self.cpi_data.drop(columns=["actual"])
        with self.assertRaises(ModelError):
            self.analyzer.analyze_cpi_impact(bad_df, self.price_data)
        
        # Test standalone function
        standalone_result = analyze_cpi_impact(self.cpi_data, self.price_data)
        self.assertEqual(standalone_result["current_regime"], cpi_impact["current_regime"])
    
    def test_analyze_fomc_impact(self):
        """Test FOMC impact analysis."""
        # Analyze FOMC impact
        fomc_impact = self.analyzer.analyze_fomc_impact(self.fomc_data, self.price_data)
        
        # Check result structure
        self.assertIn("avg_volatility_change", fomc_impact)
        self.assertIn("decision_impacts", fomc_impact)
        self.assertIn("sentiment_impacts", fomc_impact)
        self.assertIn("impact_events", fomc_impact)
        self.assertIn("upcoming_event", fomc_impact)
        
        # Check result types
        self.assertIsInstance(fomc_impact["avg_volatility_change"], float)
        self.assertIsInstance(fomc_impact["decision_impacts"], dict)
        self.assertIsInstance(fomc_impact["sentiment_impacts"], dict)
        self.assertIsInstance(fomc_impact["impact_events"], list)
        
        # Check upcoming event
        if fomc_impact["upcoming_event"]:
            self.assertIn("date", fomc_impact["upcoming_event"])
            self.assertIn("days_until", fomc_impact["upcoming_event"])
            self.assertIsInstance(fomc_impact["upcoming_event"]["days_until"], int)
        
        # Test with missing columns
        bad_df = self.fomc_data.drop(columns=["decision"])
        with self.assertRaises(ModelError):
            self.analyzer.analyze_fomc_impact(bad_df, self.price_data)
        
        # Test decision impacts
        for decision, impact in fomc_impact["decision_impacts"].items():
            self.assertIn("avg_return", impact)
            self.assertIn("avg_volatility_change", impact)
            self.assertIsInstance(impact["avg_return"], dict)
            self.assertIsInstance(impact["avg_volatility_change"], float)
    
    def test_analyze_economic_calendar(self):
        """Test economic calendar analysis."""
        # Analyze economic calendar
        calendar_analysis = self.analyzer.analyze_economic_calendar(self.calendar_data)
        
        # Check result structure
        self.assertIn("upcoming_events", calendar_analysis)
        self.assertIn("events_by_day", calendar_analysis)
        self.assertIn("event_counts", calendar_analysis)
        self.assertIn("event_clustering", calendar_analysis)
        self.assertIn("events_per_week", calendar_analysis)
        self.assertIn("week_category", calendar_analysis)
        self.assertIn("total_upcoming", calendar_analysis)
        
        # Check result types
        self.assertIsInstance(calendar_analysis["upcoming_events"], list)
        self.assertIsInstance(calendar_analysis["events_by_day"], dict)
        self.assertIsInstance(calendar_analysis["event_counts"], dict)
        self.assertIsInstance(calendar_analysis["event_clustering"], float)
        self.assertIsInstance(calendar_analysis["events_per_week"], float)
        self.assertIn(calendar_analysis["week_category"], ["high_impact", "moderate_impact", "low_impact"])
        self.assertIsInstance(calendar_analysis["total_upcoming"], int)
        
        # Check events by day
        self.assertEqual(len(calendar_analysis["events_by_day"]), 7)  # 7 days
        
        # Check value ranges
        self.assertTrue(0 <= calendar_analysis["event_clustering"] <= 1)
        self.assertTrue(calendar_analysis["events_per_week"] >= 0)
        self.assertTrue(calendar_analysis["total_upcoming"] >= 0)
        
        # Test with missing columns
        bad_df = self.calendar_data.drop(columns=["impact"])
        with self.assertRaises(ModelError):
            self.analyzer.analyze_economic_calendar(bad_df)
    
    def test_calculate_macro_sentiment(self):
        """Test macro sentiment calculation."""
        # Get analysis results
        cpi_analysis = self.analyzer.analyze_cpi_impact(self.cpi_data, self.price_data)
        fomc_analysis = self.analyzer.analyze_fomc_impact(self.fomc_data, self.price_data)
        calendar_analysis = self.analyzer.analyze_economic_calendar(self.calendar_data)
        
        # Calculate sentiment
        sentiment = self.analyzer.calculate_macro_sentiment(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis
        )
        
        # Check result structure
        self.assertIn("sentiment", sentiment)
        self.assertIn("sentiment_score", sentiment)
        self.assertIn("sentiment_factors", sentiment)
        self.assertIn("confidence", sentiment)
        
        # Check result types
        self.assertIsInstance(sentiment["sentiment"], str)
        self.assertIsInstance(sentiment["sentiment_score"], float)
        self.assertIsInstance(sentiment["sentiment_factors"], dict)
        self.assertIsInstance(sentiment["confidence"], float)
        
        # Check sentiment factors
        self.assertIn("cpi_contribution", sentiment["sentiment_factors"])
        self.assertIn("fomc_contribution", sentiment["sentiment_factors"])
        self.assertIn("calendar_contribution", sentiment["sentiment_factors"])
        
        # Check value ranges
        self.assertTrue(-1 <= sentiment["sentiment_score"] <= 1)
        self.assertTrue(0 <= sentiment["confidence"] <= 1)
        
        # Check sentiment categories
        self.assertIn(sentiment["sentiment"], [
            "bullish", "slightly_bullish", "neutral", "slightly_bearish", "bearish"
        ])
        
        # Test with different regimes
        cpi_analysis["current_regime"] = "inflationary"
        inflationary_sentiment = self.analyzer.calculate_macro_sentiment(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis
        )
        
        cpi_analysis["current_regime"] = "deflationary"
        deflationary_sentiment = self.analyzer.calculate_macro_sentiment(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis
        )
        
        # Inflationary should be more bearish
        self.assertLess(
            inflationary_sentiment["sentiment_factors"]["cpi_contribution"],
            deflationary_sentiment["sentiment_factors"]["cpi_contribution"]
        )
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Get analysis results
        cpi_analysis = self.analyzer.analyze_cpi_impact(self.cpi_data, self.price_data)
        fomc_analysis = self.analyzer.analyze_fomc_impact(self.fomc_data, self.price_data)
        calendar_analysis = self.analyzer.analyze_economic_calendar(self.calendar_data)
        macro_sentiment = self.analyzer.calculate_macro_sentiment(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis
        )
        
        # Generate signal
        signal = self.analyzer.generate_signal(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis,
            macro_sentiment
        )
        
        # Check result structure
        self.assertIn("signal", signal)
        self.assertIn("score", signal)
        self.assertIn("components", signal)
        self.assertIn("confidence", signal)
        self.assertIn("risk_level", signal)
        
        # Check result types
        self.assertIsInstance(signal["signal"], str)
        self.assertIsInstance(signal["score"], float)
        self.assertIsInstance(signal["components"], dict)
        self.assertIsInstance(signal["confidence"], float)
        self.assertIsInstance(signal["risk_level"], str)
        
        # Check component structure
        self.assertIn("cpi", signal["components"])
        self.assertIn("fomc", signal["components"])
        self.assertIn("calendar", signal["components"])
        self.assertIn("sentiment", signal["components"])
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Test signal dampening during high uncertainty
        calendar_analysis["week_category"] = "high_impact"
        high_impact_signal = self.analyzer.generate_signal(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis,
            macro_sentiment
        )
        
        calendar_analysis["week_category"] = "low_impact"
        low_impact_signal = self.analyzer.generate_signal(
            cpi_analysis,
            fomc_analysis,
            calendar_analysis,
            macro_sentiment
        )
        
        # High impact should have dampened signal
        if abs(high_impact_signal["components"]["calendar"]) > 0:
            self.assertLess(
                abs(high_impact_signal["score"]),
                abs(low_impact_signal["score"]) * 1.5  # Allow some variation
            )
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(
            self.cpi_data,
            self.fomc_data,
            self.calendar_data,
            self.price_data
        )
        
        # Check result structure
        self.assertIn("cpi_analysis", result)
        self.assertIn("fomc_analysis", result)
        self.assertIn("calendar_analysis", result)
        self.assertIn("macro_sentiment", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("risk_level", result)
        self.assertIn("timestamp", result)
        
        # Check result types
        self.assertIsInstance(result["cpi_analysis"], dict)
        self.assertIsInstance(result["fomc_analysis"], dict)
        self.assertIsInstance(result["calendar_analysis"], dict)
        self.assertIsInstance(result["macro_sentiment"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["risk_level"], str)
        self.assertIsInstance(result["timestamp"], str)
        
        # Test standalone function
        standalone_result = analyze_macro_events(
            self.cpi_data,
            self.fomc_data,
            self.calendar_data,
            self.price_data
        )
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with missing price data columns
        bad_price_data = pd.DataFrame({"price": [100, 101, 102]})
        with self.assertRaises(ModelError):
            self.analyzer.analyze_cpi_impact(self.cpi_data, bad_price_data)
        
        # Test with empty data
        empty_cpi = pd.DataFrame(columns=self.cpi_data.columns)
        empty_fomc = pd.DataFrame(columns=self.fomc_data.columns)
        empty_calendar = pd.DataFrame(columns=self.calendar_data.columns)
        
        # Should still work but with neutral results
        result = self.analyzer.analyze(
            empty_cpi,
            empty_fomc,
            empty_calendar,
            self.price_data
        )
        self.assertEqual(result["signal"], "NEUTRAL")
        self.assertEqual(result["score"], 0.0)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with all inflationary CPI
        inflated_cpi = self.cpi_data.copy()
        inflated_cpi["actual"] = inflated_cpi["previous"] + 0.5
        inflated_cpi["forecast"] = inflated_cpi["previous"] + 0.2
        
        cpi_impact = self.analyzer.analyze_cpi_impact(inflated_cpi, self.price_data)
        self.assertEqual(cpi_impact["current_regime"], "inflationary")
        
        # Test with no future FOMC events
        past_fomc = self.fomc_data[pd.to_datetime(self.fomc_data["date"]) < datetime.now()]
        fomc_impact = self.analyzer.analyze_fomc_impact(past_fomc, self.price_data)
        self.assertIsNone(fomc_impact["upcoming_event"])
        
        # Test with all high-impact events
        high_impact_calendar = self.calendar_data.copy()
        high_impact_calendar["impact"] = "High"
        calendar_analysis = self.analyzer.analyze_economic_calendar(high_impact_calendar)
        self.assertEqual(calendar_analysis["week_category"], "high_impact")


if __name__ == "__main__":
    unittest.main()
