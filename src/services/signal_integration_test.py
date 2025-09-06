"""
Tests for the signal integration service module.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import shutil
import json
import threading

from src.services.signal_integration import SignalIntegrationService


class TestSignalIntegrationService(unittest.TestCase):
    """Test cases for the SignalIntegrationService class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for signals
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock dependencies
        self.mock_market_analysis = MagicMock()
        
        # Create patches
        self.market_analysis_patch = patch('src.services.signal_integration.MarketAnalysisService')
        
        # Start patches
        self.mock_market_analysis_class = self.market_analysis_patch.start()
        
        # Configure mocks
        self.mock_market_analysis_class.return_value = self.mock_market_analysis
        
        # Create service with mocked dependencies
        self.service = SignalIntegrationService(
            signal_dir=self.temp_dir,
            worker_count=1
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop service if running
        if self.service.running:
            self.service.stop()
        
        # Stop patches
        self.market_analysis_patch.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertEqual(self.service.name, "SignalIntegrationService")
        self.assertIsNotNone(self.service.market_analysis)
        self.assertEqual(self.service.signal_dir, self.temp_dir)
        self.assertEqual(len(self.service.signal_sources), 0)  # Not registered until start
        self.assertEqual(len(self.service.signal_processors), 0)  # Not registered until start
    
    def test_start_stop(self):
        """Test service start and stop."""
        # Test start
        self.assertTrue(self.service.start())
        self.assertTrue(self.service.running)
        self.mock_market_analysis.start.assert_called_once()
        self.assertGreater(len(self.service.signal_sources), 0)  # Signal sources registered
        self.assertGreater(len(self.service.signal_processors), 0)  # Signal processors registered
        
        # Test stop
        self.assertTrue(self.service.stop())
        self.assertFalse(self.service.running)
        self.mock_market_analysis.stop.assert_called_once()
    
    def test_register_signal_sources(self):
        """Test registering signal sources."""
        self.service._register_signal_sources()
        
        # Check that all expected sources are registered
        expected_sources = [
            "market_analysis", "technical", "fundamental", "sentiment", "on_chain"
        ]
        
        for source in expected_sources:
            self.assertIn(source, self.service.signal_sources)
    
    def test_register_signal_processors(self):
        """Test registering signal processors."""
        self.service._register_signal_processors()
        
        # Check that all expected processors are registered
        expected_processors = [
            "default", "weighted_average", "majority_vote", "threshold_filter"
        ]
        
        for processor in expected_processors:
            self.assertIn(processor, self.service.signal_processors)
    
    def test_start_update_thread(self):
        """Test starting the update thread."""
        # Mock update_signals
        self.service.update_signals = MagicMock()
        
        # Start update thread
        self.service._start_update_thread()
        
        # Check that thread is running
        self.assertIsNotNone(self.service.update_thread)
        self.assertTrue(self.service.update_thread.is_alive())
        
        # Stop thread
        self.service.running = False
        self.service.update_thread.join(timeout=1.0)
    
    def test_update_signals(self):
        """Test updating signals."""
        # Register signal sources
        self.service._register_signal_sources()
        
        # Mock source functions
        for source_name in self.service.signal_sources:
            setattr(self.service, f"_get_{source_name}_signals", MagicMock(return_value={
                "BTC": {
                    "1d": {
                        "signal": "NEUTRAL",
                        "score": 0.0,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }))
        
        # Register signal processors
        self.service._register_signal_processors()
        
        # Update signals
        signals = self.service.update_signals()
        
        # Check that all sources were called
        for source_name in self.service.signal_sources:
            source_func = getattr(self.service, f"_get_{source_name}_signals")
            source_func.assert_called_once()
        
        # Check that signals were integrated
        self.assertIn("BTC", signals)
    
    def test_get_market_analysis_signals(self):
        """Test getting market analysis signals."""
        # Configure mock market analysis
        self.mock_market_analysis.get_latest_analysis.return_value = {
            "integrated": {
                "signal": "BUY",
                "score": 0.5,
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }
        }
        
        # Register signal sources
        self.service._register_signal_sources()
        
        # Get market analysis signals
        signals = self.service._get_market_analysis_signals()
        
        # Check signals
        self.assertIn("BTC", signals)
        self.assertIn("1h", signals["BTC"])
        self.assertEqual(signals["BTC"]["1h"]["signal"], "BUY")
        self.assertEqual(signals["BTC"]["1h"]["score"], 0.5)
    
    def test_integrate_signals_default(self):
        """Test integrating signals with default processor."""
        # Register signal processors
        self.service._register_signal_processors()
        
        # Create test signals
        signals = {
            "market_analysis": {
                "BTC": {
                    "1d": {
                        "signal": "BUY",
                        "score": 0.5,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            },
            "technical": {
                "BTC": {
                    "1d": {
                        "signal": "SELL",
                        "score": -0.5,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        }
        
        # Integrate signals
        integrated = self.service.integrate_signals(signals, "default")
        
        # Check result
        self.assertIn("BTC", integrated)
        self.assertIn("1d", integrated["BTC"])
        self.assertEqual(integrated["BTC"]["1d"]["signal"], "BUY")
        self.assertEqual(integrated["BTC"]["1d"]["score"], 0.5)
    
    def test_integrate_signals_weighted_average(self):
        """Test integrating signals with weighted average processor."""
        # Register signal processors
        self.service._register_signal_processors()
        
        # Set signal weights
        self.service.signal_weights = {
            "market_analysis": 2.0,
            "technical": 1.0
        }
        
        # Create test signals
        signals = {
            "market_analysis": {
                "BTC": {
                    "1d": {
                        "signal": "BUY",
                        "score": 0.6,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            },
            "technical": {
                "BTC": {
                    "1d": {
                        "signal": "SELL",
                        "score": -0.3,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        }
        
        # Integrate signals
        integrated = self.service.integrate_signals(signals, "weighted_average")
        
        # Check result
        self.assertIn("BTC", integrated)
        self.assertIn("1d", integrated["BTC"])
        
        # Expected score: (0.6 * 2 + -0.3 * 1) / (2 + 1) = 0.3
        self.assertAlmostEqual(integrated["BTC"]["1d"]["score"], 0.3, places=1)
        self.assertEqual(integrated["BTC"]["1d"]["signal"], "BUY")
    
    def test_integrate_signals_majority_vote(self):
        """Test integrating signals with majority vote processor."""
        # Register signal processors
        self.service._register_signal_processors()
        
        # Create test signals
        signals = {
            "market_analysis": {
                "BTC": {
                    "1d": {
                        "signal": "BUY",
                        "score": 0.5,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            },
            "technical": {
                "BTC": {
                    "1d": {
                        "signal": "BUY",
                        "score": 0.4,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            },
            "fundamental": {
                "BTC": {
                    "1d": {
                        "signal": "SELL",
                        "score": -0.3,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        }
        
        # Integrate signals
        integrated = self.service.integrate_signals(signals, "majority_vote")
        
        # Check result
        self.assertIn("BTC", integrated)
        self.assertIn("1d", integrated["BTC"])
        self.assertEqual(integrated["BTC"]["1d"]["signal"], "BUY")  # 2 BUY vs 1 SELL
        self.assertIn("votes", integrated["BTC"]["1d"])
        self.assertEqual(integrated["BTC"]["1d"]["votes"]["BUY"], 2)
        self.assertEqual(integrated["BTC"]["1d"]["votes"]["SELL"], 1)
    
    def test_integrate_signals_threshold_filter(self):
        """Test integrating signals with threshold filter processor."""
        # Register signal processors
        self.service._register_signal_processors()
        
        # Set threshold
        self.service.signal_threshold = 0.4
        
        # Create test signals
        signals = {
            "market_analysis": {
                "BTC": {
                    "1d": {
                        "signal": "BUY",
                        "score": 0.3,  # Below threshold
                        "timestamp": datetime.now().isoformat()
                    }
                },
                "ETH": {
                    "1d": {
                        "signal": "BUY",
                        "score": 0.5,  # Above threshold
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        }
        
        # Integrate signals
        integrated = self.service.integrate_signals(signals, "threshold_filter")
        
        # Check result
        self.assertEqual(integrated["BTC"]["1d"]["signal"], "NEUTRAL")  # Filtered to NEUTRAL
        self.assertTrue(integrated["BTC"]["1d"]["filtered"])
        self.assertEqual(integrated["ETH"]["1d"]["signal"], "BUY")  # Kept as BUY
    
    def test_score_to_signal(self):
        """Test converting score to signal."""
        # Test various scores
        self.assertEqual(self.service._score_to_signal(0.7), "STRONG BUY")
        self.assertEqual(self.service._score_to_signal(0.5), "BUY")
        self.assertEqual(self.service._score_to_signal(0.0), "NEUTRAL")
        self.assertEqual(self.service._score_to_signal(-0.5), "SELL")
        self.assertEqual(self.service._score_to_signal(-0.7), "STRONG SELL")
    
    def test_update_active_signals(self):
        """Test updating active signals."""
        # Create test signals
        signals = {
            "BTC": {
                "1d": {"signal": "BUY", "score": 0.5},
                "4h": {"signal": "SELL", "score": -0.3}
            },
            "ETH": {
                "1d": {"signal": "NEUTRAL", "score": 0.0}
            }
        }
        
        # Update active signals
        self.service._update_active_signals(signals)
        
        # Check active signals
        self.assertEqual(self.service.active_signals, signals)
        self.assertEqual(self.service.signal_metrics["active_signals_count"], 3)
    
    def test_update_signal_history(self):
        """Test updating signal history."""
        # Create test signals
        signals = {
            "BTC": {
                "1d": {"signal": "BUY", "score": 0.5}
            }
        }
        
        # Update signal history
        self.service._update_signal_history(signals)
        
        # Check signal history
        self.assertIn("BTC", self.service.signal_history)
        self.assertIn("1d", self.service.signal_history["BTC"])
        self.assertEqual(len(self.service.signal_history["BTC"]["1d"]), 1)
        self.assertEqual(self.service.signal_history["BTC"]["1d"][0]["signal"], "BUY")
        self.assertEqual(self.service.signal_history["BTC"]["1d"][0]["score"], 0.5)
    
    def test_save_load_signal_history(self):
        """Test saving and loading signal history."""
        # Create test history
        self.service.signal_history = {
            "BTC": {
                "1d": [
                    {"signal": "BUY", "score": 0.5, "timestamp": datetime.now().isoformat()}
                ]
            }
        }
        
        # Save history
        self.service._save_signal_history()
        
        # Clear history
        self.service.signal_history = {}
        
        # Load history
        self.service._load_signal_history()
        
        # Check history
        self.assertIn("BTC", self.service.signal_history)
        self.assertIn("1d", self.service.signal_history["BTC"])
        self.assertEqual(len(self.service.signal_history["BTC"]["1d"]), 1)
        self.assertEqual(self.service.signal_history["BTC"]["1d"][0]["signal"], "BUY")
    
    def test_get_active_signals(self):
        """Test getting active signals."""
        # Set active signals
        self.service.active_signals = {
            "BTC": {
                "1d": {"signal": "BUY", "score": 0.5},
                "4h": {"signal": "SELL", "score": -0.3}
            },
            "ETH": {
                "1d": {"signal": "NEUTRAL", "score": 0.0}
            }
        }
        
        # Test getting all signals
        all_signals = self.service.get_active_signals()
        self.assertEqual(all_signals, self.service.active_signals)
        
        # Test getting signals for specific asset
        btc_signals = self.service.get_active_signals("BTC")
        self.assertEqual(btc_signals, {"BTC": self.service.active_signals["BTC"]})
        
        # Test getting signals for specific timeframe
        timeframe_signals = self.service.get_active_signals(timeframe="1d")
        self.assertEqual(timeframe_signals["BTC"]["1d"], self.service.active_signals["BTC"]["1d"])
        self.assertEqual(timeframe_signals["ETH"]["1d"], self.service.active_signals["ETH"]["1d"])
        
        # Test getting signals for specific asset and timeframe
        specific_signal = self.service.get_active_signals("BTC", "4h")
        self.assertEqual(specific_signal, {"BTC": {"4h": self.service.active_signals["BTC"]["4h"]}})
    
    def test_get_signal_history(self):
        """Test getting signal history."""
        # Create test history
        now = datetime.now()
        yesterday = (now - timedelta(days=1)).isoformat()
        week_ago = (now - timedelta(days=7)).isoformat()
        
        self.service.signal_history = {
            "BTC": {
                "1d": [
                    {"signal": "BUY", "score": 0.5, "timestamp": week_ago},
                    {"signal": "SELL", "score": -0.3, "timestamp": yesterday}
                ]
            }
        }
        
        # Test getting all history
        history = self.service.get_signal_history("BTC", "1d")
        self.assertEqual(len(history), 2)
        
        # Test getting history for specific days
        recent_history = self.service.get_signal_history("BTC", "1d", days=2)
        self.assertEqual(len(recent_history), 1)
        self.assertEqual(recent_history[0]["signal"], "SELL")
    
    def test_register_custom_source(self):
        """Test registering a custom signal source."""
        # Create custom source
        def custom_source():
            return {"BTC": {"1d": {"signal": "BUY", "score": 0.5}}}
        
        # Register custom source
        self.service.register_custom_source("custom", custom_source)
        
        # Check that source was registered
        self.assertIn("custom", self.service.signal_sources)
        self.assertEqual(self.service.signal_sources["custom"], custom_source)
    
    def test_register_custom_processor(self):
        """Test registering a custom signal processor."""
        # Create custom processor
        def custom_processor(signals):
            return {"BTC": {"1d": {"signal": "BUY", "score": 0.5}}}
        
        # Register custom processor
        self.service.register_custom_processor("custom", custom_processor)
        
        # Check that processor was registered
        self.assertIn("custom", self.service.signal_processors)
        self.assertEqual(self.service.signal_processors["custom"], custom_processor)
    
    def test_update_signal_weights(self):
        """Test updating signal weights."""
        # Set initial weights
        self.service.signal_weights = {"market_analysis": 1.0, "technical": 1.0}
        
        # Update weights
        self.service.update_signal_weights({"market_analysis": 2.0, "sentiment": 0.5})
        
        # Check weights
        self.assertEqual(self.service.signal_weights["market_analysis"], 2.0)
        self.assertEqual(self.service.signal_weights["technical"], 1.0)
        self.assertEqual(self.service.signal_weights["sentiment"], 0.5)
    
    def test_check_service_health(self):
        """Test service health check."""
        # Mock health of dependent services
        self.mock_market_analysis.get_health.return_value = {"status": "running"}
        
        # Register sources and processors
        self.service.signal_sources = {"source1": None, "source2": None}
        self.service.signal_processors = {"processor1": None, "processor2": None}
        
        # Check health
        health = self.service._check_service_health()
        
        # Check result
        self.assertIn("market_analysis_health", health)
        self.assertEqual(health["signal_sources"], 2)
        self.assertEqual(health["signal_processors"], 2)
        self.assertIn("signal_metrics", health)


if __name__ == "__main__":
    unittest.main()
