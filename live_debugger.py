#!/usr/bin/env python3
"""
Live debugger for trading system and liquidation analysis.
This script provides real-time debugging capabilities with a separate window.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import time
import importlib

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LiveDebugger")

class LiveDebugger:
    """Live debugging tool with GUI window for trading system analysis"""
    
    def __init__(self, title="Live Trading System Debugger"):
        """Initialize the debugger with GUI window"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"live_debug_{self.timestamp}"
        self.message_queue = queue.Queue()
        self.running = True
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create debug log text area
        self.log_label = tk.Label(self.main_frame, text="Debug Log:")
        self.log_label.pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Status frame
        self.status_frame = tk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, expand=False)
        
        self.status_label = tk.Label(self.status_frame, text="Status:")
        self.status_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.status_value = tk.Label(self.status_frame, text="Initializing...")
        self.status_value.pack(side=tk.LEFT)
        
        # Button frame
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, expand=False, pady=(10, 0))
        
        self.save_button = tk.Button(self.button_frame, text="Save Log", command=self.save_log)
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = tk.Button(self.button_frame, text="Clear Log", command=self.clear_log)
        self.clear_button.pack(side=tk.LEFT)
        
        # Start message processing thread
        self.message_thread = threading.Thread(target=self.process_messages)
        self.message_thread.daemon = True
        self.message_thread.start()
        
        # Add initial log message
        self.log("Live debugger started. Run your analysis to see debugging information.")
        
        # Patch modules after initialization
        self.patch_modules()
        
    def on_close(self):
        """Handle window close event"""
        self.running = False
        self.root.destroy()
        
    def save_log(self):
        """Save the current log contents to a file"""
        log_path = os.path.join(self.output_dir, f"debug_log_{datetime.now().strftime('%H%M%S')}.txt")
        with open(log_path, 'w') as f:
            f.write(self.log_text.get(1.0, tk.END))
        self.log(f"Log saved to {log_path}")
        
    def clear_log(self):
        """Clear the log text area"""
        self.log_text.delete(1.0, tk.END)
        self.log("Log cleared")
        
    def log(self, message):
        """Add a message to the log queue"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.message_queue.put(f"[{timestamp}] {message}")
        logger.info(message)
        
    def set_status(self, status):
        """Update the status display"""
        self.message_queue.put(("STATUS", status))
        
    def process_messages(self):
        """Process messages from the queue and update the UI"""
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(block=False)
                    
                    if isinstance(message, tuple) and message[0] == "STATUS":
                        # Update status label
                        self.status_value.config(text=message[1])
                    else:
                        # Add message to log
                        self.log_text.insert(tk.END, message + "\n")
                        self.log_text.see(tk.END)  # Scroll to bottom
                        
                self.root.update_idletasks()  # Process UI events
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing messages: {e}")
                time.sleep(1)
                
    def run(self):
        """Start the main event loop"""
        self.root.mainloop()
        
    def patch_modules(self):
        """Patch all relevant modules for debugging"""
        self.log("Starting to patch modules...")
        success = False
        
        # Try to patch the image extractors
        try:
            self.patch_image_extractors()
            success = True
        except Exception as e:
            self.log(f"Failed to patch image extractors: {str(e)}")
            traceback.print_exc()
            
        # Try to patch the RSI volume model
        try:
            self.patch_rsi_volume_model()
            success = True
        except Exception as e:
            self.log(f"Failed to patch RSI volume model: {str(e)}")
            traceback.print_exc()
            
        if success:
            self.log("Successfully patched one or more modules")
            self.set_status("Ready")
        else:
            self.log("WARNING: Failed to patch any modules. Debugging may be limited.")
            self.set_status("Limited")
            
    def patch_image_extractors(self):
        """Patch the image extractors module for debugging"""
        try:
            # Import the module
            import image_extractors
            
            # Save original functions
            original_extract_clusters = image_extractors.extract_liq_clusters_from_image
            original_auto_detect = image_extractors.auto_detect_heatmap_scale
            
            # Create patched functions
            def patched_extract_clusters(image_bytes, price_top=None, price_bottom=None, token=None, auto_adjust_range=True):
                self.log(f"Extracting clusters from image with price range: {price_top} to {price_bottom}, token: {token}")
                try:
                    result = original_extract_clusters(image_bytes, price_top, price_bottom, token, auto_adjust_range)
                    self.log(f"Extracted {len(result)} clusters")
                    return result
                except Exception as e:
                    self.log(f"ERROR in extract_liq_clusters_from_image: {str(e)}")
                    traceback.print_exc()
                    return []
                    
            def patched_auto_detect(image_bytes):
                self.log("Auto-detecting heatmap scale")
                try:
                    result = original_auto_detect(image_bytes)
                    if result:
                        if len(result) == 3:
                            self.log(f"Detected scale: top={result[0]}, bottom={result[1]}, token={result[2]}")
                        else:
                            self.log(f"Detected scale: top={result[0]}, bottom={result[1]}")
                    else:
                        self.log("Failed to detect scale")
                    return result
                except Exception as e:
                    self.log(f"ERROR in auto_detect_heatmap_scale: {str(e)}")
                    traceback.print_exc()
                    return None
            
            # Apply patches
            image_extractors.extract_liq_clusters_from_image = patched_extract_clusters
            image_extractors.auto_detect_heatmap_scale = patched_auto_detect
            
            self.log("Successfully patched image_extractors module")
            return True
        except Exception as e:
            self.log(f"ERROR patching image_extractors: {str(e)}")
            traceback.print_exc()
            return False
            
    def patch_rsi_volume_model(self):
        """Patch the RSI volume model for debugging"""
        try:
            # Import the module
            import updated_rsi_volume_model
            
            # Get the class
            model_class = updated_rsi_volume_model.EnhancedRsiVolumePredictor
            
            # Save original methods
            original_get_liquidation_score = model_class.get_liquidation_score
            original_get_full_analysis = model_class.get_full_analysis
            original_get_target_prices = model_class.get_target_prices
            
            # Create patched methods with self reference
            def patched_get_liquidation_score(self_obj):
                debugger = self  # Capture the debugger instance
                debugger.log("Starting liquidation score calculation")
                try:
                    if not hasattr(self_obj, 'liquidation_data') or not self_obj.liquidation_data:
                        debugger.log("No liquidation data available")
                        return 0
                        
                    clusters = self_obj.liquidation_data.get('clusters', [])
                    debugger.log(f"Processing {len(clusters)} liquidation clusters")
                    
                    # Continue with original method
                    result = original_get_liquidation_score(self_obj)
                    debugger.log(f"Liquidation score calculated: {result}")
                    return result
                except Exception as e:
                    debugger.log(f"ERROR in get_liquidation_score: {str(e)}")
                    traceback.print_exc()
                    return 0
            
            def patched_get_target_prices(self_obj):
                debugger = self  # Capture the debugger instance
                debugger.log("Getting target prices")
                try:
                    result = original_get_target_prices(self_obj)
                    debugger.log(f"Target prices calculated: {result}")
                    return result
                except Exception as e:
                    debugger.log(f"ERROR in get_target_prices: {str(e)}")
                    traceback.print_exc()
                    return {"TP1": 0, "TP2": 0, "SL": 0}
            
            def patched_get_full_analysis(self_obj):
                debugger = self  # Capture the debugger instance
                debugger.log("Starting full analysis")
                try:
                    result = original_get_full_analysis(self_obj)
                    debugger.log("Full analysis completed successfully")
                    return result
                except Exception as e:
                    debugger.log(f"ERROR in get_full_analysis: {str(e)}")
                    traceback.print_exc()
                    return {"final_score": 0, "signal": "NEUTRAL", "targets": {"TP1": 0, "TP2": 0, "SL": 0}}
            
            # Apply patches
            model_class.get_liquidation_score = patched_get_liquidation_score
            model_class.get_full_analysis = patched_get_full_analysis
            model_class.get_target_prices = patched_get_target_prices
            
            self.log(f"Successfully patched {model_class.__name__}")
            return True
        except Exception as e:
            self.log(f"ERROR patching RSI volume model: {str(e)}")
            traceback.print_exc()
            return False

def main():
    """Main function to start the live debugger"""
    debugger = LiveDebugger()
    debugger.run()

if __name__ == "__main__":
    main()