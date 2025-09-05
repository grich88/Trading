import os
import sys
import traceback
import logging
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app_debug")

def main():
    st.set_page_config(page_title="Debug App", layout="wide")
    st.title("Debug App")
    
    try:
        logger.info("Starting imports...")
        
        # Try importing basic libraries
        import numpy as np
        logger.info("numpy imported")
        
        import pandas as pd
        logger.info("pandas imported")
        
        import matplotlib.pyplot as plt
        logger.info("matplotlib imported")
        
        import altair as alt
        logger.info("altair imported")
        
        from datetime import datetime, timedelta
        logger.info("datetime imported")
        
        # Try importing project modules one by one
        try:
            from enhanced_trading_system import EnhancedTradingSystem
            logger.info("EnhancedTradingSystem imported")
        except Exception as e:
            logger.error(f"Error importing EnhancedTradingSystem: {e}")
            st.error(f"Error importing EnhancedTradingSystem: {e}")
            st.code(traceback.format_exc())
        
        try:
            from historical_data_collector import fetch_historical_data, calculate_indicators
            logger.info("historical_data_collector imported")
        except Exception as e:
            logger.error(f"Error importing historical_data_collector: {e}")
            st.error(f"Error importing historical_data_collector: {e}")
            st.code(traceback.format_exc())
        
        try:
            from image_extractors import compute_tv_score_from_text, HAS_TESS, ocr_image_to_text
            logger.info("image_extractors imported")
        except Exception as e:
            logger.error(f"Error importing image_extractors: {e}")
            st.error(f"Error importing image_extractors: {e}")
            st.code(traceback.format_exc())
        
        # Display system information
        st.header("System Information")
        st.write(f"Python version: {sys.version}")
        st.write(f"Working directory: {os.getcwd()}")
        
        # Check if data directories exist
        st.header("Directory Check")
        directories = ["data", "data/configs", "data/backtests", "data/live_signals", "data/ui_exports"]
        for directory in directories:
            exists = os.path.exists(directory)
            st.write(f"{directory}: {'✅ Exists' if exists else '❌ Missing'}")
            if not exists:
                os.makedirs(directory, exist_ok=True)
                st.write(f"Created {directory}")
        
        # Try loading a simple DataFrame
        st.header("Data Loading Test")
        try:
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=10),
                'close': np.random.randn(10).cumsum() + 100,
                'volume': np.random.randint(1000, 5000, 10)
            })
            df.set_index('date', inplace=True)
            
            # Try plotting with matplotlib
            fig, ax = plt.subplots()
            ax.plot(df.index, df['close'])
            ax.set_title('Test Plot')
            st.pyplot(fig)
            
            # Try plotting with altair
            chart = alt.Chart(df.reset_index()).mark_line().encode(
                x='date:T',
                y='close:Q'
            ).properties(
                title='Test Altair Chart'
            )
            st.altair_chart(chart, use_container_width=True)
            
            st.success("Data loading and plotting successful!")
        except Exception as e:
            logger.error(f"Error in data loading test: {e}")
            st.error(f"Error in data loading test: {e}")
            st.code(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        st.error(f"Unhandled exception: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
