import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import io
import json
import time
import re
from datetime import datetime, timedelta
from historical_data_collector import fetch_historical_data, calculate_indicators
from enhanced_backtester import EnhancedBacktester
from updated_rsi_volume_model import EnhancedRsiVolumePredictor
from image_extractors import extract_liq_clusters_from_image, auto_detect_heatmap_scale, compute_tv_score_from_text, ocr_image_to_text
from coinglass_api import get_liquidation_heatmap

# Configure logging
logging.basicConfig(level=logging.INFO)

def display_chart_item(item, asset_label):
    st.subheader(f"{asset_label} Chart: {item.get('filename', '?')}")
    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown(f"**📅 Time:** {item.get('timestamp', '').split('T')[0]} {item.get('timestamp', '').split('T')[1][:8] if 'timestamp' in item else ''}")
        # Price info
        if 'price' in item and isinstance(item['price'], dict):
            price_data = item['price']
            st.markdown(f"**💰 Price:** Open: {price_data.get('open', '?')}, High: {price_data.get('high', '?')}, Low: {price_data.get('low', '?')}, Close: {price_data.get('close', '?')}, Change: {price_data.get('change', '?')}")
        # Volume
        if 'volume' in item:
            st.markdown(f"**Volume:** {item['volume']}")
        # Errors
        if 'errors' in item:
            for err in item['errors']:
                st.error(f"Extraction error: {err}")
    with cols[1]:
        # Indicators
        for key in ['webtrend', 'rsi']:
            if key in item:
                st.markdown(f"**{key.upper()}:** {item[key]}")
        # Show extracted text
        if 'extracted_text' in item:
            with st.expander("View Full Text"):
                st.code(item['extracted_text'], language="text")
    st.markdown("---")
# Initialize results to an empty DataFrame at the start of the main function
results = pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch and cache historical data."""
    # Ensure dates are in the correct format
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        df = fetch_historical_data(symbol, timeframe="4h", start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            logging.warning(f"No data fetched for {symbol}")
            # Create a sample DataFrame for testing
            df = create_sample_data()
            return df
        logging.info(f"Data fetched for {symbol}: {df.head()}")
        df = calculate_indicators(df)
        logging.info(f"Indicators calculated for {symbol}: {df.head()}")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        # Create a sample DataFrame for testing
        df = create_sample_data()
        return df

def create_sample_data():
    """Create a sample DataFrame for testing when data fetching fails."""
    try:
        # Create a date range for the past 200 days (increased from 100 to handle EMA100)
        dates = pd.date_range(end=datetime.now(), periods=200, freq='4h')
        
        # Create sample price data
        np.random.seed(42)  # For reproducibility
        close = 50000 + np.cumsum(np.random.normal(0, 500, size=len(dates)))
        high = close * (1 + np.random.uniform(0, 0.02, size=len(dates)))
        low = close * (1 - np.random.uniform(0, 0.02, size=len(dates)))
        open_price = close - np.random.normal(0, 200, size=len(dates))
        volume = np.random.uniform(1000, 5000, size=len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        # Add basic indicators manually to avoid potential issues with calculate_indicators
        df['rsi_raw'] = 50 + np.random.normal(0, 10, size=len(df))  # Random RSI values
        df['rsi_raw'] = df['rsi_raw'].clip(0, 100)  # Clip to valid RSI range
        df['rsi_sma'] = df['rsi_raw'].rolling(3).mean()  # Simple 3-period SMA of RSI
        
        # Add EMA values
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        
        # WebTrend status
        df['webtrend_status'] = ((df['close'] > df['ema20']) & 
                                (df['ema20'] > df['ema50']) & 
                                (df['ema50'] > df['ema100'])).astype(int)
        
        # ATR calculation (simplified)
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        
        # Volume metrics
        df['volume_sma5'] = df['volume'].rolling(5).mean()
        df['volume_sma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma20']
        
        # Price changes and streaks
        df['price_change'] = df['close'].diff()
        df['is_green'] = df['price_change'] > 0
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    except Exception as e:
        logging.error(f"Error creating sample data: {e}")
        # Create a minimal valid DataFrame that won't cause errors
        return create_minimal_valid_dataframe()

def create_minimal_valid_dataframe():
    """Create a minimal valid DataFrame that won't cause errors in the backtester."""
    # Create a small dataset with all required columns
    dates = pd.date_range(end=datetime.now(), periods=10, freq='4h')
    data = {
        'open': [50000] * 10,
        'high': [51000] * 10,
        'low': [49000] * 10,
        'close': [50500] * 10,
        'volume': [1000] * 10,
        'rsi_raw': [50] * 10,
        'rsi_sma': [50] * 10,
        'ema20': [50000] * 10,
        'ema50': [49500] * 10,
        'ema100': [49000] * 10,
        'webtrend_status': [1] * 10,
        'atr': [1000] * 10,
        'volume_sma5': [1000] * 10,
        'volume_sma20': [1000] * 10,
        'volume_ratio': [1.0] * 10,
        'price_change': [0] * 10,
        'is_green': [True] * 10,
        'green_streak': [1] * 10,
        'red_streak': [0] * 10
    }
    return pd.DataFrame(data, index=dates)

def compute_cross_asset_correlations(assets_data):
    """
    Compute cross-asset correlations and lead/lag relationships
    
    Parameters:
    -----------
    assets_data : dict
        Dictionary of DataFrames for each asset
        
    Returns:
    --------
    dict
        Dictionary of correlation metrics
    """
    results = {}
    
    # Get list of assets
    assets = list(assets_data.keys())
    
    # Compute pairwise correlations
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i >= j:  # Skip duplicate pairs and self-correlations
                continue
                
            df1 = assets_data[asset1]
            df2 = assets_data[asset2]
            
            # Ensure both DataFrames have data
            if df1.empty or df2.empty:
                continue
                
            # Align indexes
            df1_aligned, df2_aligned = df1.align(df2, join='inner')
            
            if len(df1_aligned) < 10:  # Need enough data points
                continue
                
            # Calculate price correlation
            price_corr = df1_aligned['close'].corr(df2_aligned['close'])
            
            # Calculate return correlation
            returns1 = df1_aligned['close'].pct_change().dropna()
            returns2 = df2_aligned['close'].pct_change().dropna()
            returns_corr = returns1.corr(returns2)
            
            # Calculate lead/lag relationships (which asset tends to move first)
            lead_lag = {}
            for lag in range(-5, 6):  # Check lags from -5 to +5
                if lag == 0:
                    continue
                    
                if lag > 0:
                    # asset1 leads asset2
                    corr = df1_aligned['close'].iloc[:-lag].corr(df2_aligned['close'].iloc[lag:])
                    lead_lag[lag] = corr
                else:
                    # asset2 leads asset1
                    corr = df1_aligned['close'].iloc[-lag:].corr(df2_aligned['close'].iloc[:lag])
                    lead_lag[lag] = corr
            
            # Find the lag with the highest correlation
            max_lag = max(lead_lag.items(), key=lambda x: abs(x[1]))
            
            # Store results
            pair_key = f"{asset1}_{asset2}"
            results[pair_key] = {
                'price_correlation': price_corr,
                'returns_correlation': returns_corr,
                'lead_lag': max_lag[0],
                'lead_lag_strength': max_lag[1]
            }
    
    return results

def get_asset_type(filename, text):
    """
    Determines the asset type (BTC, SOL, BONK, Other) based on filename and text content.
    This function uses a hierarchical approach for accuracy.
    """
    filename_upper = filename.upper()
    text_upper = text.upper()

    # 1. Filename matching (most reliable)
    if re.search(r"\bBTC\b|BITCOIN", filename_upper): return 'BTC'
    if re.search(r"\bSOL\b|SOLANA", filename_upper): return 'SOL'
    if re.search(r"\bBONK\b", filename_upper): return 'BONK'

    # 2. Keyword matching in text (with exchange context)
    if re.search(r"\b(?:COINBASE|BINANCE|BYBIT|KRAKEN)[: ]*BTC(?:USD|USDT)?\b|\bBTC(?:USD|USDT)?\b|BITCOIN", text_upper): return 'BTC'
    if re.search(r"\b(?:COINBASE|BINANCE|BYBIT|KRAKEN)[: ]*SOL(?:USD|USDT)?\b|\bSOL(?:USD|USDT)?\b|SOLANA", text_upper): return 'SOL'
    if re.search(r"\b(?:COINBASE|BINANCE|BYBIT|KRAKEN)[: ]*BONK(?:USD|USDT)?\b|\bBONK(?:USD|USDT)?\b|0\.0{3,}\d+", text_upper): return 'BONK'

    # 3. Price range heuristics as a fallback
    try:
        numbers = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)', text)
        numeric_values = [float(n.replace(',', '')) for n in numbers]
        if numeric_values:
            max_val = max(numeric_values)
            if max_val > 10000: return 'BTC'
            if 50 <= max_val <= 1000: return 'SOL'
            if max_val < 0.001 and '0.000' in text: return 'BONK'
    except (ValueError, TypeError): pass

    return 'Other'

@st.cache_data(show_spinner=False)
def analyze_uploaded_images(uploaded_files, asset_data):
    """Processes uploaded images to extract text and technical indicators."""
    results = {'candle_charts': [], 'tv_panels': [], 'heatmaps': []}
    
    # Debug info
    print(f"Starting analysis of {len(uploaded_files)} files")
    logging.info(f"Starting analysis of {len(uploaded_files)} files")
    
    for file, category in uploaded_files:
        try:
            print(f"Processing file: {file.name}, category: {category}")
            logging.info(f"Processing file: {file.name}, category: {category}")
            
            file.seek(0)
            img_bytes = file.read()
            
            # Ensure we have image data
            if not img_bytes:
                raise ValueError(f"No image data in file {file.name}")
            
            # Extract text with OCR
            text = ocr_image_to_text(io.BytesIO(img_bytes))
            
            # If OCR fails, use a placeholder but don't stop processing
            if not text:
                print(f"Warning: OCR returned no text for {file.name}")
                logging.warning(f"OCR returned no text for {file.name}")
                text = f"No text extracted from {file.name}"
            
            print(f"Extracted text from {file.name}: {text[:50]}...")
            logging.info(f"Extracted text from {file.name}: {text[:50]}...")
            
            result = {
                'filename': file.name,
                'extracted_text': text,
                'timestamp': datetime.now().isoformat(),
                'processing_details': {
                    'text_length': len(text) if text else 0,
                    'file_size': len(img_bytes),
                    'category': category
                }
            }
            
            # Use the robust get_asset_type function
            result['asset_type'] = get_asset_type(file.name, text)
            print(f"File: {file.name}, Detected Asset: {result['asset_type']}")
            logging.info(f"File: {file.name}, Detected Asset: {result['asset_type']}")
            
            # Add asset type to the result for easier debugging
            result['asset_detection'] = {
                'filename': file.name,
                'detected_asset': result['asset_type'],
                'text_sample': text[:100] if text else "No text"
            }
            
            if category == 'candle_charts' or category == 'tv_panels' or category == 'heatmaps':
                indicators, errors = extract_all_indicators(text)
                result.update(indicators)
                if errors:
                    result['errors'] = errors
            
            results[category].append(result)

        except Exception as e:
            logging.error(f"Error processing file {file.name}: {e}")
            results[category].append({
                'filename': file.name,
                'error': str(e),
                'timestamp': datetime.now().isoformat(), # Add timestamp to error items
                'extracted_text': f"Failed to extract text. Error: {e}" # Add error text
            })
            
    return results

def extract_all_indicators(text):
    """Extract all indicators from OCR text. Returns dict with indicators and errors."""
    indicators = {}
    errors = []
    if not text or len(text) < 10:
        errors.append("Text too short for indicator extraction.")
        return indicators, errors
    text_upper = text.upper()
    # WebTrend
    if "WEBTREND" in text_upper or "TREND" in text_upper:
        if re.search(r"UPTREND|UP TREND|BULLISH", text_upper):
            indicators['webtrend'] = "Uptrend"
        elif re.search(r"DOWNTREND|DOWN TREND|BEARISH", text_upper):
            indicators['webtrend'] = "Downtrend"
        else:
            indicators['webtrend'] = "Neutral"
    else:
        errors.append("WebTrend not detected.")
    # RSI
    rsi_patterns = [
        r"RSI\s*\(?\s*\d+\s*\)?\s*[=:]?\s*(\d+\.?\d*)",
        r"RSI\s*[=:]?\s*(\d+\.?\d*)",
        r"RSI\s*[-:]\s*(\d+\.?\d*)"
    ]
    found_rsi = False
    for pattern in rsi_patterns:
        rsi_match = re.search(pattern, text_upper)
        if rsi_match:
            try:
                val = float(rsi_match.group(1))
                if 0 <= val <= 100:
                    indicators['rsi'] = val
                    found_rsi = True
                    break
            except Exception:
                pass
    if not found_rsi:
        errors.append("RSI not detected.")
    # Volume
    vol_patterns = [
        r"VOL(?:UME)?\s*[=:]?\s*([\d,]+\.?\d*[KMB]?)",
        r"VOL(?:UME)?\s*[-:]\s*([\d,]+\.?\d*[KMB]?)",
        r"VOL(?:UME)?[: ]+([\d,]+\.?\d*[KMB]?)"
    ]
    found_vol = False
    for pattern in vol_patterns:
        vol_match = re.search(pattern, text_upper)
        if vol_match:
            try:
                vol_str = vol_match.group(1).replace(',', '')
                multiplier = 1
                if vol_str.endswith('K'):
                    vol_str = vol_str[:-1]
                    multiplier = 1000
                elif vol_str.endswith('M'):
                    vol_str = vol_str[:-1]
                    multiplier = 1000000
                elif vol_str.endswith('B'):
                    vol_str = vol_str[:-1]
                    multiplier = 1000000000
                indicators['volume'] = float(vol_str) * multiplier
                found_vol = True
                break
            except Exception:
                pass
    if not found_vol:
        errors.append("Volume not detected.")
    # Price
    price_info = {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'change': '0%'}
    ohlc_patterns = [
        r"O[:\s]+([\d,.]+)[\s]+H[:\s]+([\d,.]+)[\s]+L[:\s]+([\d,.]+)[\s]+C[:\s]+([\d,.]+)",
        r"OPEN[:\s]+([\d,.]+)[\s]+HIGH[:\s]+([\d,.]+)[\s]+LOW[:\s]+([\d,.]+)[\s]+CLOSE[:\s]+([\d,.]+)",
        r"O:([\d,.]+)\s+H:([\d,.]+)\s+L:([\d,.]+)\s+C:([\d,.]+)"
    ]
    found_ohlc = False
    for pattern in ohlc_patterns:
        try:
            ohlc_match = re.search(pattern, text_upper)
            if ohlc_match:
                price_info['open'] = float(ohlc_match.group(1).replace(',', ''))
                price_info['high'] = float(ohlc_match.group(2).replace(',', ''))
                price_info['low'] = float(ohlc_match.group(3).replace(',', ''))
                price_info['close'] = float(ohlc_match.group(4).replace(',', ''))
                found_ohlc = True
                break
        except Exception:
            pass
    if not found_ohlc:
        errors.append("OHLC price not detected.")
    # Change
    change_patterns = [
        r"CHANGE\s*:\s*([+\-]?\s*\d+\.\d+\s*%)",
        r"(?:CHANGE|CHG)[:\s]+([-+]?[\d.]+%)",
        r"([-+]?\d+\.?\d*%)[\s]+(?:CHANGE|CHG)",
        r"(?:CHANGE|CHG)[\s]+([-+]?\d+\.?\d*)",
        r"([-+]?\d+\.?\d*%)[\s]+\((?:CHANGE|CHG)\)"
    ]
    found_change = False
    for pattern in change_patterns:
        try:
            change_match = re.search(pattern, text_upper)
            if change_match:
                change_val = change_match.group(1).replace(" ", "")
                if not change_val.endswith('%'):
                    change_val += '%'
                price_info['change'] = change_val
                found_change = True
                break
        except Exception:
            pass
    if not found_change:
        errors.append("Change % not detected.")
    indicators['price'] = price_info
    if price_info['open'] == 0 and price_info['high'] == 0 and price_info['low'] == 0 and price_info['close'] == 0:
        errors.append("Price extraction failed.")
    return indicators, errors

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Comprehensive Trading Model UI", layout="wide")
    
    # Add an initial setup page for inputting information
    with st.sidebar:
        st.header("Initial Setup")
        st.text_input("API Key", "")
        st.text_input("Secret Key", "")
        st.button("Save Settings")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Main Analysis", "Cross-Asset Analysis", "Image Analysis", "Settings"])
    
    with tab4:
        st.header("Settings")
        
        # Asset selection with multi-select
        assets = st.multiselect("Select Assets", ["BTC", "ETH", "SOL", "BONK", "DOGE", "XRP"], default=["BTC", "SOL"])
        primary_asset = st.selectbox("Primary Asset for Analysis", assets, index=0 if "BTC" in assets else 0)
        
        # Time period settings
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            date_mode = st.selectbox("Date Mode", ["Full Period", "Last Year", "Last Month", "Last Week"])
        with col2:
            end_date = st.date_input("End Date", datetime.now())
            timeframe = st.selectbox("Timeframe", ["4h", "1h", "1d"], index=0)
        
        # Backtest parameters
        st.subheader("Backtest Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            window = st.slider("Window Size", 10, 100, 50)
            regime_filter = st.checkbox("Regime Filter", True)
        with col2:
            lookahead = st.slider("Lookahead Period", 1, 10, 5)
            adaptive_cutoff = st.checkbox("Adaptive Cutoff", True)
        with col3:
            cutoff = st.slider("Signal Cutoff", 0.0, 1.0, 0.5)
            conflict_min = st.slider("Conflict Min", 0, 10, 5)
        
        # Advanced settings
        st.subheader("Advanced Settings")
        col1, col2 = st.columns(2)
        with col1:
            fees = st.slider("Fees (%)", 0.0, 1.0, 0.1)
            tp_sl = st.checkbox("TP/SL", True)
        with col2:
            auto_optimize = st.checkbox("Auto Optimize", True)
            weight_tuning = st.checkbox("Weight Tuning", True)
        
        # API settings
        st.subheader("API Settings")
        use_coinglass = st.checkbox("Use Coinglass Heatmap API", False)
    
    # Fetch historical data for all selected assets
    assets_data = {}
    # Add progress bar and status text for data fetching
    with st.spinner("Fetching data..."):
        progress_bar = st.progress(0)
        for i, asset in enumerate(assets):
            assets_data[asset] = get_history(asset, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            progress_bar.progress((i + 1) / len(assets))
    
    # Process uploaded files
    with tab3:
        st.header("🖼️ Image-Based Analysis")
        
        # Add a dashboard summary at the top
        st.markdown("""
        <div style="border:1px solid #4b8bf4; border-radius:5px; padding:15px; background-color:#f0f8ff; margin-bottom:20px;">
            <h3 style="margin-top:0; color:#2c5aa0;">🔍 Analysis Dashboard</h3>
            <p style="margin-bottom:10px;">Upload and analyze trading charts, technical indicators, and market data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a more organized layout with two main columns
        upload_col, analysis_col = st.columns([1, 1])
        
        with upload_col:
            st.markdown("""
            <div style="border:1px solid #4b8bf4; border-radius:5px; padding:10px; background-color:#f0f8ff; margin-bottom:15px;">
                <p style="margin:0; font-weight:bold; color:#2c5aa0;">📊 Upload Trading Data</p>
                <p style="margin:5px 0 0 0; font-size:12px; color:#666;">
                    Upload charts and images for cross-reference analysis
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three upload sections with better styling
            exp1 = st.expander("📈 Candle Charts", expanded=True)
            with exp1:
                candle_charts = st.file_uploader(
                    "Upload candlestick charts", 
                    type=["png", "jpg", "jpeg"], 
                    accept_multiple_files=True, 
                    key="candle_charts",
                    help="Upload candlestick chart screenshots from TradingView or other platforms"
                )
            
            exp2 = st.expander("📊 TradingView Panels", expanded=True)
            with exp2:
                tv_panels = st.file_uploader(
                    "Upload technical analysis panels", 
                    type=["png", "jpg", "jpeg"], 
                    accept_multiple_files=True, 
                    key="tv_panels",
                    help="Upload TradingView technical analysis panel screenshots"
                )
                tv_text = st.text_area(
                    "Or paste TradingView text here", 
                    help="Paste technical analysis text from TradingView here"
                )
            
            exp3 = st.expander("🔥 Liquidation Heatmaps", expanded=True)
            with exp3:
                heatmaps = st.file_uploader(
                    "Upload heatmap images", 
                    type=["png", "jpg", "jpeg"], 
                    accept_multiple_files=True, 
                    key="heatmaps",
                    help="Upload screenshots of liquidation heatmaps from Coinglass or similar platforms"
                )
                webtrend_text = st.text_area(
                    "WebTrend Text (if available)",
                    help="Paste WebTrend indicator data if available"
                )
        
        with analysis_col:
            st.markdown("""
            <div style="border:1px solid #4b8bf4; border-radius:5px; padding:10px; background-color:#f0f8ff; margin-bottom:15px;">
                <p style="margin:0; font-weight:bold; color:#2c5aa0;">🔍 Analysis Controls</p>
                <p style="margin:5px 0 0 0; font-size:12px; color:#666;">
                    Run analysis and view results here
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("Upload images on the left panel, then click the button below to analyze them")
            
            # Show upload status
            uploaded_count = len(candle_charts or []) + len(tv_panels or []) + len(heatmaps or [])
            if uploaded_count > 0:
                st.success(f"✅ {uploaded_count} files uploaded and ready for analysis")
                
                # Add colorful indicators for each file type
                status_cols = st.columns(3)
                with status_cols[0]:
                    if len(candle_charts or []) > 0:
                        st.markdown(f"""
                        <div style="border:1px solid #00cc00; border-radius:5px; padding:5px; background-color:#f0fff0; text-align:center;">
                            <p style="margin:0; font-weight:bold; color:#006600;">📈 {len(candle_charts)} Charts</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="border:1px solid #cccccc; border-radius:5px; padding:5px; background-color:#f5f5f5; text-align:center;">
                            <p style="margin:0; color:#666666;">📈 No Charts</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with status_cols[1]:
                    if len(tv_panels or []) > 0:
                        st.markdown(f"""
                        <div style="border:1px solid #4b8bf4; border-radius:5px; padding:5px; background-color:#f0f8ff; text-align:center;">
                            <p style="margin:0; font-weight:bold; color:#2c5aa0;">📊 {len(tv_panels)} Panels</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="border:1px solid #cccccc; border-radius:5px; padding:5px; background-color:#f5f5f5; text-align:center;">
                            <p style="margin:0; color:#666666;">📊 No Panels</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with status_cols[2]:
                    if len(heatmaps or []) > 0:
                        st.markdown(f"""
                        <div style="border:1px solid #ff9900; border-radius:5px; padding:5px; background-color:#fff8f0; text-align:center;">
                            <p style="margin:0; font-weight:bold; color:#cc6600;">🔥 {len(heatmaps)} Heatmaps</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="border:1px solid #cccccc; border-radius:5px; padding:5px; background-color:#f5f5f5; text-align:center;">
                            <p style="margin:0; color:#666666;">🔥 No Heatmaps</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No files uploaded yet. Please upload some images to analyze.")
            
            # Analysis buttons with progress indicators
            st.markdown("### Analysis Options")
            analyze_button = st.button(
                "🔎 Analyze All Images", 
                key="analyze_button",
                help="Extract data from all uploaded images and run analysis",
                use_container_width=True
            )
        
        # Organize uploaded files
        uploaded_files = {
            'candle_charts': candle_charts,
            'tv_panels': tv_panels,
            'heatmaps': heatmaps
        }
        
        # Analyze uploaded images when the button is clicked
        if analyze_button:
            # Show progress indicator
            st.info("Starting image analysis...")
            
            # Consolidate all uploaded files into a flat list of tuples
            all_files = (
                [(f, 'candle_charts') for f in candle_charts] +
                [(f, 'tv_panels') for f in tv_panels] +
                [(f, 'heatmaps') for f in heatmaps]
            )
            if not all_files:
                st.warning("Please upload at least one image to analyze.")
            else:
                with st.spinner("Analyzing uploaded images..."):
                    image_analysis = analyze_uploaded_images(all_files, assets_data)
                    st.success("Image analysis completed.")
                
                # Enhance the display of analysis results
                if image_analysis:
                    st.subheader("Analysis Results")
                    
                    # Add debug information
                    with st.expander("Debug Information (Click to expand)", expanded=True):
                        st.write("Raw Image Analysis Results:")
                        st.json(image_analysis)
                    
                    # Create tabs for different types of analysis
                    # Create asset-based tabs
                    asset_tabs = st.tabs(["BTC", "SOL", "BONK", "Other"])   
                    
                    # Group charts by asset type for all categories
                    btc_charts = []
                    sol_charts = []
                    bonk_charts = []
                    other_charts = []
                    
                    # For debugging
                    st.write("Processing image analysis results...")
                    
                    # Process candle charts
                    if 'candle_charts' in image_analysis and image_analysis['candle_charts']:
                        # For debugging
                        st.write(f"Found {len(image_analysis['candle_charts'])} candle charts")
                        
                        # Debug summary of detection
                        detected_summary = [
                            f"{item.get('filename','?')} -> {item.get('asset_type','Unknown')}" 
                            for item in image_analysis['candle_charts']
                        ]
                        st.write("Detected asset mapping:")
                        st.write("; ".join(detected_summary))
                        
                        # Categorize charts properly - each chart should only appear in ONE tab
                        for item in image_analysis['candle_charts']:
                            # Use the asset_type that was determined during analysis
                            if 'asset_type' in item:
                                if item['asset_type'] == 'BTC':
                                    btc_charts.append(item)
                                elif item['asset_type'] == 'SOL':
                                    sol_charts.append(item)
                                elif item['asset_type'] == 'BONK':
                                    bonk_charts.append(item)
                                else:  # 'Other' or any other value
                                    other_charts.append(item)
                            else:
                                # Fallback: if no asset_type was set, try to determine from filename
                                filename = item.get('filename', '').upper()
                                
                                if 'BTC' in filename or 'BITCOIN' in filename:
                                    item['asset_type'] = 'BTC'
                                    btc_charts.append(item)
                                elif 'SOL' in filename or 'SOLANA' in filename:
                                    item['asset_type'] = 'SOL'
                                    sol_charts.append(item)
                                elif 'BONK' in filename:
                                    item['asset_type'] = 'BONK'
                                    bonk_charts.append(item)
                                else:
                                    # If we still can't determine, put in Other
                                    item['asset_type'] = 'Other'
                                    other_charts.append(item)
                            
                            # BTC Tab
                            with asset_tabs[0]:
                                if btc_charts:
                                    for item in btc_charts:
                                        with st.container():
                                            st.subheader(f"Bitcoin Chart: {item['filename']}")
                                            cols = st.columns([1, 2])
                                            
                                            with cols[0]:
                                                st.markdown(f"**📅 Time:** {item['timestamp'].split('T')[0]} {item['timestamp'].split('T')[1][:8]}")
                                                
                                                # Add a clear price information box
                                                if 'price' in item and isinstance(item['price'], dict):
                                                    price_data = item['price']
                                                    price_html = "<div style='border:1px solid #f7931a; border-radius:5px; padding:10px; background-color:#fff8f0; margin-top:10px;'>"
                                                    price_html += "<p style='margin:0; font-weight:bold; color:#f7931a;'>💰 Price Information:</p>"
                                                    
                                                    # Format each price component
                                                    if 'open' in price_data:
                                                        price_html += f"<p style='margin:2px 0;'>Open: <b>{price_data['open']}</b></p>"
                                                    if 'high' in price_data:
                                                        price_html += f"<p style='margin:2px 0;'>High: <span style='color:#00cc00; font-weight:bold;'>{price_data['high']}</span></p>"
                                                    if 'low' in price_data:
                                                        price_html += f"<p style='margin:2px 0;'>Low: <span style='color:#cc0000; font-weight:bold;'>{price_data['low']}</span></p>"
                                                    if 'close' in price_data:
                                                        price_html += f"<p style='margin:2px 0;'>Close: <b>{price_data['close']}</b></p>"
                                                    if 'change' in price_data:
                                                        change_color = "#00cc00" if "+" in str(price_data['change']) else "#cc0000"
                                                        price_html += f"<p style='margin:2px 0;'>Change: <span style='color:{change_color}; font-weight:bold;'>{price_data['change']}</span></p>"
                                                    
                                                    price_html += "</div>"
                                                    st.markdown(price_html, unsafe_allow_html=True)
                                                    
                                                # Display volume if available
                                                if 'volume' in item:
                                                    vol_val = item['volume']
                                                    
                                                    # Format volume for better readability
                                                    formatted_vol = vol_val
                                                    if isinstance(vol_val, (int, float)):
                                                        if vol_val >= 1_000_000:
                                                            formatted_vol = f"{vol_val/1_000_000:.2f}M"
                                                        elif vol_val >= 1_000:
                                                            formatted_vol = f"{vol_val/1_000:.2f}K"
                                                    
                                                    st.markdown(f"""
                                                    <div style="border:1px solid #4b8bf4; border-radius:5px; padding:10px; background-color:#f9f9f9; margin:10px 0;">
                                                        <div style="font-weight:bold; margin-bottom:5px;">Volume</div>
                                                        <div style="text-align:center; margin-top:5px;">
                                                            <span style="font-size:16px; font-weight:bold; color:#4b8bf4;">{formatted_vol}</span>
                                                        </div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                
                                                # Extract price if available
                                                if 'extracted_text' in item:
                                                    text = item['extracted_text']
                                                    # Try to find price information
                                                    try:
                                                        # Look for price patterns in the text
                                                        price_info = ""
                                                        
                                                        # Pattern for COINBASE:BTCUSD
                                                        if "COINBASE:BTCUSD" in text or "BTCUSD" in text or "@COINBASE" in text:
                                                            price_lines = [line for line in text.split('\n') if 
                                                                          ("BTCUSD" in line or "COINBASE" in line) and 
                                                                          any(s in line for s in ["O", "H", "L", "C", "Vol"])]
                                                            
                                                            if price_lines:
                                                                # Clean up and format the price information
                                                                price_info = price_lines[0].strip()
                                                                # Try to extract actual price values
                                                                price_values = {}
                                                                
                                                                # Look for open, high, low, close patterns
                                                                import re
                                                                o_match = re.search(r'[OoQ][ ]*[0-9,.]+', text)
                                                                h_match = re.search(r'[Hh][ ]*[0-9,.]+', text) 
                                                                l_match = re.search(r'[Ll][ ]*[0-9,.]+', text)
                                                                c_match = re.search(r'[Cc€eE][ ]*[0-9,.]+', text)
                                                                
                                                                if o_match: price_values['Open'] = o_match.group(0)
                                                                if h_match: price_values['High'] = h_match.group(0)
                                                                if l_match: price_values['Low'] = l_match.group(0)
                                                                if c_match: price_values['Close'] = c_match.group(0)
                                                                
                                                                if price_values:
                                                                    st.markdown(f"""
                                                                    <div style="border:1px solid #00cc00; border-radius:5px; padding:10px; background-color:#f0fff0;">
                                                                        <p style="margin:0; font-weight:bold; color:#006600;">💰 Price Information:</p>
                                                                        <p style="margin:0; font-size:18px;">
                                                                            {' | '.join([f"{k}: {v}" for k, v in price_values.items()])}
                                                                        </p>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)
                                                                else:
                                                                    st.markdown(f"**💰 Price Data:** ```{price_info}```")
                                                        
                                                        # Look for percentage change
                                                        pct_match = re.search(r'[+\-][0-9.]+%', text)
                                                        if pct_match:
                                                            pct = pct_match.group(0)
                                                            color = "#00cc00" if "+" in pct else "#cc0000"
                                                            st.markdown(f"""
                                                            <div style="margin-top:10px;">
                                                                <span style="font-weight:bold;">Change: </span>
                                                                <span style="color:{color}; font-weight:bold;">{pct}</span>
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                                    except Exception as e:
                                                        st.markdown(f"<small>Error parsing price: {str(e)}</small>", unsafe_allow_html=True)
                                            
                                            with cols[1]:
                                                # Create a technical indicators card
                                                st.markdown("""
                                                <div style="border:1px solid #f7931a; border-radius:5px; padding:10px; background-color:#fff8f0; margin-bottom:15px;">
                                                    <p style="margin:0; font-weight:bold; color:#f7931a;">📊 Technical Indicators</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # Debug information
                                                st.text(f"Available keys in this chart data: {list(item.keys())}")
                                                
                                                # Extract WebTrend manually from the text if it's not already in the indicators
                                                if 'webtrend' not in item and 'extracted_text' in item:
                                                    text = item['extracted_text'].upper()
                                                    if 'WEBTREND' in text or 'WEB TREND' in text:
                                                        import re
                                                        # Find uptrend/downtrend patterns
                                                        if re.search(r"UPTREND", text) or re.search(r"UP TREND", text) or "UPTREND" in text or "UP TREND" in text:
                                                            item['webtrend'] = "Uptrend"
                                                        elif re.search(r"DOWNTREND", text) or re.search(r"DOWN TREND", text) or "DOWNTREND" in text or "DOWN TREND" in text:
                                                            item['webtrend'] = "Downtrend"
                                                
                                                # Display RSI if available
                                                if 'rsi' in item:
                                                    rsi_val = item['rsi']
                                                    if isinstance(rsi_val, (int, float)):
                                                        # Determine RSI status and color
                                                        if rsi_val > 70:
                                                            rsi_color = "#cc0000"
                                                            rsi_status = "Overbought"
                                                            rsi_desc = "Potential reversal/pullback"
                                                        elif rsi_val < 30:
                                                            rsi_color = "#00cc00"
                                                            rsi_status = "Oversold"
                                                            rsi_desc = "Potential reversal/bounce"
                                                        elif rsi_val > 60:
                                                            rsi_color = "#ff9900"
                                                            rsi_status = "Bullish Momentum"
                                                            rsi_desc = "Strong upward momentum"
                                                        elif rsi_val < 40:
                                                            rsi_color = "#ff9900"
                                                            rsi_status = "Bearish Momentum"
                                                            rsi_desc = "Strong downward momentum"
                                                        else:
                                                            rsi_color = "#666666"
                                                            rsi_status = "Neutral"
                                                            rsi_desc = "No strong momentum bias"
                                                        
                                                        # Create a simple RSI display
                                                        col1, col2 = st.columns([1, 1])
                                                        with col1:
                                                            st.metric("RSI", f"{rsi_val:.2f}")
                                                        with col2:
                                                            if rsi_val > 70:
                                                                st.error(f"Overbought: {rsi_status}")
                                                            elif rsi_val < 30:
                                                                st.success(f"Oversold: {rsi_status}")
                                                            else:
                                                                st.info(f"Status: {rsi_status}")
                                                
                                                # Display WebTrend if available or try to detect from text
                                                trend_value = None
                                                if 'webtrend' in item:
                                                    trend_value = item['webtrend']
                                                elif 'extracted_text' in item:
                                                    # Try to extract WebTrend from text
                                                    text = item['extracted_text'].upper()
                                                    if "UPTREND" in text or "UP TREND" in text:
                                                        trend_value = "Uptrend"
                                                    elif "DOWNTREND" in text or "DOWN TREND" in text:
                                                        trend_value = "Downtrend"
                                                
                                                if trend_value and isinstance(trend_value, str):
                                                    # Determine trend status and color
                                                    if "up" in trend_value.lower():
                                                        trend_color = "#00cc00"
                                                        trend_status = "Uptrend"
                                                        trend_icon = "📈"
                                                        trend_desc = "Market in bullish trend"
                                                    elif "down" in trend_value.lower():
                                                        trend_color = "#cc0000"
                                                        trend_status = "Downtrend"
                                                        trend_icon = "📉"
                                                        trend_desc = "Market in bearish trend"
                                                    else:
                                                        trend_color = "#ff9900"
                                                        trend_status = "Neutral"
                                                        trend_icon = "⚖️"
                                                        trend_desc = "No clear trend direction"
                                                        
                                                    # Create WebTrend visualization using simpler approach
                                                    if trend_status == "Uptrend":
                                                        st.success(f"📈 **WebTrend: {trend_status}** - {trend_desc}")
                                                    elif trend_status == "Downtrend":
                                                        st.error(f"📉 **WebTrend: {trend_status}** - {trend_desc}")
                                                    else:
                                                        st.warning(f"⚖️ **WebTrend: {trend_status}** - {trend_desc}")
                                                else:
                                                    # Show default WebTrend indicator when none is detected
                                                    st.info("⚖️ **WebTrend: Not Detected** - WebTrend indicator not found in chart")
                                                
                                                # Display Moving Averages if available
                                                if 'moving_averages' in item and isinstance(item['moving_averages'], dict):
                                                    ma_data = item['moving_averages']
                                                    if ma_data:
                                                        st.markdown("<p style='font-weight:bold;'>Moving Averages:</p>", unsafe_allow_html=True)
                                                        
                                                        # Create a responsive grid layout for MAs
                                                        num_mas = len(ma_data)
                                                        ma_cols = st.columns(min(3, num_mas))
                                                        
                                                        for i, (ma_name, ma_value) in enumerate(ma_data.items()):
                                                            with ma_cols[i % len(ma_cols)]:
                                                                st.markdown(f"""
                                                                <div style="border:1px solid #ccc; border-radius:5px; padding:5px; text-align:center; margin-bottom:5px;">
                                                                    <div style="font-size:12px; color:#666;">{ma_name}</div>
                                                                    <div style="font-weight:bold;">{ma_value}</div>
                                                                </div>
                                                                """, unsafe_allow_html=True)
                                                
                                                # Try to extract additional indicators from text
                                                if 'extracted_text' in item:
                                                    text = item['extracted_text']
                                                    indicators = {}
                                                    
                                                    # More comprehensive indicator extraction
                                                    import re
                                                    
                                                    # Look for RSI
                                                    if "RSI" in text:
                                                        try:
                                                            # Match RSI followed by a number
                                                            rsi_match = re.search(r'RSI[^0-9]*([0-9.]+)', text)
                                                            if rsi_match:
                                                                rsi_value = rsi_match.group(1)
                                                                indicators["RSI"] = rsi_value
                                                            else:
                                                                rsi_lines = [line for line in text.split('\n') if "RSI" in line and any(c.isdigit() for c in line)]
                                                                if rsi_lines:
                                                                    indicators["RSI"] = rsi_lines[0].strip()
                                                        except Exception as e:
                                                            pass
                                                    
                                                    # Look for various Moving Averages
                                                    ma_types = ["EMA", "SMA", "WMA", "HMA"]
                                                    ma_indicators = {}
                                                    
                                                    for ma_type in ma_types:
                                                        if ma_type in text:
                                                            try:
                                                                # Look for patterns like "EMA(20)" or "EMA 20"
                                                                ma_matches = re.findall(rf'{ma_type}[^0-9]*(\d+)[^0-9]*', text)
                                                                for match in ma_matches:
                                                                    ma_indicators[f"{ma_type}({match})"] = True
                                                            except:
                                                                pass
                                                    
                                                    if ma_indicators:
                                                        indicators["Moving Averages"] = ", ".join(ma_indicators.keys())
                                                    
                                                    # Look for WebTrend status
                                                    if "WebTrend" in text or "Uptrend" in text or "Downtrend" in text:
                                                        try:
                                                            # Identify trend direction
                                                            if "Uptrend 1" in text or "Uptrend\n1" in text:
                                                                indicators["Trend"] = "Uptrend"
                                                            elif "Downtrend 1" in text or "Downtrend\n1" in text:
                                                                indicators["Trend"] = "Downtrend"
                                                            else:
                                                                trend_lines = [line for line in text.split('\n') if "trend" in line.lower() and any(c.isdigit() for c in line)]
                                                                if trend_lines:
                                                                    indicators["Trend"] = trend_lines[0].strip()
                                                        except:
                                                            pass
                                                    
                                                    # Look for Volume
                                                    if "Vol" in text:
                                                        try:
                                                            vol_match = re.search(r'Vol[^0-9]*([0-9.]+)', text)
                                                            if vol_match:
                                                                indicators["Volume"] = vol_match.group(1)
                                                        except:
                                                            pass
                                                    
                                                    # Look for MACD
                                                    if "MACD" in text:
                                                        try:
                                                            macd_lines = [line for line in text.split('\n') if "MACD" in line and any(c.isdigit() for c in line)]
                                                            if macd_lines:
                                                                indicators["MACD"] = macd_lines[0].strip()
                                                        except:
                                                            pass
                                                    
                                                    # Display indicators in a nice format if found
                                                    if indicators:
                                                        st.markdown("""
                                                        <div style="border:1px solid #4b8bf4; border-radius:5px; padding:10px; background-color:#f0f8ff; margin-bottom:15px;">
                                                            <p style="margin:0; font-weight:bold; color:#2c5aa0;">📊 Technical Indicators</p>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        
                                                        # Display indicators with enhanced visualization
                                                        for k, v in indicators.items():
                                                            if k == "RSI":
                                                                try:
                                                                    # Convert value to float if it's a number
                                                                    if isinstance(v, str):
                                                                        v = v.strip()
                                                                        # Extract the first number found in the string
                                                                        match = re.search(r'(\d+\.?\d*)', v)
                                                                        if match:
                                                                            rsi_val = float(match.group(1))
                                                                        else:
                                                                            rsi_val = 50  # Default if no number found
                                                                    else:
                                                                        rsi_val = float(v) if isinstance(v, (int, float)) else 50
                                                                    
                                                                    # Determine color and status based on RSI value
                                                                    if rsi_val > 70:
                                                                        rsi_color = "#cc0000"
                                                                        rsi_status = "Overbought"
                                                                        rsi_desc = "Potential reversal/pullback"
                                                                    elif rsi_val < 30:
                                                                        rsi_color = "#00cc00"
                                                                        rsi_status = "Oversold"
                                                                        rsi_desc = "Potential reversal/bounce"
                                                                    elif rsi_val > 60:
                                                                        rsi_color = "#ff9900"
                                                                        rsi_status = "Bullish Momentum"
                                                                        rsi_desc = "Strong upward momentum"
                                                                    elif rsi_val < 40:
                                                                        rsi_color = "#ff9900"
                                                                        rsi_status = "Bearish Momentum"
                                                                        rsi_desc = "Strong downward momentum"
                                                                    else:
                                                                        rsi_color = "#666666"
                                                                        rsi_status = "Neutral"
                                                                        rsi_desc = "No strong momentum bias"
                                                                    
                                                                    # Create an enhanced RSI visualization with gauge
                                                                    st.markdown(f"""
                                                                    <div style="border:1px solid {rsi_color}; border-radius:5px; padding:10px; background-color:rgba({rsi_color.lstrip('#')[:2]}, {rsi_color.lstrip('#')[2:4]}, {rsi_color.lstrip('#')[4:]}, 0.1); margin-bottom:15px;">
                                                                        <div style="display:flex; justify-content:space-between; align-items:center;">
                                                                            <span style="font-weight:bold; font-size:16px;">RSI</span>
                                                                            <span style="color:{rsi_color}; font-weight:bold; font-size:20px;">{rsi_val:.2f}</span>
                                                                        </div>
                                                                        
                                                                        <div style="margin:8px 0; position:relative; height:8px; background-color:#e0e0e0; border-radius:4px; overflow:hidden;">
                                                                            <div style="position:absolute; height:100%; width:30%; left:0; background-color:#00cc00; border-right:1px solid #fff;"></div>
                                                                            <div style="position:absolute; height:100%; width:40%; left:30%; background-color:#666666; border-right:1px solid #fff;"></div>
                                                                            <div style="position:absolute; height:100%; width:30%; left:70%; background-color:#cc0000;"></div>
                                                                            <div style="position:absolute; height:100%; width:2px; left:calc({min(max(rsi_val, 0), 100)}% - 1px); background-color:#000; z-index:1;"></div>
                                                                        </div>
                                                                        
                                                                        <div style="display:flex; justify-content:space-between; font-size:10px; color:#666; margin-bottom:8px;">
                                                                            <span>0</span>
                                                                            <span>30</span>
                                                                            <span>70</span>
                                                                            <span>100</span>
                                                                        </div>
                                                                        
                                                                        <div style="background-color:rgba({rsi_color.lstrip('#')[:2]}, {rsi_color.lstrip('#')[2:4]}, {rsi_color.lstrip('#')[4:]}, 0.2); padding:5px; border-radius:3px;">
                                                                            <span style="font-weight:bold; color:{rsi_color};">{rsi_status}:</span> {rsi_desc}
                                                                        </div>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)
                                                                except Exception as e:
                                                                    # Fall back to simple display if visualization fails
                                                                    st.markdown(f"**RSI:** {v} (Error: {str(e)})")
                                                            elif k == "Trend":
                                                                # Create enhanced WebTrend visualization
                                                                trend_value = str(v).lower()
                                                                
                                                                if "up" in trend_value:
                                                                    trend_color = "#00cc00"
                                                                    trend_status = "Uptrend"
                                                                    trend_icon = "📈"
                                                                    trend_desc = "Market in bullish trend"
                                                                elif "down" in trend_value:
                                                                    trend_color = "#cc0000"
                                                                    trend_status = "Downtrend"
                                                                    trend_icon = "📉"
                                                                    trend_desc = "Market in bearish trend"
                                                                else:
                                                                    trend_color = "#ff9900"
                                                                    trend_status = "Neutral"
                                                                    trend_icon = "⚖️"
                                                                    trend_desc = "No clear trend direction"
                                                                
                                                                st.markdown(f"""
                                                                <div style="border:1px solid {trend_color}; border-radius:5px; padding:10px; background-color:rgba({trend_color.lstrip('#')[:2]}, {trend_color.lstrip('#')[2:4]}, {trend_color.lstrip('#')[4:]}, 0.1); margin-bottom:15px;">
                                                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                                                        <span style="font-weight:bold; font-size:16px;">WebTrend</span>
                                                                        <span style="font-size:24px;">{trend_icon}</span>
                                                                    </div>
                                                                    
                                                                    <div style="text-align:center; margin:10px 0; font-size:20px; font-weight:bold; color:{trend_color};">
                                                                        {trend_status}
                                                                    </div>
                                                                    
                                                                    <div style="background-color:rgba({trend_color.lstrip('#')[:2]}, {trend_color.lstrip('#')[2:4]}, {trend_color.lstrip('#')[4:]}, 0.2); padding:5px; border-radius:3px; text-align:center;">
                                                                        {trend_desc}
                                                                    </div>
                                                                </div>
                                                                """, unsafe_allow_html=True)
                                                            else:
                                                                st.markdown(f"- **{k}:** {v}")
                                            
                                            # Show extracted text in a collapsible section
                                            with st.expander("View Full Text"):
                                                st.code(item['extracted_text'], language="text")
                                            
                                            st.markdown("---")
                                else:
                                    st.info("No Bitcoin charts available")
                            
                            # SOL Tab
                            with asset_tabs[1]:
                                # Debug information for SOL tab
                                st.write(f"SOL charts found: {len(sol_charts)}")
                                if sol_charts:
                                    st.success(f"Found {len(sol_charts)} Solana charts to display")
                                    for item in sol_charts:
                                        # Use the display_chart_item function for consistent display
                                        display_chart_item(item, "Solana")
                                else:
                                    st.info("No Solana charts available")
                            
                            # BONK Tab
                            with asset_tabs[2]:
                                # Debug information for BONK tab
                                st.write(f"BONK charts found: {len(bonk_charts)}")
                                if bonk_charts:
                                    st.success(f"Found {len(bonk_charts)} BONK charts to display")
                                    for item in bonk_charts:
                                        # Use the display_chart_item function for consistent display
                                        display_chart_item(item, "BONK")
                                else:
                                    st.info("No BONK charts available")
                            
                            # Other Tab
                            with asset_tabs[3]:
                                # Debug information for Other tab
                                st.write(f"Other charts found: {len(other_charts)}")
                                if other_charts:
                                    st.success(f"Found {len(other_charts)} other charts to display")
                                    for item in other_charts:
                                        # Use the display_chart_item function for consistent display
                                        display_chart_item(item, "Other")
                                else:
                                    st.info("No other charts available")
                            
                            # Process TV panels
                            if 'tv_panels' in image_analysis and image_analysis['tv_panels']:
                                st.write(f"Found {len(image_analysis['tv_panels'])} TradingView panels")
                                # Process TV panels here
                            
                            # Process heatmaps
                            if 'heatmaps' in image_analysis and image_analysis['heatmaps']:
                                st.write(f"Found {len(image_analysis['heatmaps'])} heatmaps")
                                # Process heatmaps here
                                
                            # Display analysis summary
                            st.subheader("Analysis Summary")
                            st.write("Analysis completed successfully!")

                            # End of analysis section

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Comprehensive Trading Model UI", layout="wide")
                                                            # Match RSI followed by a number
                                                            rsi_match = re.search(r'RSI[^0-9]*([0-9.]+)', text)
                                                            if rsi_match:
                                                                rsi_value = rsi_match.group(1)
                                                                indicators["RSI"] = rsi_value
                                                            else:
                                                                rsi_lines = [line for line in text.split('\n') if "RSI" in line and any(c.isdigit() for c in line)]
                                                                if rsi_lines:
                                                                    indicators["RSI"] = rsi_lines[0].strip()
                                                        except Exception as e:
                                                            pass
                                                    
                                                    # Look for various Moving Averages
                                                    ma_types = ["EMA", "SMA", "WMA", "HMA"]
                                                    ma_indicators = {}
                                                    
                                                    for ma_type in ma_types:
                                                        if ma_type in text:
                                                            try:
                                                                # Look for patterns like "EMA(20)" or "EMA 20"
                                                                ma_matches = re.findall(rf'{ma_type}[^0-9]*(\d+)[^0-9]*', text)
                                                                for match in ma_matches:
                                                                    ma_indicators[f"{ma_type}({match})"] = True
                                                            except:
                                                                pass
                                                    
                                                    if ma_indicators:
                                                        indicators["Moving Averages"] = ", ".join(ma_indicators.keys())
                                                    
                                                    # Look for WebTrend status
                                                    if "WebTrend" in text or "Uptrend" in text or "Downtrend" in text:
                                                        try:
                                                            # Identify trend direction
                                                            if "Uptrend 1" in text or "Uptrend\n1" in text:
                                                                indicators["Trend"] = "Uptrend"
                                                            elif "Downtrend 1" in text or "Downtrend\n1" in text:
                                                                indicators["Trend"] = "Downtrend"
                                                            else:
                                                                trend_lines = [line for line in text.split('\n') if "trend" in line.lower() and any(c.isdigit() for c in line)]
                                                                if trend_lines:
                                                                    indicators["Trend"] = trend_lines[0].strip()
                                                        except:
                                                            pass
                                                    
                                                    # Look for Volume
                                                    if "Vol" in text:
                                                        try:
                                                            vol_match = re.search(r'Vol[^0-9]*([0-9.]+)', text)
                                                            if vol_match:
                                                                indicators["Volume"] = vol_match.group(1)
                                                        except:
                                                            pass
                                                    
                                                    # Look for MACD
                                                    if "MACD" in text:
                                                        try:
                                                            macd_lines = [line for line in text.split('\n') if "MACD" in line and any(c.isdigit() for c in line)]
                                                            if macd_lines:
                                                                indicators["MACD"] = macd_lines[0].strip()
                                                        except:
                                                            pass
                                                    
                                                    # Display indicators in a nice format if found
                                                    if indicators:
                                                        st.markdown("""
                                                        <div style="border:1px solid #4b8bf4; border-radius:5px; padding:10px; background-color:#f0f8ff; margin-bottom:15px;">
                                                            <p style="margin:0; font-weight:bold; color:#2c5aa0;">📊 Technical Indicators</p>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        
                                                        # Display indicators with enhanced visualization
                                                        for k, v in indicators.items():
                                                            if k == "RSI":
                                                                try:
                                                                    # Convert value to float if it's a number
                                                                    if isinstance(v, str):
                                                                        v = v.strip()
                                                                        # Extract the first number found in the string
                                                                        match = re.search(r'(\d+\.?\d*)', v)
                                                                        if match:
                                                                            rsi_val = float(match.group(1))
                                                                        else:
                                                                            rsi_val = 50  # Default if no number found
                                                                    else:
                                                                        rsi_val = float(v) if isinstance(v, (int, float)) else 50
                                                                    
                                                                    # Determine color and status based on RSI value
                                                                    if rsi_val > 70:
                                                                        rsi_color = "#cc0000"
                                                                        rsi_status = "Overbought"
                                                                        rsi_desc = "Potential reversal/pullback"
                                                                    elif rsi_val < 30:
                                                                        rsi_color = "#00cc00"
                                                                        rsi_status = "Oversold"
                                                                        rsi_desc = "Potential reversal/bounce"
                                                                    elif rsi_val > 60:
                                                                        rsi_color = "#ff9900"
                                                                        rsi_status = "Bullish Momentum"
                                                                        rsi_desc = "Strong upward momentum"
                                                                    elif rsi_val < 40:
                                                                        rsi_color = "#ff9900"
                                                                        rsi_status = "Bearish Momentum"
                                                                        rsi_desc = "Strong downward momentum"
                                                                    else:
                                                                        rsi_color = "#666666"
                                                                        rsi_status = "Neutral"
                                                                        rsi_desc = "No strong momentum bias"
                                                                    
                                                                    # Create an enhanced RSI visualization with gauge
                                                                    st.markdown(f"""
                                                                    <div style="border:1px solid {rsi_color}; border-radius:5px; padding:10px; background-color:rgba({rsi_color.lstrip('#')[:2]}, {rsi_color.lstrip('#')[2:4]}, {rsi_color.lstrip('#')[4:]}, 0.1); margin-bottom:15px;">
                                                                        <div style="display:flex; justify-content:space-between; align-items:center;">
                                                                            <span style="font-weight:bold; font-size:16px;">RSI</span>
                                                                            <span style="color:{rsi_color}; font-weight:bold; font-size:20px;">{rsi_val:.2f}</span>
                                                                        </div>
                                                                        
                                                                        <div style="margin:8px 0; position:relative; height:8px; background-color:#e0e0e0; border-radius:4px; overflow:hidden;">
                                                                            <div style="position:absolute; height:100%; width:30%; left:0; background-color:#00cc00; border-right:1px solid #fff;"></div>
                                                                            <div style="position:absolute; height:100%; width:40%; left:30%; background-color:#666666; border-right:1px solid #fff;"></div>
                                                                            <div style="position:absolute; height:100%; width:30%; left:70%; background-color:#cc0000;"></div>
                                                                            <div style="position:absolute; height:100%; width:2px; left:calc({rsi_val}% - 1px); background-color:#000; z-index:1;"></div>
                                                                        </div>
                                                                        
                                                                        <div style="display:flex; justify-content:space-between; font-size:10px; color:#666; margin-bottom:8px;">
                                                                            <span>0</span>
                                                                            <span>30</span>
                                                                            <span>70</span>
                                                                            <span>100</span>
                                                                        </div>
                                                                        
                                                                        <div style="background-color:rgba({rsi_color.lstrip('#')[:2]}, {rsi_color.lstrip('#')[2:4]}, {rsi_color.lstrip('#')[4:]}, 0.2); padding:5px; border-radius:3px;">
                                                                            <span style="font-weight:bold; color:{rsi_color};">{rsi_status}:</span> {rsi_desc}
                                                                        </div>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)
                                                                except Exception as e:
                                                                    # Fall back to simple display if visualization fails
                                                                    st.markdown(f"**RSI:** {v} (Error: {str(e)})")
                                                            elif k == "Trend":
                                                                # Create enhanced WebTrend visualization
                                                                trend_value = str(v).lower()
                                                                
                                                                if "up" in trend_value:
                                                                    trend_color = "#00cc00"
                                                                    trend_status = "Uptrend"
                                                                    trend_icon = "📈"
                                                                    trend_desc = "Market in bullish trend"
                                                                elif "down" in trend_value:
                                                                    trend_color = "#cc0000"
                                                                    trend_status = "Downtrend"
                                                                    trend_icon = "📉"
                                                                    trend_desc = "Market in bearish trend"
                                                                else:
                                                                    trend_color = "#ff9900"
                                                                    trend_status = "Neutral"
                                                                    trend_icon = "⚖️"
                                                                    trend_desc = "No clear trend direction"
                                                                
                                                                st.markdown(f"""
                                                                <div style="border:1px solid {trend_color}; border-radius:5px; padding:10px; background-color:rgba({trend_color.lstrip('#')[:2]}, {trend_color.lstrip('#')[2:4]}, {trend_color.lstrip('#')[4:]}, 0.1); margin-bottom:15px;">
                                                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                                                        <span style="font-weight:bold; font-size:16px;">WebTrend</span>
                                                                        <span style="font-size:24px;">{trend_icon}</span>
                                                                    </div>
                                                                    
                                                                    <div style="text-align:center; margin:10px 0; font-size:20px; font-weight:bold; color:{trend_color};">
                                                                        {trend_status}
                                                                    </div>
                                                                    
                                                                    <div style="background-color:rgba({trend_color.lstrip('#')[:2]}, {trend_color.lstrip('#')[2:4]}, {trend_color.lstrip('#')[4:]}, 0.2); padding:5px; border-radius:3px; text-align:center;">
                                                                        {trend_desc}
                                                                    </div>
                                                                </div>
                                                                """, unsafe_allow_html=True)
                                                            else:
                                                                st.markdown(f"- **{k}:** {v}")
                                            
                                            # Show extracted text in a collapsible section
                                            with st.expander("View Full Text"):
                                                st.code(item['extracted_text'], language="text")
                                            
                                            st.markdown("---")
                                else:
                                    st.info("No Solana charts available")
                            
                            # BONK Tab
                            with asset_tabs[2]:
                                # Debug information for BONK tab
                                st.write(f"BONK charts found: {len(bonk_charts)}")
                                if bonk_charts:
                                    st.success(f"Found {len(bonk_charts)} BONK charts to display")
                                    for item in bonk_charts:
                                        # Use the display_chart_item function for consistent display
                                        display_chart_item(item, "BONK")
                                else:
                                    st.info("No BONK charts available")
                                                    price_data = item['price']
                                                    price_html = "<div style='border:1px solid #ff9900; border-radius:5px; padding:10px; background-color:#fff8f0; margin-top:10px;'>"
                                                    price_html += "<p style='margin:0; font-weight:bold; color:#cc6600;'>💰 Price Information:</p>"
                                                    
                                                    # Format each price component - handle small decimal values for BONK
                                                    if 'open' in price_data:
                                                        open_val = price_data['open']
                                                        if isinstance(open_val, float) and open_val < 0.001:
                                                            price_html += f"<p style='margin:2px 0;'>Open: <b>{open_val:.8f}</b></p>"
                                                        else:
                                                            price_html += f"<p style='margin:2px 0;'>Open: <b>{open_val}</b></p>"
                                                    
                                                    if 'high' in price_data:
                                                        high_val = price_data['high']
                                                        if isinstance(high_val, float) and high_val < 0.001:
                                                            price_html += f"<p style='margin:2px 0;'>High: <span style='color:#00cc00; font-weight:bold;'>{high_val:.8f}</span></p>"
                                                        else:
                                                            price_html += f"<p style='margin:2px 0;'>High: <span style='color:#00cc00; font-weight:bold;'>{high_val}</span></p>"
                                                    
                                                    if 'low' in price_data:
                                                        low_val = price_data['low']
                                                        if isinstance(low_val, float) and low_val < 0.001:
                                                            price_html += f"<p style='margin:2px 0;'>Low: <span style='color:#cc0000; font-weight:bold;'>{low_val:.8f}</span></p>"
                                                        else:
                                                            price_html += f"<p style='margin:2px 0;'>Low: <span style='color:#cc0000; font-weight:bold;'>{low_val}</span></p>"
                                                    
                                                    if 'close' in price_data:
                                                        close_val = price_data['close']
                                                        if isinstance(close_val, float) and close_val < 0.001:
                                                            price_html += f"<p style='margin:2px 0;'>Close: <b>{close_val:.8f}</b></p>"
                                                        else:
                                                            price_html += f"<p style='margin:2px 0;'>Close: <b>{close_val}</b></p>"
                                                    
                                                    if 'change' in price_data:
                                                        change_color = "#00cc00" if "+" in str(price_data['change']) else "#cc0000"
                                                        price_html += f"<p style='margin:2px 0;'>Change: <span style='color:{change_color}; font-weight:bold;'>{price_data['change']}</span></p>"
                                                    
                                                    price_html += "</div>"
                                                    st.markdown(price_html, unsafe_allow_html=True)
                                                
                                                # Extract price if available
                                                if 'extracted_text' in item:
                                                    text = item['extracted_text']
                                                    # Try to find price information
                                                    try:
                                                        if "COINBASE:BONKUSD" in text or "BONKUSD" in text:
                                                            price_line = [line for line in text.split('\n') if "BONKUSD" in line or "COINBASE" in line]
                                                            if price_line:
                                                                st.markdown(f"**💰 Price:** ```{price_line[0]}```")
                                                    except:
                                                        pass
                                            
                                            with cols[1]:
                                                # Try to extract key indicators
                                                if 'extracted_text' in item:
                                                    text = item['extracted_text']
                                                    indicators = {}
                                                    
                                                    # More comprehensive indicator extraction
                                                    import re
                                                    
                                                    # Look for RSI
                                                    if "RSI" in text:
                                                        try:
                                                            # Match RSI followed by a number
                                                            rsi_match = re.search(r'RSI[^0-9]*([0-9.]+)', text)
                                                            if rsi_match:
                                                                rsi_value = rsi_match.group(1)
                                                                indicators["RSI"] = rsi_value
                                                            else:
                                                                rsi_lines = [line for line in text.split('\n') if "RSI" in line and any(c.isdigit() for c in line)]
                                                                if rsi_lines:
                                                                    indicators["RSI"] = rsi_lines[0].strip()
                                                        except Exception as e:
                                                            pass
                                                    
                                                    # Look for various Moving Averages
                                                    ma_types = ["EMA", "SMA", "WMA", "HMA"]
                                                    ma_indicators = {}
                                                    
                                                    for ma_type in ma_types:
                                                        if ma_type in text:
                                                            try:
                                                                # Look for patterns like "EMA(20)" or "EMA 20"
                                                                ma_matches = re.findall(rf'{ma_type}[^0-9]*(\d+)[^0-9]*', text)
                                                                for match in ma_matches:
                                                                    ma_indicators[f"{ma_type}({match})"] = True
                                                            except:
                                                                pass
                                                    
                                                    if ma_indicators:
                                                        indicators["Moving Averages"] = ", ".join(ma_indicators.keys())
                                                    
                                                    # Look for WebTrend status
                                                    if "WebTrend" in text or "Uptrend" in text or "Downtrend" in text:
                                                        try:
                                                            # Identify trend direction
                                                            if "Uptrend 1" in text or "Uptrend\n1" in text:
                                                                indicators["Trend"] = "Uptrend"
                                                            elif "Downtrend 1" in text or "Downtrend\n1" in text:
                                                                indicators["Trend"] = "Downtrend"
                                                            else:
                                                                trend_lines = [line for line in text.split('\n') if "trend" in line.lower() and any(c.isdigit() for c in line)]
                                                                if trend_lines:
                                                                    indicators["Trend"] = trend_lines[0].strip()
                                                        except:
                                                            pass
                                                    
                                                    # Look for Volume
                                                    if "Vol" in text:
                                                        try:
                                                            vol_match = re.search(r'Vol[^0-9]*([0-9.]+)', text)
                                                            if vol_match:
                                                                indicators["Volume"] = vol_match.group(1)
                                                        except:
                                                            pass
                                                    
                                                    # Look for MACD
                                                    if "MACD" in text:
                                                        try:
                                                            macd_lines = [line for line in text.split('\n') if "MACD" in line and any(c.isdigit() for c in line)]
                                                            if macd_lines:
                                                                indicators["MACD"] = macd_lines[0].strip()
                                                        except:
                                                            pass
                                                    
                                                    # Display indicators in a nice format if found
                                                    if indicators:
                                                        st.markdown("""
                                                        <div style="border:1px solid #4b8bf4; border-radius:5px; padding:10px; background-color:#f0f8ff; margin-bottom:15px;">
                                                            <p style="margin:0; font-weight:bold; color:#2c5aa0;">📊 Technical Indicators</p>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        
                                                        # Display indicators with enhanced visualization
                                                        for k, v in indicators.items():
                                                            if k == "RSI":
                                                                try:
                                                                    # Convert value to float if it's a number
                                                                    if isinstance(v, str):
                                                                        v = v.strip()
                                                                        # Extract the first number found in the string
                                                                        match = re.search(r'(\d+\.?\d*)', v)
                                                                        if match:
                                                                            rsi_val = float(match.group(1))
                                                                        else:
                                                                            rsi_val = 50  # Default if no number found
                                                                    else:
                                                                        rsi_val = float(v) if isinstance(v, (int, float)) else 50
                                                                    
                                                                    # Determine color and status based on RSI value
                                                                    if rsi_val > 70:
                                                                        rsi_color = "#cc0000"
                                                                        rsi_status = "Overbought"
                                                                        rsi_desc = "Potential reversal/pullback"
                                                                    elif rsi_val < 30:
                                                                        rsi_color = "#00cc00"
                                                                        rsi_status = "Oversold"
                                                                        rsi_desc = "Potential reversal/bounce"
                                                                    elif rsi_val > 60:
                                                                        rsi_color = "#ff9900"
                                                                        rsi_status = "Bullish Momentum"
                                                                        rsi_desc = "Strong upward momentum"
                                                                    elif rsi_val < 40:
                                                                        rsi_color = "#ff9900"
                                                                        rsi_status = "Bearish Momentum"
                                                                        rsi_desc = "Strong downward momentum"
                                                                    else:
                                                                        rsi_color = "#666666"
                                                                        rsi_status = "Neutral"
                                                                        rsi_desc = "No strong momentum bias"
                                                                    
                                                                    # Create an enhanced RSI visualization with gauge
                                                                    st.markdown(f"""
                                                                    <div style="border:1px solid {rsi_color}; border-radius:5px; padding:10px; background-color:rgba({rsi_color.lstrip('#')[:2]}, {rsi_color.lstrip('#')[2:4]}, {rsi_color.lstrip('#')[4:]}, 0.1); margin-bottom:15px;">
                                                                        <div style="display:flex; justify-content:space-between; align-items:center;">
                                                                            <span style="font-weight:bold; font-size:16px;">RSI</span>
                                                                            <span style="color:{rsi_color}; font-weight:bold; font-size:20px;">{rsi_val:.2f}</span>
                                                                        </div>
                                                                        
                                                                        <div style="margin:8px 0; position:relative; height:8px; background-color:#e0e0e0; border-radius:4px; overflow:hidden;">
                                                                            <div style="position:absolute; height:100%; width:30%; left:0; background-color:#00cc00; border-right:1px solid #fff;"></div>
                                                                            <div style="position:absolute; height:100%; width:40%; left:30%; background-color:#666666; border-right:1px solid #fff;"></div>
                                                                            <div style="position:absolute; height:100%; width:30%; left:70%; background-color:#cc0000;"></div>
                                                                            <div style="position:absolute; height:100%; width:2px; left:calc({rsi_val}% - 1px); background-color:#000; z-index:1;"></div>
                                                                        </div>
                                                                        
                                                                        <div style="display:flex; justify-content:space-between; font-size:10px; color:#666; margin-bottom:8px;">
                                                                            <span>0</span>
                                                                            <span>30</span>
                                                                            <span>70</span>
                                                                            <span>100</span>
                                                                        </div>
                                                                        
                                                                        <div style="background-color:rgba({rsi_color.lstrip('#')[:2]}, {rsi_color.lstrip('#')[2:4]}, {rsi_color.lstrip('#')[4:]}, 0.2); padding:5px; border-radius:3px;">
                                                                            <span style="font-weight:bold; color:{rsi_color};">{rsi_status}:</span> {rsi_desc}
                                                                        </div>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)
                                                                except Exception as e:
                                                                    # Fall back to simple display if visualization fails
                                                                    st.markdown(f"**RSI:** {v} (Error: {str(e)})")
                                                            elif k == "Trend":
                                                                # Create enhanced WebTrend visualization
                                                                trend_value = str(v).lower()
                                                                
                                                                if "up" in trend_value:
                                                                    trend_color = "#00cc00"
                                                                    trend_status = "Uptrend"
                                                                    trend_icon = "📈"
                                                                    trend_desc = "Market in bullish trend"
                                                                elif "down" in trend_value:
                                                                    trend_color = "#cc0000"
                                                                    trend_status = "Downtrend"
                                                                    trend_icon = "📉"
                                                                    trend_desc = "Market in bearish trend"
                                                                else:
                                                                    trend_color = "#ff9900"
                                                                    trend_status = "Neutral"
                                                                    trend_icon = "⚖️"
                                                                    trend_desc = "No clear trend direction"
                                                                
                                                                st.markdown(f"""
                                                                <div style="border:1px solid {trend_color}; border-radius:5px; padding:10px; background-color:rgba({trend_color.lstrip('#')[:2]}, {trend_color.lstrip('#')[2:4]}, {trend_color.lstrip('#')[4:]}, 0.1); margin-bottom:15px;">
                                                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                                                        <span style="font-weight:bold; font-size:16px;">WebTrend</span>
                                                                        <span style="font-size:24px;">{trend_icon}</span>
                                                                    </div>
                                                                    
                                                                    <div style="text-align:center; margin:10px 0; font-size:20px; font-weight:bold; color:{trend_color};">
                                                                        {trend_status}
                                                                    </div>
                                                                    
                                                                    <div style="background-color:rgba({trend_color.lstrip('#')[:2]}, {trend_color.lstrip('#')[2:4]}, {trend_color.lstrip('#')[4:]}, 0.2); padding:5px; border-radius:3px; text-align:center;">
                                                                        {trend_desc}
                                                                    </div>
                                                                </div>
                                                                """, unsafe_allow_html=True)
                                                            else:
                                                                st.markdown(f"- **{k}:** {v}")
                                            
                                            # Show extracted text in a collapsible section
                                            with st.expander("View Full Text"):
                                                st.code(item['extracted_text'], language="text")
                                            
                                            st.markdown("---")
                                else:
                                    st.info("No BONK charts available")
                            
                            # Other Tab
                            with asset_tabs[3]:
                                # Debug information for Other tab
                                st.write(f"Other charts found: {len(other_charts)}")
                                if other_charts:
                                    st.success(f"Found {len(other_charts)} other charts to display")
                                    for item in other_charts:
                                        # Use the display_chart_item function for consistent display
                                        display_chart_item(item, "Other")
                                else:
                                    st.info("No other charts available")
                                                    text = item['extracted_text']
                                                    # Look for asset name
                                                    for asset in ["ETH", "ETHEREUM", "XRP", "RIPPLE", "ADA", "CARDANO", "DOGE", "DOGECOIN"]:
                                                        if asset in text.upper():
                                                            st.markdown(f"**💎 Asset:** {asset}")
                                                            break
                                            
                                            with cols[1]:
                                                # Try to extract key indicators
                                                if 'extracted_text' in item:
                                                    text = item['extracted_text']
                                                    indicators = {}
                                                    
                                                    # More comprehensive indicator extraction
                                                    import re
                                                    
                                                    # Look for RSI
                                                    if "RSI" in text:
                                                        try:
                                                            # Match RSI followed by a number
                                                            rsi_match = re.search(r'RSI[^0-9]*([0-9.]+)', text)
                                                            if rsi_match:
                                                                rsi_value = rsi_match.group(1)
                                                                indicators["RSI"] = rsi_value
                                                            else:
                                                                rsi_lines = [line for line in text.split('\n') if "RSI" in line and any(c.isdigit() for c in line)]
                                                                if rsi_lines:
                                                                    indicators["RSI"] = rsi_lines[0].strip()
                                                        except Exception as e:
                                                            pass
                                                    
                                                    # Look for various Moving Averages
                                                    ma_types = ["EMA", "SMA", "WMA", "HMA"]
                                                    ma_indicators = {}
                                                    
                                                    for ma_type in ma_types:
                                                        if ma_type in text:
                                                            try:
                                                                # Look for patterns like "EMA(20)" or "EMA 20"
                                                                ma_matches = re.findall(rf'{ma_type}[^0-9]*(\d+)[^0-9]*', text)
                                                                for match in ma_matches:
                                                                    ma_indicators[f"{ma_type}({match})"] = True
                                                            except:
                                                                pass
                                                    
                                                    if ma_indicators:
                                                        indicators["Moving Averages"] = ", ".join(ma_indicators.keys())
                                                    
                                                    # Look for WebTrend status
                                                    if "WebTrend" in text or "Uptrend" in text or "Downtrend" in text:
                                                        try:
                                                            # Identify trend direction
                                                            if "Uptrend 1" in text or "Uptrend\n1" in text:
                                                                indicators["Trend"] = "Uptrend"
                                                            elif "Downtrend 1" in text or "Downtrend\n1" in text:
                                                                indicators["Trend"] = "Downtrend"
                                                            else:
                                                                trend_lines = [line for line in text.split('\n') if "trend" in line.lower() and any(c.isdigit() for c in line)]
                                                                if trend_lines:
                                                                    indicators["Trend"] = trend_lines[0].strip()
                                                        except:
                                                            pass
                                                    
                                                    # Look for Volume
                                                    if "Vol" in text:
                                                        try:
                                                            vol_match = re.search(r'Vol[^0-9]*([0-9.]+)', text)
                                                            if vol_match:
                                                                indicators["Volume"] = vol_match.group(1)
                                                        except:
                                                            pass
                                                    
                                                    # Look for MACD
                                                    if "MACD" in text:
                                                        try:
                                                            macd_lines = [line for line in text.split('\n') if "MACD" in line and any(c.isdigit() for c in line)]
                                                            if macd_lines:
                                                                indicators["MACD"] = macd_lines[0].strip()
                                                        except:
                                                            pass
                                                    
                                                    # Display indicators in a nice format if found
                                                    if indicators:
                                                        st.markdown("""
                                                        <div style="border:1px solid #4b8bf4; border-radius:5px; padding:10px; background-color:#f0f8ff; margin-bottom:15px;">
                                                            <p style="margin:0; font-weight:bold; color:#2c5aa0;">📊 Technical Indicators</p>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                        
                                                        # Display indicators with enhanced visualization
                                                        for k, v in indicators.items():
                                                            if k == "RSI":
                                                                try:
                                                                    # Convert value to float if it's a number
                                                                    if isinstance(v, str):
                                                                        v = v.strip()
                                                                        # Extract the first number found in the string
                                                                        match = re.search(r'(\d+\.?\d*)', v)
                                                                        if match:
                                                                            rsi_val = float(match.group(1))
                                                                        else:
                                                                            rsi_val = 50  # Default if no number found
                                                                    else:
                                                                        rsi_val = float(v) if isinstance(v, (int, float)) else 50
                                                                    
                                                                    # Determine color and status based on RSI value
                                                                    if rsi_val > 70:
                                                                        rsi_color = "#cc0000"
                                                                        rsi_status = "Overbought"
                                                                        rsi_desc = "Potential reversal/pullback"
                                                                    elif rsi_val < 30:
                                                                        rsi_color = "#00cc00"
                                                                        rsi_status = "Oversold"
                                                                        rsi_desc = "Potential reversal/bounce"
                                                                    elif rsi_val > 60:
                                                                        rsi_color = "#ff9900"
                                                                        rsi_status = "Bullish Momentum"
                                                                        rsi_desc = "Strong upward momentum"
                                                                    elif rsi_val < 40:
                                                                        rsi_color = "#ff9900"
                                                                        rsi_status = "Bearish Momentum"
                                                                        rsi_desc = "Strong downward momentum"
                                                                    else:
                                                                        rsi_color = "#666666"
                                                                        rsi_status = "Neutral"
                                                                        rsi_desc = "No strong momentum bias"
                                                                    
                                                                    # Create an enhanced RSI visualization with gauge
                                                                    st.markdown(f"""
                                                                    <div style="border:1px solid {rsi_color}; border-radius:5px; padding:10px; background-color:rgba({rsi_color.lstrip('#')[:2]}, {rsi_color.lstrip('#')[2:4]}, {rsi_color.lstrip('#')[4:]}, 0.1); margin-bottom:15px;">
                                                                        <div style="display:flex; justify-content:space-between; align-items:center;">
                                                                            <span style="font-weight:bold; font-size:16px;">RSI</span>
                                                                            <span style="color:{rsi_color}; font-weight:bold; font-size:20px;">{rsi_val:.2f}</span>
                                                                        </div>
                                                                        
                                                                        <div style="margin:8px 0; position:relative; height:8px; background-color:#e0e0e0; border-radius:4px; overflow:hidden;">
                                                                            <div style="position:absolute; height:100%; width:30%; left:0; background-color:#00cc00; border-right:1px solid #fff;"></div>
                                                                            <div style="position:absolute; height:100%; width:40%; left:30%; background-color:#666666; border-right:1px solid #fff;"></div>
                                                                            <div style="position:absolute; height:100%; width:30%; left:70%; background-color:#cc0000;"></div>
                                                                            <div style="position:absolute; height:100%; width:2px; left:calc({rsi_val}% - 1px); background-color:#000; z-index:1;"></div>
                                                                        </div>
                                                                        
                                                                        <div style="display:flex; justify-content:space-between; font-size:10px; color:#666; margin-bottom:8px;">
                                                                            <span>0</span>
                                                                            <span>30</span>
                                                                            <span>70</span>
                                                                            <span>100</span>
                                                                        </div>
                                                                        
                                                                        <div style="background-color:rgba({rsi_color.lstrip('#')[:2]}, {rsi_color.lstrip('#')[2:4]}, {rsi_color.lstrip('#')[4:]}, 0.2); padding:5px; border-radius:3px;">
                                                                            <span style="font-weight:bold; color:{rsi_color};">{rsi_status}:</span> {rsi_desc}
                                                                        </div>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)
                                                                except Exception as e:
                                                                    # Fall back to simple display if visualization fails
                                                                    st.markdown(f"**RSI:** {v} (Error: {str(e)})")
                                                            elif k == "Trend":
                                                                # Create enhanced WebTrend visualization
                                                                trend_value = str(v).lower()
                                                                
                                                                if "up" in trend_value:
                                                                    trend_color = "#00cc00"
                                                                    trend_status = "Uptrend"
                                                                    trend_icon = "📈"
                                                                    trend_desc = "Market in bullish trend"
                                                                elif "down" in trend_value:
                                                                    trend_color = "#cc0000"
                                                                    trend_status = "Downtrend"
                                                                    trend_icon = "📉"
                                                                    trend_desc = "Market in bearish trend"
                                                                else:
                                                                    trend_color = "#ff9900"
                                                                    trend_status = "Neutral"
                                                                    trend_icon = "⚖️"
                                                                    trend_desc = "No clear trend direction"
                                                                
                                                                st.markdown(f"""
                                                                <div style="border:1px solid {trend_color}; border-radius:5px; padding:10px; background-color:rgba({trend_color.lstrip('#')[:2]}, {trend_color.lstrip('#')[2:4]}, {trend_color.lstrip('#')[4:]}, 0.1); margin-bottom:15px;">
                                                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                                                        <span style="font-weight:bold; font-size:16px;">WebTrend</span>
                                                                        <span style="font-size:24px;">{trend_icon}</span>
                                                                    </div>
                                                                    
                                                                    <div style="text-align:center; margin:10px 0; font-size:20px; font-weight:bold; color:{trend_color};">
                                                                        {trend_status}
                                                                    </div>
                                                                    
                                                                    <div style="background-color:rgba({trend_color.lstrip('#')[:2]}, {trend_color.lstrip('#')[2:4]}, {trend_color.lstrip('#')[4:]}, 0.2); padding:5px; border-radius:3px; text-align:center;">
                                                                        {trend_desc}
                                                                    </div>
                                                                </div>
                                                                """, unsafe_allow_html=True)
                                                            else:
                                                                st.markdown(f"- **{k}:** {v}")
                                            
                                            # Show extracted text in a collapsible section
                                            with st.expander("View Full Text"):
                                                st.code(item['extracted_text'], language="text")
                                            
                                            st.markdown("---")
                                else:
                                    st.info("No other charts available")
                        else:
                            st.info("No candle chart analysis available")
                    
                    # TradingView Panels tab
                    analysis_tabs = st.tabs(["Candle Charts", "TradingView Panels", "Heatmaps"])
                    with analysis_tabs[1]:
                        if 'tv_panels' in image_analysis and image_analysis['tv_panels']:
                            # Group panels by asset type
                            btc_panels = []
                            sol_panels = []
                            bonk_panels = []
                            other_panels = []
                            
                            # For debugging
                            st.write(f"Found {len(image_analysis['tv_panels'])} TradingView panels")
                            
                            for item in image_analysis['tv_panels']:
                                # Force-add the item to specific chart sections for visibility
                                if 'asset_type' in item:
                                    if item['asset_type'] == 'BTC':
                                        btc_panels.append(item)
                                    elif item['asset_type'] == 'SOL':
                                        sol_panels.append(item)
                                    elif item['asset_type'] == 'BONK':
                                        bonk_panels.append(item)
                                    elif item['asset_type'] == 'All':
                                        # Add to all categories
                                        btc_panels.append(item)
                                        sol_panels.append(item)
                                        bonk_panels.append(item)
                                        other_panels.append(item)
                                    else:
                                        other_panels.append(item)
                                else:
                                    # If no asset_type, analyze text and add to ALL relevant tabs
                                    text = item.get('extracted_text', '').upper()
                                    filename = item.get('filename', '').upper()
                                    
                                    # Check for BTC indicators - add to BTC panels if found
                                    if 'BITCOIN' in text or 'BTCUSD' in text or 'BTC' in filename or '110' in text:
                                        btc_panels.append(item)
                                    
                                    # Check for SOL indicators - add to SOL panels if found
                                    if 'SOLANA' in text or 'SOLUSD' in text or 'SOL' in filename or '213' in text:
                                        sol_panels.append(item)
                                    
                                    # Check for BONK indicators - add to BONK panels if found
                                    if 'BONK' in text or 'BONKUSD' in text or 'BONK' in filename or '0.0000' in text:
                                        bonk_panels.append(item)
                                    
                                    # If no clear category is found, add to other and to all categories for visibility
                                    if not ('BITCOIN' in text or 'BTCUSD' in text or 'BTC' in filename or
                                            'SOLANA' in text or 'SOLUSD' in text or 'SOL' in filename or
                                            'BONK' in text or 'BONKUSD' in text or 'BONK' in filename):
                                        other_panels.append(item)
                                        # Also add to all to ensure visibility
                                        btc_panels.append(item)
                                        sol_panels.append(item)
                                        bonk_panels.append(item)
                            
                            # Create asset-based tabs
                            panel_tabs = st.tabs(["BTC", "SOL", "BONK", "Other"])
                            
                            # BTC Tab
                            with panel_tabs[0]:
                                if btc_panels:
                                    for item in btc_panels:
                                        display_chart_item(item, "Bitcoin Panel")
                                else:
                                    st.info("No Bitcoin TradingView panels available")
                            
                            # SOL Tab
                            with panel_tabs[1]:
                                if sol_panels:
                                    for item in sol_panels:
                                         display_chart_item(item, "Solana Panel")
                                else:
                                     st.info("No Solana TradingView panels available")
                            
                            # BONK Tab
                            with panel_tabs[2]:
                                if bonk_panels:
                                    for item in bonk_panels:
                                        display_chart_item(item, "BONK Panel")
                                else:
                                    st.info("No BONK TradingView panels available")
                            
                            # Other Tab
                            with panel_tabs[3]:
                                if other_panels:
                                    for item in other_panels:
                                        display_chart_item(item, "Other Panel")
                                else:
                                    st.info("No other TradingView panels available")
                    
                    # Heatmaps tab
                    with analysis_tabs[2]:
                        if 'heatmaps' in image_analysis and image_analysis['heatmaps']:
                            # Group heatmaps by asset type and price range
                            btc_heatmaps = []
                            sol_heatmaps = []
                            bonk_heatmaps = []
                            other_heatmaps = []
                            
                            # For debugging
                            st.write(f"Found {len(image_analysis['heatmaps'])} heatmaps")
                            
                            # Process each heatmap and add to multiple categories as needed
                            for item in image_analysis['heatmaps']:
                                # First try to use the asset_type field if available
                                if 'asset_type' in item:
                                    if item['asset_type'] == 'BTC':
                                        btc_heatmaps.append(item)
                                    elif item['asset_type'] == 'SOL':
                                        sol_heatmaps.append(item)
                                    elif item['asset_type'] == 'BONK':
                                        bonk_heatmaps.append(item)
                                    elif item['asset_type'] == 'All':
                                        # Add to all categories for maximum visibility
                                        btc_heatmaps.append(item)
                                        sol_heatmaps.append(item)
                                        bonk_heatmaps.append(item)
                                        other_heatmaps.append(item)
                                    else:
                                        other_heatmaps.append(item)
                                else:
                                    # No clear asset type, so try multiple categorization methods
                                    added_somewhere = False
                                    
                                    # Check price scale if available
                                    if 'price_scale' in item and 'top' in item['price_scale']:
                                        top_price = item['price_scale']['top']
                                        if top_price > 50000:  # Likely BTC
                                            btc_heatmaps.append(item)
                                            added_somewhere = True
                                        elif 100 < top_price < 1000:  # Likely SOL
                                            sol_heatmaps.append(item)
                                            added_somewhere = True
                                        elif top_price < 1:  # Likely BONK
                                            bonk_heatmaps.append(item)
                                            added_somewhere = True
                                    
                                    # Check text and filename references
                                    text = item.get('extracted_text', '').upper()
                                    filename = item.get('filename', '').upper()
                                    
                                    # Check for BTC references
                                    if 'BITCOIN' in text or 'BTCUSD' in text or 'BTC' in filename or '110' in text:
                                        btc_heatmaps.append(item)
                                        added_somewhere = True
                                    
                                    # Check for SOL references
                                    if 'SOLANA' in text or 'SOLUSD' in text or 'SOL' in filename or '213' in text:
                                        sol_heatmaps.append(item)
                                        added_somewhere = True
                                    
                                    # Check for BONK references
                                    if 'BONK' in text or 'BONKUSD' in text or 'BONK' in filename or '0.0000' in text:
                                        bonk_heatmaps.append(item)
                                        added_somewhere = True
                                    
                                    # If not categorized yet, add to multiple categories for visibility
                                    if not added_somewhere:
                                        st.info(f"Adding heatmap {item.get('filename', 'unknown')} to all asset categories for visibility")
                                        btc_heatmaps.append(item)
                                        sol_heatmaps.append(item)
                                        bonk_heatmaps.append(item)
                                        other_heatmaps.append(item)
                            
                            # Create asset-based tabs
                            heatmap_tabs = st.tabs(["BTC", "SOL", "BONK", "Other"])
                            
                            # BTC Tab
                            with heatmap_tabs[0]:
                                if btc_heatmaps:
                                    for item in btc_heatmaps:
                                        display_chart_item(item, "Bitcoin Heatmap")
                                else:
                                    st.info("No Bitcoin heatmaps available")
                            
                            # SOL Tab
                            with heatmap_tabs[1]:
                                if sol_heatmaps:
                                    for item in sol_heatmaps:
                                        display_chart_item(item, "Solana Heatmap")
                                else:
                                    st.info("No Solana heatmaps available")
                            
                            # BONK Tab
                            with heatmap_tabs[2]:
                                if bonk_heatmaps:
                                    for item in bonk_heatmaps:
                                        display_chart_item(item, "BONK Heatmap")
                                else:
                                    st.info("No BONK heatmaps available")
                            
                            # Other Tab
                            with heatmap_tabs[3]:
                                if other_heatmaps:
                                    for item in other_heatmaps:
                                        with st.container():
                                            st.subheader(f"Liquidation Heatmap: {item['filename']}")
                                            
                                            # Display time and price range
                                            cols = st.columns([1, 1])
                                            with cols[0]:
                                                st.markdown(f"**📅 Time:** {item['timestamp'].split('T')[0]} {item['timestamp'].split('T')[1][:8]}")
                                                
                                                if 'price_scale' in item:
                                                    st.markdown(f"""
                                                    <div style="border:1px solid #ccc; border-radius:5px; padding:10px; background-color:#f9f9f9;">
                                                        <p style="margin:0; font-weight:bold;">💰 Price Range:</p>
                                                        <p style="margin:0; font-size:18px;">
                                                            ${item['price_scale']['bottom']:,.2f} - ${item['price_scale']['top']:,.2f}
                                                        </p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                            
                                            # Display clusters in a visually appealing way
                                            if 'clusters' in item and item['clusters'] is not None and len(item['clusters']) > 0:
                                                st.markdown("### 🔥 Liquidation Clusters")
                                                
                                                # Create a visualization of the clusters
                                                fig, ax = plt.subplots(figsize=(10, 5))
                                                clusters_df = pd.DataFrame(item['clusters'], columns=['Price', 'Strength'])
                                                clusters_df = clusters_df.sort_values('Price', ascending=False)
                                                
                                                # Use different colors based on strength
                                                colors = []
                                                for strength in clusters_df['Strength']:
                                                    if strength > 0.8:
                                                        colors.append('#4682b4')  # Strong blue
                                                    elif strength > 0.5:
                                                        colors.append('#7cb9e8')  # Medium blue
                                                    else:
                                                        colors.append('#b0e0e6')  # Light blue
                                                
                                                # Plot horizontal bars for each cluster
                                                bars = ax.barh(range(len(clusters_df)), clusters_df['Strength'], color=colors, height=0.6)
                                                
                                                # Add price labels
                                                for i, (price, strength) in enumerate(zip(clusters_df['Price'], clusters_df['Strength'])):
                                                    ax.text(strength + 0.02, i, f"${price:,.2f}", va='center', fontweight='bold')
                                                
                                                ax.set_yticks(range(len(clusters_df)))
                                                ax.set_yticklabels([f"Cluster {i+1}" for i in range(len(clusters_df))])
                                                ax.set_xlim(0, 1.2)
                                                ax.set_title("Liquidation Clusters", fontsize=16)
                                                ax.set_xlabel("Strength", fontsize=12)
                                                ax.grid(axis='x', linestyle='--', alpha=0.7)
                                                
                                                # Add a color gradient to the background
                                                ax.set_facecolor('#f9f9f9')
                                                
                                                st.pyplot(fig)
                                                
                                                # Display the clusters in a styled table
                                                st.markdown("#### Liquidation Levels")
                                                styled_df = clusters_df.style.format({
                                                    'Price': '${:,.2f}', 
                                                    'Strength': '{:.2f}'
                                                }).background_gradient(cmap='Blues', subset=['Strength'])
                                                
                                                st.dataframe(styled_df, use_container_width=True)
                                                
                                                # Add trading implications
                                                st.markdown("#### Trading Implications")
                                                
                                                # Find strongest cluster
                                                strongest_cluster = clusters_df.loc[clusters_df['Strength'].idxmax()]
                                                
                                                st.markdown(f"""
                                                * **Strongest liquidation level:** ${strongest_cluster['Price']:,.2f} (Strength: {strongest_cluster['Strength']:.2f})
                                                * **Potential support/resistance:** Major liquidation levels often act as support or resistance
                                                * **Stop loss placement:** Consider placing stops away from major liquidation levels to avoid cascades
                                                """)
                                            else:
                                                st.info("No liquidation clusters detected in this heatmap")
                                            
                                            st.markdown("---")
                                else:
                                    st.info("No other heatmaps available")
                        else:
                            st.info("No heatmap analysis available")
                else:
                    st.warning("No analysis results available.")
    
    # Cross-asset analysis
    with tab2:
        st.header("Cross-Asset Analysis")
        
        if st.button("Compute Cross-Asset Correlations", key="cross_asset_button"):
            with st.spinner("Computing correlations..."):
                correlations = compute_cross_asset_correlations(assets_data)
                
                # Display correlation matrix
                if correlations:
                    st.subheader("Asset Correlations")
                    
                    # Create a DataFrame for better visualization
                    corr_data = []
                    for pair, metrics in correlations.items():
                        asset1, asset2 = pair.split('_')
                        corr_data.append({
                            'Asset Pair': f"{asset1}/{asset2}",
                            'Price Correlation': metrics['price_correlation'],
                            'Returns Correlation': metrics['returns_correlation'],
                            'Lead/Lag': f"{metrics['lead_lag']} periods",
                            'Lead/Lag Strength': metrics['lead_lag_strength']
                        })
                    
                    if corr_data:
                        # Convert to DataFrame for display
                        corr_df = pd.DataFrame(corr_data)
                        
                        # Style the DataFrame
                        st.dataframe(corr_df.style.format({
                            'Price Correlation': '{:.4f}',
                            'Returns Correlation': '{:.4f}',
                            'Lead/Lag Strength': '{:.4f}'
                        }).background_gradient(subset=['Price Correlation'], cmap='coolwarm', vmin=-1, vmax=1))
                        
                        # Plot correlation heatmap
                        if len(assets) > 1:
                            st.subheader("Correlation Heatmap")
                            
                            # Create correlation matrix
                            corr_matrix = pd.DataFrame(index=assets, columns=assets)
                            for i, asset1 in enumerate(assets):
                                corr_matrix.loc[asset1, asset1] = 1.0
                                for j, asset2 in enumerate(assets):
                                    if i < j:
                                        pair_key = f"{asset1}_{asset2}"
                                        if pair_key in correlations:
                                            corr_matrix.loc[asset1, asset2] = correlations[pair_key]['price_correlation']
                                            corr_matrix.loc[asset2, asset1] = correlations[pair_key]['price_correlation']
                            
                            # Fill NaN values
                            corr_matrix = corr_matrix.fillna(0)
                            
                            # Plot heatmap
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(corr_matrix.astype(float), cmap='coolwarm', vmin=-1, vmax=1)
                            
                            # Add colorbar
                            plt.colorbar(im)
                            
                            # Add labels
                            ax.set_xticks(np.arange(len(assets)))
                            ax.set_yticks(np.arange(len(assets)))
                            ax.set_xticklabels(assets)
                            ax.set_yticklabels(assets)
                            
                            # Rotate x labels
                            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                            
                            # Add values to cells
                            for i in range(len(assets)):
                                for j in range(len(assets)):
                                    text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                                ha="center", va="center", color="black")
                            
                            ax.set_title("Asset Price Correlations")
                            fig.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Add normalized price chart
                            st.subheader("Normalized Price Comparison")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            for asset, df in assets_data.items():
                                if not df.empty:
                                    # Normalize prices to compare different scales
                                    normalized = df['close'] / df['close'].iloc[0]
                                    ax.plot(normalized, label=asset, linewidth=2)
                            
                            ax.set_title("Normalized Price Comparison")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Normalized Price")
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            plt.xticks(rotation=45)
                            fig.tight_layout()
                            
                            st.pyplot(fig)
                    else:
                        st.info("Not enough data to compute correlations.")
                else:
                    st.warning("No correlation data available. Please check your asset selection.")
    
    # Main analysis tab
    with tab1:
        st.header(f"Analysis for {primary_asset}")
        
        # Get data for primary asset
        data = assets_data.get(primary_asset, pd.DataFrame())
        
        # Add status text for backtesting
        st.info("Preparing data for backtest...")

        # Ensure data is a DataFrame and not a list
        if isinstance(data, list):
            st.error("Data is not in the correct format. Please check the data source.")
            st.stop()

        if data.empty:
            st.error(f"No data available for {primary_asset}")
            st.stop()
        
        # Process liquidation data
        heatmap = None
        if use_coinglass:
            try:
                with st.spinner("Fetching liquidation data from Coinglass..."):
                    heatmap = get_liquidation_heatmap(primary_asset)
            except Exception as e:
                st.error(f"Failed to fetch Coinglass data: {e}")
        elif 'heatmaps' in uploaded_files and uploaded_files['heatmaps']:
            try:
                with st.spinner("Processing uploaded heatmaps..."):
                    # Use the first heatmap for now
                    file = uploaded_files['heatmaps'][0]
                    data_bytes = file.read()
                    file.seek(0)  # Reset file pointer for future reads
                    
                    # Try to auto-detect price scale
                    price_scale = auto_detect_heatmap_scale(data_bytes)
                    if price_scale:
                        top_val, bot_val = price_scale
                    else:
                        # Use current price range as fallback
                        current_price = data['close'].iloc[-1]
                        top_val = current_price * 1.2
                        bot_val = current_price * 0.8
                    
                    heatmap = extract_liq_clusters_from_image(data_bytes, price_top=top_val, price_bottom=bot_val)
            except Exception as e:
                st.error(f"Failed to process liquidation heatmap: {e}")
        
        # Run backtest
        try:
            with st.spinner("Running backtest..."):
                backtester = EnhancedBacktester(
                    data,
                    asset_type=primary_asset,
                    static_liquidation_data=heatmap,
                    regime_filter=regime_filter,
                    fee_bps=fees,
                    adaptive_cutoff=adaptive_cutoff,
                    min_agree_features=conflict_min
                )
                results = backtester.run_backtest(window_size=window, lookahead_candles=lookahead)
                st.success("Backtest completed.")
            
            # Display results
            st.subheader("Backtest Results")
            st.dataframe(results)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            try:
                metrics = backtester.performance()
                
                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                if isinstance(metrics, dict):
                    with col1:
                        st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%")
                    with col2:
                        st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
                    with col4:
                        st.metric("Max Drawdown", f"{metrics.get('max_dd', 0)*100:.2f}%")
                
                # Example plot using matplotlib
                st.subheader("Performance Chart")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if 'cum_bh' in results.columns and 'cum_strat' in results.columns:
                    ax.plot(results['cum_bh'], label='Buy & Hold', alpha=0.7)
                    ax.plot(results['cum_strat'], label='Strategy', linewidth=2)
                    
                    # Add grid and legend
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Set labels
                    ax.set_title(f'{primary_asset} Cumulative Returns')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Return (%)')
                    
                    # Format y-axis as percentage
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                    
                    # Rotate x-axis labels
                    plt.xticks(rotation=45)
                    
                    # Tight layout
                    fig.tight_layout()
                    
                    st.pyplot(fig)
                else:
                    st.warning("Performance data not available for plotting.")
            except Exception as e:
                st.error(f"Error calculating performance metrics: {e}")
                st.write("Performance metrics not available.")
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            st.write("Backtest failed. Please check your settings and try again.")
        
        # Signal analysis
        st.subheader("Current Signal Analysis")
        if not data.empty:
            # Get the latest data point
            latest_data = data.iloc[-1]
            
            # Create columns for signal components
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price", f"${latest_data['close']:.2f}")
                st.metric("RSI", f"{latest_data['rsi_raw']:.2f}")
            
            with col2:
                st.metric("Volume Ratio", f"{latest_data['volume_ratio']:.2f}")
                trend = "Bullish" if latest_data['webtrend_status'] else "Bearish"
                st.metric("WebTrend", trend)
            
            with col3:
                # Ensure results is initialized before use
                if 'results' in locals() and not results.empty and 'signal' in results.columns:
                    latest_signal = results['signal'].iloc[-1]
                    signal_value = results['signal_value'].iloc[-1] if 'signal_value' in results.columns else 0
                    st.metric("Signal", latest_signal)
                    st.metric("Signal Strength", f"{signal_value:.2f}")
                else:
                    st.warning("No results available to display signals.")
        
        # Cross-reference with other assets
        st.subheader("Cross-Reference with Other Assets")
        if len(assets) > 1:
            # Create a chart showing price movements of all assets
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for asset, df in assets_data.items():
                if not df.empty:
                    # Normalize prices to compare different scales
                    normalized = df['close'] / df['close'].iloc[0]
                    ax.plot(normalized, label=asset)
            
            ax.set_title("Normalized Price Comparison")
            ax.set_xlabel("Date")
            ax.set_ylabel("Normalized Price")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.xticks(rotation=45)
            fig.tight_layout()
            
            st.pyplot(fig)

# Add a PowerShell progress bar
def display_progress():
    for i in range(101):
        print(f"Progress: {i}%", end="\r")
        time.sleep(0.01)
    print("Processing complete.")

if __name__ == "__main__":
    display_progress()
    main()