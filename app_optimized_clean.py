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
            
            exp3 = st.expander("🔥 Liquidation Heatmaps", expanded=True)
            with exp3:
                heatmaps = st.file_uploader(
                    "Upload heatmap images", 
                    type=["png", "jpg", "jpeg"], 
                    accept_multiple_files=True, 
                    key="heatmaps",
                    help="Upload screenshots of liquidation heatmaps from Coinglass or similar platforms"
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
                    image_analysis = analyze_uploaded_images(all_files, None)
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
                                display_chart_item(item, "Bitcoin")
                        else:
                            st.info("No Bitcoin charts available")
                    
                    # SOL Tab
                    with asset_tabs[1]:
                        if sol_charts:
                            for item in sol_charts:
                                display_chart_item(item, "Solana")
                        else:
                            st.info("No Solana charts available")
                    
                    # BONK Tab
                    with asset_tabs[2]:
                        if bonk_charts:
                            for item in bonk_charts:
                                display_chart_item(item, "BONK")
                        else:
                            st.info("No BONK charts available")
                    
                    # Other Tab
                    with asset_tabs[3]:
                        if other_charts:
                            for item in other_charts:
                                display_chart_item(item, "Other")
                        else:
                            st.info("No other charts available")
                    
                    # Display analysis summary
                    st.subheader("Analysis Summary")
                    st.write("Analysis completed successfully!")

if __name__ == "__main__":
    main()
