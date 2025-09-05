"""
Coinglass API Integration for RSI + Volume Predictive Scoring Model

This module provides functions to fetch liquidation data from Coinglass API
for use in the Enhanced RSI + Volume Predictive Scoring Model.

API Documentation: https://coinglass.readme.io/reference/api-overview
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("coinglass_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Coinglass API configuration
API_KEY = os.getenv("COINGLASS_API_KEY", "cac741e49e3143269f61c21498c63d8a")
BASE_URL = "https://open-api.coinglass.com/api/pro/v1"
HEADERS = {
    "coinglassSecret": API_KEY,
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://coinglass.com",
    "Referer": "https://coinglass.com/",
}

# Cache directory for API responses
CACHE_DIR = "data/coinglass_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_response(endpoint, params=None, cache_duration=3600):
    """
    Get cached API response if available and not expired
    
    Parameters:
    -----------
    endpoint : str
        API endpoint
    params : dict
        Query parameters
    cache_duration : int
        Cache duration in seconds (default: 1 hour)
        
    Returns:
    --------
    dict or None
        Cached response if available, None otherwise
    """
    # Create cache key from endpoint and params
    cache_key = endpoint
    if params:
        cache_key += "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    
    cache_key = cache_key.replace("/", "_")
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    # Check if cache file exists and is not expired
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < cache_duration:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache file: {e}")
    
    return None

def cache_response(endpoint, params, response):
    """
    Cache API response
    
    Parameters:
    -----------
    endpoint : str
        API endpoint
    params : dict
        Query parameters
    response : dict
        API response
    """
    # Create cache key from endpoint and params
    cache_key = endpoint
    if params:
        cache_key += "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    
    cache_key = cache_key.replace("/", "_")
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    try:
        with open(cache_file, "w") as f:
            json.dump(response, f)
    except Exception as e:
        logger.warning(f"Error caching response: {e}")

def make_api_request(endpoint, params=None, use_cache=True, cache_duration=3600, retries: int = 3, backoff: float = 0.8):
    """
    Make API request to Coinglass
    
    Parameters:
    -----------
    endpoint : str
        API endpoint
    params : dict
        Query parameters
    use_cache : bool
        Whether to use cached responses
    cache_duration : int
        Cache duration in seconds
        
    Returns:
    --------
    dict
        API response
    """
    # Check cache first if enabled
    if use_cache:
        cached = get_cached_response(endpoint, params, cache_duration)
        if cached:
            logger.info(f"Using cached response for {endpoint}")
            return cached
    
    # Make API request
    url = f"{BASE_URL}{endpoint}"
    
    attempt = 0
    last_err = None
    while attempt < retries:
        try:
            if params:
                response = requests.get(url, headers=HEADERS, params=params, timeout=20)
            else:
                response = requests.get(url, headers=HEADERS, timeout=20)
            if response.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(backoff * (attempt + 1))
                attempt += 1
                continue
            response.raise_for_status()
            data = response.json()
            if use_cache:
                cache_response(endpoint, params, data)
            return data
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(backoff * (attempt + 1))
            attempt += 1
    logger.error(f"API request failed after retries: {last_err}")
    return {"success": False, "msg": last_err, "data": None}

def get_liquidation_data(symbol, interval="4h", limit=100, use_cache=True):
    """
    Get liquidation data for a symbol
    
    Parameters:
    -----------
    symbol : str
        Symbol (e.g., "BTC", "SOL")
    interval : str
        Time interval (default: "4h")
    limit : int
        Number of data points to fetch
    use_cache : bool
        Whether to use cached responses
        
    Returns:
    --------
    dict
        Liquidation data
    """
    endpoint = "/futures/liquidation/chart"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    response = make_api_request(endpoint, params, use_cache)
    
    if response.get("success"):
        return response.get("data", {})
    else:
        logger.error(f"Failed to get liquidation data: {response.get('msg')}")
        return {}

def get_liquidation_heatmap(symbol, use_cache=True):
    """
    Get liquidation heatmap for a symbol
    
    Parameters:
    -----------
    symbol : str
        Symbol (e.g., "BTC", "SOL")
    use_cache : bool
        Whether to use cached responses
        
    Returns:
    --------
    dict
        Liquidation heatmap data
    """
    endpoint = "/futures/liquidation/heatmap"
    params = {
        "symbol": symbol
    }
    
    response = make_api_request(endpoint, params, use_cache)
    if response.get("success"):
        return response.get("data", {})
    # Fallback 1: alternate path (some tiers)
    alt = make_api_request("/futures/liquidation/heatmap/chart", params, use_cache)
    if alt.get("success"):
        return alt.get("data", {})
    # Fallback 2: shorter param names
    alt2 = make_api_request(endpoint, {"symbol": symbol.upper()}, use_cache)
    if alt2.get("success"):
        return alt2.get("data", {})
    logger.error(f"Failed to get liquidation heatmap: {response.get('msg')} | alt: {alt.get('msg')} | alt2: {alt2.get('msg')}")
    return {}

def get_open_interest(symbol, interval="4h", limit=100, use_cache=True):
    """
    Get open interest data for a symbol
    
    Parameters:
    -----------
    symbol : str
        Symbol (e.g., "BTC", "SOL")
    interval : str
        Time interval (default: "4h")
    limit : int
        Number of data points to fetch
    use_cache : bool
        Whether to use cached responses
        
    Returns:
    --------
    dict
        Open interest data
    """
    endpoint = "/futures/openInterest/chart"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    response = make_api_request(endpoint, params, use_cache)
    
    if response.get("success"):
        return response.get("data", {})
    else:
        logger.error(f"Failed to get open interest data: {response.get('msg')}")
        return {}

def extract_liquidation_clusters(heatmap_data, price_series, num_clusters=5):
    """
    Extract liquidation clusters from heatmap data
    
    Parameters:
    -----------
    heatmap_data : dict
        Liquidation heatmap data from Coinglass API
    price_series : array-like
        Price series for the asset
    num_clusters : int
        Number of clusters to extract
        
    Returns:
    --------
    dict
        Liquidation clusters and cleared zone status
    """
    if not heatmap_data or "longLiquidationList" not in heatmap_data or "shortLiquidationList" not in heatmap_data:
        logger.warning("Invalid heatmap data")
        return {'clusters': [], 'cleared_zone': False}
    
    # Extract long and short liquidation data
    long_liquidations = heatmap_data.get("longLiquidationList", [])
    short_liquidations = heatmap_data.get("shortLiquidationList", [])
    
    # Combine liquidations
    all_liquidations = []
    
    for liq in long_liquidations:
        if "price" in liq and "amount" in liq:
            all_liquidations.append((float(liq["price"]), float(liq["amount"]), "long"))
    
    for liq in short_liquidations:
        if "price" in liq and "amount" in liq:
            all_liquidations.append((float(liq["price"]), float(liq["amount"]), "short"))
    
    # Sort liquidations by amount (intensity)
    all_liquidations.sort(key=lambda x: x[1], reverse=True)
    
    # Get top liquidation clusters
    top_liquidations = all_liquidations[:min(num_clusters*2, len(all_liquidations))]
    
    # Convert to clusters format
    clusters = []
    max_amount = max([liq[1] for liq in top_liquidations]) if top_liquidations else 1
    
    for price, amount, liq_type in top_liquidations:
        # Normalize intensity to 0-1 range
        intensity = min(1.0, amount / max_amount)
        
        # Only include significant liquidations (intensity > 0.2)
        if intensity > 0.2:
            clusters.append((price, intensity))
    
    # Determine if price has cleared major liquidation zones
    current_price = price_series[-1] if len(price_series) > 0 else 0
    
    # Count liquidation clusters above and below current price
    clusters_above = [c for c in clusters if c[0] > current_price]
    clusters_below = [c for c in clusters if c[0] < current_price]
    
    # Calculate total intensity above and below
    intensity_above = sum(c[1] for c in clusters_above)
    intensity_below = sum(c[1] for c in clusters_below)
    
    # Price has cleared major liquidation zones if more intensity is below than above
    cleared_zone = intensity_below > intensity_above
    
    return {
        'clusters': clusters,
        'cleared_zone': cleared_zone
    }

def get_historical_liquidation_data(symbol, start_date, end_date, interval="4h", use_cache=True):
    """
    Get historical liquidation data for a symbol
    
    Parameters:
    -----------
    symbol : str
        Symbol (e.g., "BTC", "SOL")
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str
        Time interval (default: "4h")
    use_cache : bool
        Whether to use cached responses
        
    Returns:
    --------
    dict
        Historical liquidation data
    """
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    endpoint = "/futures/liquidation/history"
    params = {
        "symbol": symbol,
        "startTime": start_ts,
        "endTime": end_ts
    }
    
    response = make_api_request(endpoint, params, use_cache)
    
    if response.get("success"):
        return response.get("data", {})
    else:
        logger.error(f"Failed to get historical liquidation data: {response.get('msg')}")
        return {}

def get_liquidation_data_for_backtesting(symbol, price_series, timestamp_series, use_cache=True):
    """
    Get liquidation data for backtesting
    
    Parameters:
    -----------
    symbol : str
        Symbol (e.g., "BTC", "SOL")
    price_series : array-like
        Price series for the asset
    timestamp_series : array-like
        Timestamp series for the price data
    use_cache : bool
        Whether to use cached responses
        
    Returns:
    --------
    list
        List of liquidation data for each timestamp
    """
    # Get current liquidation heatmap
    heatmap_data = get_liquidation_heatmap(symbol, use_cache)
    
    # Extract liquidation clusters
    liquidation_data = extract_liquidation_clusters(heatmap_data, price_series)
    
    # For backtesting, we'll use the same liquidation data for all timestamps
    # In a real implementation, we would fetch historical liquidation data for each timestamp
    return [liquidation_data for _ in range(len(price_series))]

if __name__ == "__main__":
    # Test the API
    print("Testing Coinglass API...")
    
    # Test getting liquidation data
    btc_liquidation = get_liquidation_data("BTC")
    print(f"BTC Liquidation Data: {json.dumps(btc_liquidation, indent=2)[:500]}...")
    
    # Test getting liquidation heatmap
    btc_heatmap = get_liquidation_heatmap("BTC")
    print(f"BTC Liquidation Heatmap: {json.dumps(btc_heatmap, indent=2)[:500]}...")
    
    # Test getting open interest
    btc_oi = get_open_interest("BTC")
    print(f"BTC Open Interest: {json.dumps(btc_oi, indent=2)[:500]}...")
    
    # Test extracting liquidation clusters
    sample_price_series = [40000, 41000, 42000, 43000, 44000, 45000]
    clusters = extract_liquidation_clusters(btc_heatmap, sample_price_series)
    print(f"BTC Liquidation Clusters: {clusters}")
    
    print("API test completed.")
