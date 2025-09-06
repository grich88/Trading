"""
Streamlit Dashboard Application.

This module provides a web-based dashboard UI for the crypto trading algorithm
system, displaying all signals and analytics in real-time.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any
import json

from src.services.unified_dashboard import UnifiedDashboardService, AlertLevel
from src.config.config import Config


# Page configuration
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 18px !important;
    }
    .alert-critical {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-warning {
        background-color: #ffffcc;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-info {
        background-color: #ccddff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .signal-bullish {
        color: #28a745;
        font-weight: bold;
    }
    .signal-bearish {
        color: #dc3545;
        font-weight: bold;
    }
    .signal-neutral {
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


class DashboardApp:
    """Main dashboard application class."""
    
    def __init__(self):
        """Initialize the dashboard application."""
        self.config = Config()
        self.dashboard_service = UnifiedDashboardService(self.config)
        
        # Initialize session state
        if 'symbols' not in st.session_state:
            st.session_state.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        if 'timeframe' not in st.session_state:
            st.session_state.timeframe = '1h'
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'dashboard_data' not in st.session_state:
            st.session_state.dashboard_data = None
    
    def run(self):
        """Run the dashboard application."""
        # Header
        st.title("🚀 Crypto Trading Algorithm Dashboard")
        st.markdown("---")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        if st.session_state.dashboard_data:
            self._render_main_content()
        else:
            self._fetch_dashboard_data()
            if st.session_state.dashboard_data:
                self._render_main_content()
            else:
                st.info("Loading dashboard data... Please wait.")
        
        # Auto-refresh
        if st.session_state.auto_refresh:
            st.experimental_rerun()
    
    def _render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("Dashboard Controls")
        
        # Symbol selection
        st.sidebar.subheader("Trading Pairs")
        
        # Predefined symbols
        available_symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'ADA/USDT', 'DOT/USDT', 'MATIC/USDT', 'AVAX/USDT'
        ]
        
        st.session_state.symbols = st.sidebar.multiselect(
            "Select symbols to monitor:",
            available_symbols,
            default=st.session_state.symbols
        )
        
        # Timeframe selection
        st.sidebar.subheader("Timeframe")
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        st.session_state.timeframe = st.sidebar.selectbox(
            "Select timeframe:",
            timeframes,
            index=timeframes.index(st.session_state.timeframe)
        )
        
        # Refresh controls
        st.sidebar.subheader("Refresh Settings")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Now"):
                st.session_state.dashboard_data = None
                st.experimental_rerun()
        
        with col2:
            st.session_state.auto_refresh = st.checkbox(
                "Auto-refresh",
                value=st.session_state.auto_refresh
            )
        
        # Export options
        st.sidebar.subheader("Export Data")
        export_format = st.sidebar.selectbox(
            "Export format:",
            ['JSON', 'CSV', 'HTML']
        )
        
        if st.sidebar.button("📥 Export Dashboard"):
            self._export_dashboard(export_format.lower())
        
        # Info
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This dashboard integrates 10 critical signal categories for "
            "comprehensive market analysis."
        )
    
    def _fetch_dashboard_data(self):
        """Fetch dashboard data from the service."""
        try:
            with st.spinner("Fetching latest market data..."):
                st.session_state.dashboard_data = self.dashboard_service.get_dashboard_data(
                    symbols=st.session_state.symbols,
                    timeframe=st.session_state.timeframe
                )
        except Exception as e:
            st.error(f"Error fetching dashboard data: {str(e)}")
    
    def _render_main_content(self):
        """Render main dashboard content."""
        data = st.session_state.dashboard_data
        
        # Alerts section
        self._render_alerts(data.get('alerts', []))
        
        # Main metrics row
        self._render_key_metrics(data)
        
        # Market overview
        st.header("📊 Market Overview")
        self._render_market_overview(data.get('market_overview', []))
        
        # Signals section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📡 Active Signals")
            self._render_signals(data.get('active_signals', []))
        
        with col2:
            st.header("📈 Signal Summary")
            self._render_signal_summary(data.get('signal_summary', {}))
        
        # Recommendations
        st.header("💡 Trading Recommendations")
        self._render_recommendations(data.get('recommendations', []))
        
        # Correlations
        st.header("🔗 Signal Correlations")
        self._render_correlations(data.get('correlations', {}))
        
        # Performance metrics
        st.header("📊 System Performance")
        self._render_performance(data.get('performance', {}))
    
    def _render_alerts(self, alerts: List[Any]):
        """Render alerts section."""
        if not alerts:
            return
        
        st.header("🚨 Active Alerts")
        
        for alert in alerts[:5]:  # Show top 5 alerts
            css_class = f"alert-{alert.level.value}"
            
            alert_html = f"""
            <div class="{css_class}">
                <strong>{alert.title}</strong><br>
                {alert.message}<br>
                <small>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """
            
            st.markdown(alert_html, unsafe_allow_html=True)
    
    def _render_key_metrics(self, data: Dict[str, Any]):
        """Render key metrics row."""
        st.header("🎯 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        summary = data.get('signal_summary', {})
        
        with col1:
            st.metric(
                "Total Signals",
                summary.get('total_signals', 0),
                delta=None
            )
        
        with col2:
            avg_strength = summary.get('average_strength', 0)
            st.metric(
                "Avg Signal Strength",
                f"{avg_strength:.1f}%",
                delta=None
            )
        
        with col3:
            distribution = summary.get('distribution', {})
            bullish = distribution.get('bullish', 0)
            bearish = distribution.get('bearish', 0)
            
            sentiment = "Neutral"
            if bullish > bearish * 1.5:
                sentiment = "Bullish"
            elif bearish > bullish * 1.5:
                sentiment = "Bearish"
            
            st.metric(
                "Market Sentiment",
                sentiment,
                delta=f"{bullish}↑ {bearish}↓"
            )
        
        with col4:
            alerts_count = len(data.get('alerts', []))
            critical_count = sum(1 for a in data.get('alerts', []) 
                               if a.level == AlertLevel.CRITICAL)
            
            st.metric(
                "Active Alerts",
                alerts_count,
                delta=f"{critical_count} critical" if critical_count > 0 else None
            )
    
    def _render_market_overview(self, overviews: List[Any]):
        """Render market overview section."""
        if not overviews:
            st.info("No market data available")
            return
        
        # Create DataFrame for display
        df_data = []
        for overview in overviews:
            df_data.append({
                'Symbol': overview.symbol,
                'Price': f"${overview.price:,.2f}",
                '24h Change': f"{overview.price_change_24h:+.2f}%",
                'Volume': f"${overview.volume_24h:,.0f}",
                'Trend': overview.trend.upper(),
                'Volatility': f"{overview.volatility:.2f}%",
                'Sentiment': f"{overview.sentiment_score:.2f}"
            })
        
        df = pd.DataFrame(df_data)
        
        # Style the dataframe
        def highlight_change(val):
            if isinstance(val, str) and '%' in val:
                value = float(val.replace('%', '').replace('+', ''))
                if value > 0:
                    return 'color: green'
                elif value < 0:
                    return 'color: red'
            return ''
        
        styled_df = df.style.applymap(highlight_change, subset=['24h Change'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Price chart
        if len(overviews) > 0:
            fig = go.Figure()
            
            for overview in overviews:
                # Simulated price data for visualization
                prices = [overview.price * (1 + (i-50)*0.001) for i in range(100)]
                
                fig.add_trace(go.Scatter(
                    y=prices,
                    name=overview.symbol,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Price Trends",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_signals(self, signals: List[Any]):
        """Render active signals section."""
        if not signals:
            st.info("No active signals")
            return
        
        # Group signals by category
        signals_by_category = {}
        for signal in signals:
            category = signal.category.value
            if category not in signals_by_category:
                signals_by_category[category] = []
            signals_by_category[category].append(signal)
        
        # Display signals by category
        for category, cat_signals in signals_by_category.items():
            with st.expander(f"{category.upper()} Signals ({len(cat_signals)})"):
                for signal in cat_signals:
                    # Signal type styling
                    signal_class = f"signal-{signal.signal_type}"
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(
                            f"<span class='{signal_class}'>{signal.source}</span>: {signal.message}",
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.metric("Strength", f"{signal.strength:.0f}%")
                    
                    with col3:
                        st.metric("Confidence", f"{signal.confidence:.0f}%")
                    
                    st.markdown("---")
    
    def _render_signal_summary(self, summary: Dict[str, Any]):
        """Render signal summary section."""
        if not summary:
            st.info("No signal summary available")
            return
        
        # Distribution pie chart
        distribution = summary.get('distribution', {})
        if distribution:
            fig = px.pie(
                values=list(distribution.values()),
                names=list(distribution.keys()),
                title="Signal Distribution",
                color_discrete_map={
                    'bullish': '#28a745',
                    'bearish': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Strongest signal
        strongest = summary.get('strongest_signal')
        if strongest:
            st.subheader("Strongest Signal")
            st.info(
                f"**{strongest['source']}** ({strongest['type']})\n\n"
                f"Strength: {strongest['strength']:.0f}%\n\n"
                f"{strongest['message']}"
            )
    
    def _render_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Render trading recommendations."""
        if not recommendations:
            st.info("No recommendations available")
            return
        
        for rec in recommendations:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"**{rec['symbol']}**")
            
            with col2:
                action_color = 'green' if 'BUY' in rec['action'] else 'red' if 'SELL' in rec['action'] else 'gray'
                st.markdown(
                    f"<span style='color: {action_color}; font-weight: bold;'>{rec['action']}</span>",
                    unsafe_allow_html=True
                )
            
            with col3:
                st.text(f"Confidence: {rec['confidence']}")
            
            with col4:
                st.text(f"Risk: {rec['risk_level']}")
            
            with col5:
                st.text(f"Size: {rec['position_size']}")
            
            # Key factors
            with st.expander(f"Key Factors for {rec['symbol']}"):
                for factor in rec.get('key_factors', []):
                    st.write(f"• {factor}")
            
            st.markdown("---")
    
    def _render_correlations(self, correlations: Dict[str, Any]):
        """Render signal correlations section."""
        if not correlations or not correlations.get('correlations'):
            st.info("No correlation data available")
            return
        
        # Correlation insights
        insights = correlations.get('insights', [])
        if insights:
            for insight in insights:
                st.info(insight)
        
        # Correlation matrix visualization
        corr_data = correlations.get('correlations', [])
        if corr_data:
            # Create correlation matrix
            sources = set()
            for corr in corr_data:
                sources.add(corr['source1'])
                sources.add(corr['source2'])
            
            sources = sorted(list(sources))
            n = len(sources)
            
            if n > 1:
                matrix = [[0.0 for _ in range(n)] for _ in range(n)]
                
                for i in range(n):
                    matrix[i][i] = 1.0
                
                for corr in corr_data:
                    i = sources.index(corr['source1'])
                    j = sources.index(corr['source2'])
                    matrix[i][j] = corr['correlation']
                    matrix[j][i] = corr['correlation']
                
                # Create heatmap
                fig = px.imshow(
                    matrix,
                    labels=dict(x="Signal Source", y="Signal Source", color="Correlation"),
                    x=sources,
                    y=sources,
                    color_continuous_scale='RdBu',
                    title="Signal Correlation Matrix"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance(self, performance: Dict[str, Any]):
        """Render performance metrics."""
        if not performance:
            st.info("No performance data available")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            signal_acc = performance.get('signal_accuracy', 0) * 100
            st.metric("Signal Accuracy", f"{signal_acc:.1f}%")
        
        with col2:
            alert_acc = performance.get('alert_accuracy', 0) * 100
            st.metric("Alert Accuracy", f"{alert_acc:.1f}%")
        
        with col3:
            uptime = performance.get('system_uptime', 0)
            st.metric("System Uptime", f"{uptime:.1f}%")
        
        # Recommendation performance
        rec_perf = performance.get('recommendation_performance', {})
        if rec_perf:
            st.subheader("Recommendation Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                win_rate = rec_perf.get('win_rate', 0) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col2:
                avg_return = rec_perf.get('avg_return', 0) * 100
                st.metric("Avg Return", f"{avg_return:+.2f}%")
            
            with col3:
                sharpe = rec_perf.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    def _export_dashboard(self, format: str):
        """Export dashboard data in specified format."""
        if not st.session_state.dashboard_data:
            st.sidebar.error("No data to export")
            return
        
        try:
            exported_data = self.dashboard_service.export_dashboard_data(
                st.session_state.dashboard_data,
                format
            )
            
            # Prepare download
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dashboard_export_{timestamp}.{format}"
            
            if format == 'json':
                mime_type = 'application/json'
            elif format == 'csv':
                mime_type = 'text/csv'
            else:  # html
                mime_type = 'text/html'
            
            st.sidebar.download_button(
                label=f"Download {format.upper()}",
                data=exported_data,
                file_name=filename,
                mime=mime_type
            )
            
            st.sidebar.success(f"Export ready for download!")
            
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")


def main():
    """Main entry point for the dashboard application."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
