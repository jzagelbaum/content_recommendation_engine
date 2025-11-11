"""
Real-time Analytics Dashboard
=============================

Real-time analytics and monitoring dashboard using Streamlit for
interactive visualization of recommendation engine metrics.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import asyncio

# Import monitoring service
from monitoring_service import MonitoringService, MonitoringServiceFactory

# Configure Streamlit page
st.set_page_config(
    page_title="Content Recommendation Engine - Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'monitoring_service' not in st.session_state:
    st.session_state.monitoring_service = MonitoringServiceFactory.create_from_config()

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

def load_dashboard_data(time_range_hours: int = 24) -> Dict[str, Any]:
    """Load dashboard data from monitoring service"""
    try:
        return st.session_state.monitoring_service.generate_performance_dashboard(time_range_hours)
    except Exception as e:
        st.error(f"Failed to load dashboard data: {e}")
        return {}

def create_kpi_cards(dashboard_data: Dict[str, Any]):
    """Create KPI cards for key metrics"""
    rec_metrics = dashboard_data.get("recommendation_metrics", {})
    eng_metrics = dashboard_data.get("user_engagement", {})
    sys_metrics = dashboard_data.get("system_health", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Success Rate",
            value=f"{rec_metrics.get('success_rate', 0):.1f}%",
            delta=f"{2.3:.1f}%" if rec_metrics.get('success_rate', 0) > 95 else f"{-1.2:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Avg Response Time",
            value=f"{rec_metrics.get('average_response_time', 0):.0f}ms",
            delta=f"{-45:.0f}ms"
        )
    
    with col3:
        st.metric(
            label="Active Users",
            value=f"{eng_metrics.get('unique_users', 0):,}",
            delta=f"{234:,}"
        )
    
    with col4:
        st.metric(
            label="System Availability",
            value=f"{sys_metrics.get('availability', 0):.1f}%",
            delta=f"{0.2:.1f}%"
        )

def create_recommendation_charts(dashboard_data: Dict[str, Any]):
    """Create recommendation performance charts"""
    st.subheader("📈 Recommendation Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Algorithm Distribution
        algo_data = dashboard_data.get("algorithm_performance", {})
        if algo_data:
            fig = go.Figure(data=[go.Pie(
                labels=list(algo_data.keys()),
                values=list(algo_data.values()),
                hole=0.3
            )])
            fig.update_layout(
                title="Algorithm Usage Distribution",
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Success Rate Gauge
        success_rate = dashboard_data.get("recommendation_metrics", {}).get("success_rate", 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=success_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Success Rate (%)"},
            delta={'reference': 95},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 90], 'color': "lightgray"},
                    {'range': [90, 95], 'color': "yellow"},
                    {'range': [95, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_user_engagement_charts(dashboard_data: Dict[str, Any]):
    """Create user engagement charts"""
    st.subheader("👥 User Engagement Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interaction Types Breakdown
        interaction_data = dashboard_data.get("interaction_breakdown", {})
        if interaction_data:
            fig = px.bar(
                x=list(interaction_data.keys()),
                y=list(interaction_data.values()),
                title="User Interactions by Type",
                color=list(interaction_data.values()),
                color_continuous_scale="viridis"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Engagement Trends (mock time series data)
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
        engagement_scores = np.random.normal(75, 10, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=engagement_scores,
            mode='lines+markers',
            name='Engagement Score',
            line=dict(color='blue', width=3)
        ))
        fig.update_layout(
            title="User Engagement Trend (7 days)",
            xaxis_title="Date",
            yaxis_title="Engagement Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def create_system_health_dashboard(dashboard_data: Dict[str, Any]):
    """Create system health monitoring dashboard"""
    st.subheader("🔧 System Health Monitoring")
    
    sys_metrics = dashboard_data.get("system_health", {})
    
    # Create gauge charts for system metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_config = [
        ("CPU %", sys_metrics.get("cpu_utilization", 0), [0, 80, 100], ["lightgreen", "yellow", "red"]),
        ("Memory %", sys_metrics.get("memory_utilization", 0), [0, 85, 100], ["lightgreen", "yellow", "red"]),
        ("Availability %", sys_metrics.get("availability", 0), [95, 99, 100], ["red", "yellow", "lightgreen"]),
        ("Error Rate %", sys_metrics.get("error_rate", 0), [0, 5, 10], ["lightgreen", "yellow", "red"])
    ]
    
    cols = [col1, col2, col3, col4]
    
    for i, (title, value, ranges, colors) in enumerate(metrics_config):
        with cols[i]:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': title},
                gauge={
                    'axis': {'range': [ranges[0], ranges[2]]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [ranges[0], ranges[1]], 'color': colors[0]},
                        {'range': [ranges[1], ranges[2]], 'color': colors[1]}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

def create_alerts_panel(dashboard_data: Dict[str, Any]):
    """Create alerts and notifications panel"""
    st.subheader("🚨 Alerts & Notifications")
    
    # Mock alerts data
    alerts = [
        {
            "severity": "high",
            "metric": "Response Time",
            "message": "Average response time exceeded 2 seconds",
            "timestamp": datetime.now() - timedelta(minutes=15)
        },
        {
            "severity": "medium",
            "metric": "Cache Hit Rate",
            "message": "Cache hit rate dropped below 70%",
            "timestamp": datetime.now() - timedelta(hours=2)
        },
        {
            "severity": "low",
            "metric": "Disk Usage",
            "message": "Disk usage approaching 80%",
            "timestamp": datetime.now() - timedelta(hours=6)
        }
    ]
    
    if not alerts:
        st.success("✅ No active alerts - All systems operating normally")
    else:
        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            st.markdown(f"""
            <div class="{severity_class}">
                <strong>{alert['severity'].upper()}</strong> - {alert['metric']}<br>
                {alert['message']}<br>
                <small>🕒 {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

def create_real_time_metrics():
    """Create real-time metrics section"""
    st.subheader("⚡ Real-time Metrics")
    
    # Create placeholder for real-time updates
    placeholder = st.empty()
    
    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Real-time request count
            current_requests = np.random.randint(50, 150)
            st.metric("Requests/min", f"{current_requests}", f"{np.random.randint(-10, 20)}")
        
        with col2:
            # Real-time cache hit rate
            cache_rate = np.random.uniform(75, 85)
            st.metric("Cache Hit Rate", f"{cache_rate:.1f}%", f"{np.random.uniform(-2, 2):.1f}%")
        
        with col3:
            # Real-time error count
            error_count = np.random.randint(0, 5)
            st.metric("Errors/min", f"{error_count}", f"{np.random.randint(-2, 3)}")

def create_performance_trends():
    """Create performance trends section"""
    st.subheader("📊 Performance Trends")
    
    # Generate mock time series data
    hours = list(range(24))
    response_times = [400 + 50 * np.sin(i * np.pi / 12) + np.random.normal(0, 20) for i in hours]
    request_counts = [1000 + 500 * np.sin((i - 6) * np.pi / 12) + np.random.normal(0, 50) for i in hours]
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Response Time (ms)', 'Request Count'),
        vertical_spacing=0.1
    )
    
    # Response time trend
    fig.add_trace(
        go.Scatter(x=hours, y=response_times, mode='lines+markers', name='Response Time'),
        row=1, col=1
    )
    
    # Request count trend
    fig.add_trace(
        go.Scatter(x=hours, y=request_counts, mode='lines+markers', name='Request Count', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_yaxes(title_text="Response Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Request Count", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application"""
    # Header
    st.markdown('<h1 class="main-header">🎯 Content Recommendation Engine</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("⚙️ Dashboard Controls")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["1 Hour", "6 Hours", "24 Hours", "7 Days"],
        index=2
    )
    
    time_range_hours = {
        "1 Hour": 1,
        "6 Hours": 6,
        "24 Hours": 24,
        "7 Days": 168
    }[time_range]
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    # Refresh interval
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.session_state.last_update = datetime.now()
        st.experimental_rerun()
    
    # Last update timestamp
    st.sidebar.markdown(f"**Last Updated:** {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Load dashboard data
    with st.spinner("Loading dashboard data..."):
        dashboard_data = load_dashboard_data(time_range_hours)
    
    if not dashboard_data:
        st.error("Failed to load dashboard data. Please check the monitoring service.")
        return
    
    # Main dashboard content
    
    # KPI Cards
    create_kpi_cards(dashboard_data)
    
    st.divider()
    
    # Real-time metrics
    create_real_time_metrics()
    
    st.divider()
    
    # Recommendation performance
    create_recommendation_charts(dashboard_data)
    
    st.divider()
    
    # User engagement
    create_user_engagement_charts(dashboard_data)
    
    st.divider()
    
    # System health
    create_system_health_dashboard(dashboard_data)
    
    st.divider()
    
    # Performance trends
    create_performance_trends()
    
    st.divider()
    
    # Alerts panel
    create_alerts_panel(dashboard_data)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()

if __name__ == "__main__":
    main()