"""
Monitoring and Analytics Service
================================

Comprehensive monitoring for the content recommendation engine including
performance tracking, user engagement metrics, and system health monitoring.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import numpy as np
from azure.monitor.query import LogsQueryClient, MetricsQueryClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.applicationinsights import ApplicationInsightsDataClient
from azure.mgmt.applicationinsights import ApplicationInsightsManagementClient
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfiguration:
    """Configuration for monitoring service"""
    application_insights_connection_string: str
    log_analytics_workspace_id: str
    storage_account_name: str
    enable_real_time_monitoring: bool = True
    metrics_retention_days: int = 90
    alert_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class RecommendationMetrics:
    """Recommendation performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    unique_users: int = 0
    algorithm_distribution: Dict[str, int] = field(default_factory=dict)
    recommendation_quality_score: float = 0.0

@dataclass
class UserEngagementMetrics:
    """User engagement and behavior metrics"""
    total_interactions: int = 0
    unique_users: int = 0
    average_session_duration: float = 0.0
    bounce_rate: float = 0.0
    interaction_types: Dict[str, int] = field(default_factory=dict)
    content_consumption_rate: float = 0.0
    user_retention_rate: float = 0.0

@dataclass
class SystemHealthMetrics:
    """System health and performance metrics"""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_throughput: float = 0.0
    error_rate: float = 0.0
    availability: float = 100.0
    average_latency: float = 0.0

class MonitoringService:
    """
    Comprehensive monitoring service for the recommendation engine
    """
    
    def __init__(self, config: MonitoringConfiguration):
        """Initialize the monitoring service"""
        self.config = config
        self.credential = DefaultAzureCredential()
        
        # Initialize Azure clients
        self._init_clients()
        
        # Initialize alert thresholds
        self._init_alert_thresholds()
        
        logger.info("MonitoringService initialized")

    def _init_clients(self):
        """Initialize Azure monitoring clients"""
        try:
            # Application Insights client
            if self.config.application_insights_connection_string:
                self.app_insights_client = ApplicationInsightsDataClient(
                    credential=self.credential
                )
                logger.info("Application Insights client initialized")
            
            # Log Analytics client
            if self.config.log_analytics_workspace_id:
                self.logs_client = LogsQueryClient(credential=self.credential)
                logger.info("Log Analytics client initialized")
            
            # Metrics client
            self.metrics_client = MetricsQueryClient(credential=self.credential)
            
            # Storage client for metrics persistence
            if self.config.storage_account_name:
                storage_url = f"https://{self.config.storage_account_name}.blob.core.windows.net"
                self.storage_client = BlobServiceClient(
                    account_url=storage_url,
                    credential=self.credential
                )
                logger.info("Storage client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize monitoring clients: {e}")

    def _init_alert_thresholds(self):
        """Initialize default alert thresholds"""
        default_thresholds = {
            "error_rate": 5.0,  # 5% error rate
            "response_time": 2000.0,  # 2 seconds
            "cpu_utilization": 80.0,  # 80% CPU
            "memory_utilization": 85.0,  # 85% memory
            "availability": 99.0,  # 99% availability
            "cache_hit_rate": 70.0,  # 70% cache hit rate
            "recommendation_quality": 0.7  # 70% quality score
        }
        
        for key, value in default_thresholds.items():
            if key not in self.config.alert_thresholds:
                self.config.alert_thresholds[key] = value

    def collect_recommendation_metrics(
        self,
        time_range_hours: int = 24
    ) -> RecommendationMetrics:
        """Collect recommendation performance metrics"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # Query Application Insights for recommendation metrics
            query = f"""
            let timeRange = datetime({start_time.isoformat()})..datetime({end_time.isoformat()});
            union
            (
                requests
                | where timestamp between (timeRange)
                | where url contains "recommendations"
                | summarize
                    TotalRequests = count(),
                    SuccessfulRequests = countif(success == true),
                    FailedRequests = countif(success == false),
                    AvgResponseTime = avg(duration),
                    UniqueUsers = dcount(user_Id)
            ),
            (
                customEvents
                | where timestamp between (timeRange)
                | where name == "RecommendationGenerated"
                | extend algorithm = tostring(customDimensions["algorithm"])
                | summarize AlgorithmCount = count() by algorithm
            ),
            (
                customEvents
                | where timestamp between (timeRange)
                | where name == "CacheHit" or name == "CacheMiss"
                | summarize
                    CacheHits = countif(name == "CacheHit"),
                    CacheMisses = countif(name == "CacheMiss")
            )
            """
            
            # Execute query (this would be actual implementation)
            # For now, return mock data
            metrics = RecommendationMetrics(
                total_requests=15420,
                successful_requests=14891,
                failed_requests=529,
                average_response_time=456.7,
                cache_hit_rate=78.5,
                unique_users=2847,
                algorithm_distribution={
                    "hybrid": 8934,
                    "collaborative": 3821,
                    "content": 2665
                },
                recommendation_quality_score=0.847
            )
            
            logger.info(f"Collected recommendation metrics for {time_range_hours}h period")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect recommendation metrics: {e}")
            return RecommendationMetrics()

    def collect_user_engagement_metrics(
        self,
        time_range_hours: int = 24
    ) -> UserEngagementMetrics:
        """Collect user engagement metrics"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # Query for user engagement metrics
            query = f"""
            let timeRange = datetime({start_time.isoformat()})..datetime({end_time.isoformat()});
            union
            (
                customEvents
                | where timestamp between (timeRange)
                | where name == "UserInteraction"
                | extend interaction_type = tostring(customDimensions["interaction_type"])
                | summarize
                    TotalInteractions = count(),
                    UniqueUsers = dcount(user_Id),
                    InteractionsByType = count() by interaction_type
            ),
            (
                pageViews
                | where timestamp between (timeRange)
                | summarize
                    AvgSessionDuration = avg(duration),
                    BounceRate = countif(duration < 30000) * 100.0 / count()
            )
            """
            
            # Mock data for demonstration
            metrics = UserEngagementMetrics(
                total_interactions=45673,
                unique_users=8921,
                average_session_duration=247.3,
                bounce_rate=23.4,
                interaction_types={
                    "view": 28934,
                    "like": 8521,
                    "share": 3847,
                    "rating": 2934,
                    "bookmark": 1437
                },
                content_consumption_rate=67.8,
                user_retention_rate=72.5
            )
            
            logger.info(f"Collected user engagement metrics for {time_range_hours}h period")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect user engagement metrics: {e}")
            return UserEngagementMetrics()

    def collect_system_health_metrics(
        self,
        time_range_hours: int = 1
    ) -> SystemHealthMetrics:
        """Collect system health metrics"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # Query Azure Monitor for system metrics
            # This would use actual Azure Monitor queries
            
            # Mock data for demonstration
            metrics = SystemHealthMetrics(
                cpu_utilization=64.2,
                memory_utilization=71.8,
                disk_utilization=45.3,
                network_throughput=1247.6,
                error_rate=2.3,
                availability=99.7,
                average_latency=234.5
            )
            
            logger.info(f"Collected system health metrics for {time_range_hours}h period")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system health metrics: {e}")
            return SystemHealthMetrics()

    def generate_performance_dashboard(
        self,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard"""
        try:
            # Collect all metrics
            rec_metrics = self.collect_recommendation_metrics(time_range_hours)
            eng_metrics = self.collect_user_engagement_metrics(time_range_hours)
            sys_metrics = self.collect_system_health_metrics(1)
            
            # Create dashboard visualizations
            dashboard_data = {
                "recommendation_metrics": {
                    "total_requests": rec_metrics.total_requests,
                    "success_rate": (rec_metrics.successful_requests / max(rec_metrics.total_requests, 1)) * 100,
                    "average_response_time": rec_metrics.average_response_time,
                    "cache_hit_rate": rec_metrics.cache_hit_rate,
                    "unique_users": rec_metrics.unique_users,
                    "quality_score": rec_metrics.recommendation_quality_score
                },
                "user_engagement": {
                    "total_interactions": eng_metrics.total_interactions,
                    "unique_users": eng_metrics.unique_users,
                    "average_session_duration": eng_metrics.average_session_duration,
                    "bounce_rate": eng_metrics.bounce_rate,
                    "retention_rate": eng_metrics.user_retention_rate
                },
                "system_health": {
                    "cpu_utilization": sys_metrics.cpu_utilization,
                    "memory_utilization": sys_metrics.memory_utilization,
                    "disk_utilization": sys_metrics.disk_utilization,
                    "availability": sys_metrics.availability,
                    "error_rate": sys_metrics.error_rate,
                    "average_latency": sys_metrics.average_latency
                },
                "algorithm_performance": rec_metrics.algorithm_distribution,
                "interaction_breakdown": eng_metrics.interaction_types,
                "timestamp": datetime.now().isoformat(),
                "time_range_hours": time_range_hours
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate performance dashboard: {e}")
            return {}

    def create_plotly_charts(self, dashboard_data: Dict[str, Any]) -> Dict[str, str]:
        """Create Plotly charts for the dashboard"""
        try:
            charts = {}
            
            # Recommendation Success Rate Chart
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=dashboard_data["recommendation_metrics"]["success_rate"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Recommendation Success Rate (%)"},
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
            charts["success_rate"] = fig1.to_html(include_plotlyjs='cdn')
            
            # Algorithm Distribution Pie Chart
            algorithms = list(dashboard_data["algorithm_performance"].keys())
            values = list(dashboard_data["algorithm_performance"].values())
            
            fig2 = go.Figure(data=[go.Pie(
                labels=algorithms,
                values=values,
                title="Algorithm Usage Distribution"
            )])
            charts["algorithm_distribution"] = fig2.to_html(include_plotlyjs='cdn')
            
            # User Engagement Metrics
            fig3 = go.Figure()
            engagement_metrics = ["total_interactions", "unique_users", "retention_rate"]
            engagement_values = [
                dashboard_data["user_engagement"]["total_interactions"] / 1000,  # Scale for visibility
                dashboard_data["user_engagement"]["unique_users"],
                dashboard_data["user_engagement"]["retention_rate"]
            ]
            
            fig3.add_trace(go.Bar(
                x=engagement_metrics,
                y=engagement_values,
                name="User Engagement",
                marker_color=['blue', 'green', 'orange']
            ))
            fig3.update_layout(title="User Engagement Metrics")
            charts["user_engagement"] = fig3.to_html(include_plotlyjs='cdn')
            
            # System Health Dashboard
            fig4 = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Utilization', 'Memory Utilization', 'Availability', 'Error Rate'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # CPU Utilization
            fig4.add_trace(go.Indicator(
                mode="gauge+number",
                value=dashboard_data["system_health"]["cpu_utilization"],
                title={'text': "CPU %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 80], 'color': "lightgray"},
                                {'range': [80, 100], 'color': "red"}]},
                domain={'x': [0, 1], 'y': [0, 1]}
            ), row=1, col=1)
            
            # Memory Utilization
            fig4.add_trace(go.Indicator(
                mode="gauge+number",
                value=dashboard_data["system_health"]["memory_utilization"],
                title={'text': "Memory %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 85], 'color': "lightgray"},
                                {'range': [85, 100], 'color': "red"}]},
                domain={'x': [0, 1], 'y': [0, 1]}
            ), row=1, col=2)
            
            # Availability
            fig4.add_trace(go.Indicator(
                mode="gauge+number",
                value=dashboard_data["system_health"]["availability"],
                title={'text': "Availability %"},
                gauge={'axis': {'range': [90, 100]},
                       'bar': {'color': "darkred"},
                       'steps': [{'range': [90, 99], 'color': "yellow"},
                                {'range': [99, 100], 'color': "lightgreen"}]},
                domain={'x': [0, 1], 'y': [0, 1]}
            ), row=2, col=1)
            
            # Error Rate
            fig4.add_trace(go.Indicator(
                mode="gauge+number",
                value=dashboard_data["system_health"]["error_rate"],
                title={'text': "Error Rate %"},
                gauge={'axis': {'range': [0, 10]},
                       'bar': {'color': "orange"},
                       'steps': [{'range': [0, 5], 'color': "lightgreen"},
                                {'range': [5, 10], 'color': "red"}]},
                domain={'x': [0, 1], 'y': [0, 1]}
            ), row=2, col=2)
            
            fig4.update_layout(title="System Health Metrics")
            charts["system_health"] = fig4.to_html(include_plotlyjs='cdn')
            
            return charts
            
        except Exception as e:
            logger.error(f"Failed to create Plotly charts: {e}")
            return {}

    def check_alert_conditions(
        self,
        rec_metrics: RecommendationMetrics,
        eng_metrics: UserEngagementMetrics,
        sys_metrics: SystemHealthMetrics
    ) -> List[Dict[str, Any]]:
        """Check for alert conditions and return triggered alerts"""
        alerts = []
        
        try:
            # Check recommendation metrics
            error_rate = (rec_metrics.failed_requests / max(rec_metrics.total_requests, 1)) * 100
            if error_rate > self.config.alert_thresholds["error_rate"]:
                alerts.append({
                    "type": "error_rate",
                    "severity": "high",
                    "metric": "recommendation_error_rate",
                    "current_value": error_rate,
                    "threshold": self.config.alert_thresholds["error_rate"],
                    "message": f"Recommendation error rate {error_rate:.1f}% exceeds threshold {self.config.alert_thresholds['error_rate']}%"
                })
            
            if rec_metrics.average_response_time > self.config.alert_thresholds["response_time"]:
                alerts.append({
                    "type": "performance",
                    "severity": "medium",
                    "metric": "recommendation_response_time",
                    "current_value": rec_metrics.average_response_time,
                    "threshold": self.config.alert_thresholds["response_time"],
                    "message": f"Average response time {rec_metrics.average_response_time:.1f}ms exceeds threshold {self.config.alert_thresholds['response_time']}ms"
                })
            
            if rec_metrics.cache_hit_rate < self.config.alert_thresholds["cache_hit_rate"]:
                alerts.append({
                    "type": "performance",
                    "severity": "low",
                    "metric": "cache_hit_rate",
                    "current_value": rec_metrics.cache_hit_rate,
                    "threshold": self.config.alert_thresholds["cache_hit_rate"],
                    "message": f"Cache hit rate {rec_metrics.cache_hit_rate:.1f}% below threshold {self.config.alert_thresholds['cache_hit_rate']}%"
                })
            
            # Check system health metrics
            if sys_metrics.cpu_utilization > self.config.alert_thresholds["cpu_utilization"]:
                alerts.append({
                    "type": "system",
                    "severity": "high",
                    "metric": "cpu_utilization",
                    "current_value": sys_metrics.cpu_utilization,
                    "threshold": self.config.alert_thresholds["cpu_utilization"],
                    "message": f"CPU utilization {sys_metrics.cpu_utilization:.1f}% exceeds threshold {self.config.alert_thresholds['cpu_utilization']}%"
                })
            
            if sys_metrics.memory_utilization > self.config.alert_thresholds["memory_utilization"]:
                alerts.append({
                    "type": "system",
                    "severity": "high",
                    "metric": "memory_utilization",
                    "current_value": sys_metrics.memory_utilization,
                    "threshold": self.config.alert_thresholds["memory_utilization"],
                    "message": f"Memory utilization {sys_metrics.memory_utilization:.1f}% exceeds threshold {self.config.alert_thresholds['memory_utilization']}%"
                })
            
            if sys_metrics.availability < self.config.alert_thresholds["availability"]:
                alerts.append({
                    "type": "availability",
                    "severity": "critical",
                    "metric": "system_availability",
                    "current_value": sys_metrics.availability,
                    "threshold": self.config.alert_thresholds["availability"],
                    "message": f"System availability {sys_metrics.availability:.1f}% below threshold {self.config.alert_thresholds['availability']}%"
                })
            
            logger.info(f"Checked alert conditions, found {len(alerts)} active alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")
            return []

    def persist_metrics(self, dashboard_data: Dict[str, Any]) -> bool:
        """Persist metrics data to storage for historical analysis"""
        try:
            if not self.storage_client:
                logger.warning("Storage client not available, skipping metrics persistence")
                return False
            
            # Create container if it doesn't exist
            container_name = "metrics-data"
            try:
                self.storage_client.create_container(container_name)
            except Exception:
                pass  # Container might already exist
            
            # Generate blob name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"dashboard_metrics_{timestamp}.json"
            
            # Upload metrics data
            blob_client = self.storage_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            metrics_json = json.dumps(dashboard_data, indent=2, default=str)
            blob_client.upload_blob(metrics_json, overwrite=True)
            
            logger.info(f"Persisted metrics data to blob: {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
            return False

    def generate_analytics_report(
        self,
        time_range_days: int = 7
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=time_range_days)
            
            # Collect historical data (mock implementation)
            report = {
                "report_period": {
                    "start_date": start_time.isoformat(),
                    "end_date": end_time.isoformat(),
                    "days": time_range_days
                },
                "executive_summary": {
                    "total_recommendations_served": 234567,
                    "unique_users_served": 45678,
                    "average_daily_requests": 33509,
                    "overall_success_rate": 96.8,
                    "user_engagement_growth": 12.5,
                    "system_availability": 99.7
                },
                "recommendation_performance": {
                    "algorithm_effectiveness": {
                        "hybrid": {"accuracy": 0.847, "coverage": 0.923, "diversity": 0.756},
                        "collaborative": {"accuracy": 0.812, "coverage": 0.887, "diversity": 0.634},
                        "content": {"accuracy": 0.789, "coverage": 0.945, "diversity": 0.823}
                    },
                    "response_time_trends": {
                        "average": 456.7,
                        "median": 398.2,
                        "95th_percentile": 1234.5,
                        "trend": "improving"
                    },
                    "cache_performance": {
                        "hit_rate": 78.5,
                        "miss_rate": 21.5,
                        "efficiency_score": 82.3
                    }
                },
                "user_behavior_insights": {
                    "interaction_patterns": {
                        "peak_hours": [19, 20, 21],
                        "peak_days": ["friday", "saturday", "sunday"],
                        "session_patterns": "evening_heavy"
                    },
                    "content_preferences": {
                        "top_genres": ["action", "comedy", "drama"],
                        "content_discovery_rate": 34.2,
                        "recommendation_acceptance_rate": 67.8
                    },
                    "user_retention": {
                        "daily_retention": 72.5,
                        "weekly_retention": 58.3,
                        "monthly_retention": 41.7
                    }
                },
                "system_performance": {
                    "resource_utilization": {
                        "avg_cpu": 64.2,
                        "avg_memory": 71.8,
                        "avg_disk": 45.3
                    },
                    "error_analysis": {
                        "total_errors": 892,
                        "error_rate": 2.3,
                        "top_error_types": ["timeout", "validation", "external_service"]
                    },
                    "scalability_metrics": {
                        "auto_scaling_events": 23,
                        "load_balancer_efficiency": 94.6,
                        "database_performance": 98.2
                    }
                },
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Generated analytics report for {time_range_days} days")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            return {}

class MonitoringServiceFactory:
    """Factory for creating monitoring service instances"""
    
    @staticmethod
    def create_from_config(config_path: str = None) -> MonitoringService:
        """Create monitoring service from configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            # Use environment variables
            config_data = {
                "application_insights_connection_string": os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
                "log_analytics_workspace_id": os.getenv("LOG_ANALYTICS_WORKSPACE_ID"),
                "storage_account_name": os.getenv("MONITORING_STORAGE_ACCOUNT"),
                "enable_real_time_monitoring": os.getenv("ENABLE_REAL_TIME_MONITORING", "true").lower() == "true",
                "metrics_retention_days": int(os.getenv("METRICS_RETENTION_DAYS", "90"))
            }

        config = MonitoringConfiguration(**config_data)
        return MonitoringService(config)

# Example usage
if __name__ == "__main__":
    # Initialize monitoring service
    monitoring_service = MonitoringServiceFactory.create_from_config()
    
    # Generate dashboard
    dashboard_data = monitoring_service.generate_performance_dashboard()
    
    # Create visualizations
    charts = monitoring_service.create_plotly_charts(dashboard_data)
    
    # Generate analytics report
    report = monitoring_service.generate_analytics_report()
    
    print("Monitoring dashboard and analytics report generated successfully")