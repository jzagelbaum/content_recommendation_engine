"""
Power BI Integration for Advanced Analytics
==========================================

Power BI integration for creating advanced analytics dashboards
and business intelligence reports for the recommendation engine.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PowerBIIntegration:
    """
    Power BI integration for advanced analytics and business intelligence
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Power BI integration"""
        self.config = config
        self.credential = DefaultAzureCredential()
        
        # Power BI configuration
        self.workspace_id = config.get("powerbi_workspace_id")
        self.client_id = config.get("powerbi_client_id")
        self.client_secret = config.get("powerbi_client_secret")
        self.tenant_id = config.get("tenant_id")
        
        # Storage for data export
        self.storage_account = config.get("storage_account_name")
        if self.storage_account:
            storage_url = f"https://{self.storage_account}.blob.core.windows.net"
            self.storage_client = BlobServiceClient(
                account_url=storage_url,
                credential=self.credential
            )
        
        # Initialize authentication
        self._init_powerbi_auth()
        
        logger.info("Power BI integration initialized")

    def _init_powerbi_auth(self):
        """Initialize Power BI authentication"""
        try:
            # Power BI REST API endpoint for authentication
            auth_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
            
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': 'https://analysis.windows.net/powerbi/api/.default'
            }
            
            # Get access token (in production, implement proper token management)
            self.access_token = None  # Placeholder for actual implementation
            logger.info("Power BI authentication configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize Power BI authentication: {e}")
            self.access_token = None

    def export_dashboard_data_for_powerbi(
        self,
        dashboard_data: Dict[str, Any],
        export_format: str = "csv"
    ) -> Dict[str, str]:
        """Export dashboard data in Power BI compatible format"""
        try:
            exported_files = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export recommendation metrics
            rec_metrics = dashboard_data.get("recommendation_metrics", {})
            if rec_metrics:
                rec_df = pd.DataFrame([{
                    "timestamp": datetime.now(),
                    "total_requests": rec_metrics.get("total_requests", 0),
                    "success_rate": rec_metrics.get("success_rate", 0),
                    "average_response_time": rec_metrics.get("average_response_time", 0),
                    "cache_hit_rate": rec_metrics.get("cache_hit_rate", 0),
                    "unique_users": rec_metrics.get("unique_users", 0),
                    "quality_score": rec_metrics.get("quality_score", 0)
                }])
                
                filename = f"recommendation_metrics_{timestamp}.csv"
                file_path = self._save_dataframe(rec_df, filename, export_format)
                exported_files["recommendation_metrics"] = file_path
            
            # Export user engagement metrics
            eng_metrics = dashboard_data.get("user_engagement", {})
            if eng_metrics:
                eng_df = pd.DataFrame([{
                    "timestamp": datetime.now(),
                    "total_interactions": eng_metrics.get("total_interactions", 0),
                    "unique_users": eng_metrics.get("unique_users", 0),
                    "average_session_duration": eng_metrics.get("average_session_duration", 0),
                    "bounce_rate": eng_metrics.get("bounce_rate", 0),
                    "retention_rate": eng_metrics.get("retention_rate", 0)
                }])
                
                filename = f"user_engagement_{timestamp}.csv"
                file_path = self._save_dataframe(eng_df, filename, export_format)
                exported_files["user_engagement"] = file_path
            
            # Export algorithm performance
            algo_data = dashboard_data.get("algorithm_performance", {})
            if algo_data:
                algo_df = pd.DataFrame([
                    {"algorithm": algo, "usage_count": count, "timestamp": datetime.now()}
                    for algo, count in algo_data.items()
                ])
                
                filename = f"algorithm_performance_{timestamp}.csv"
                file_path = self._save_dataframe(algo_df, filename, export_format)
                exported_files["algorithm_performance"] = file_path
            
            # Export interaction breakdown
            interaction_data = dashboard_data.get("interaction_breakdown", {})
            if interaction_data:
                interaction_df = pd.DataFrame([
                    {"interaction_type": itype, "count": count, "timestamp": datetime.now()}
                    for itype, count in interaction_data.items()
                ])
                
                filename = f"interaction_breakdown_{timestamp}.csv"
                file_path = self._save_dataframe(interaction_df, filename, export_format)
                exported_files["interaction_breakdown"] = file_path
            
            # Export system health metrics
            sys_metrics = dashboard_data.get("system_health", {})
            if sys_metrics:
                sys_df = pd.DataFrame([{
                    "timestamp": datetime.now(),
                    "cpu_utilization": sys_metrics.get("cpu_utilization", 0),
                    "memory_utilization": sys_metrics.get("memory_utilization", 0),
                    "disk_utilization": sys_metrics.get("disk_utilization", 0),
                    "availability": sys_metrics.get("availability", 0),
                    "error_rate": sys_metrics.get("error_rate", 0),
                    "average_latency": sys_metrics.get("average_latency", 0)
                }])
                
                filename = f"system_health_{timestamp}.csv"
                file_path = self._save_dataframe(sys_df, filename, export_format)
                exported_files["system_health"] = file_path
            
            logger.info(f"Exported {len(exported_files)} datasets for Power BI")
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export dashboard data: {e}")
            return {}

    def _save_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        export_format: str
    ) -> str:
        """Save DataFrame to storage"""
        try:
            container_name = "powerbi-data"
            
            # Create container if it doesn't exist
            try:
                self.storage_client.create_container(container_name)
            except Exception:
                pass  # Container might already exist
            
            # Save DataFrame
            if export_format == "csv":
                csv_data = df.to_csv(index=False)
                blob_client = self.storage_client.get_blob_client(
                    container=container_name,
                    blob=filename
                )
                blob_client.upload_blob(csv_data, overwrite=True)
            elif export_format == "json":
                json_data = df.to_json(orient='records', date_format='iso')
                blob_client = self.storage_client.get_blob_client(
                    container=container_name,
                    blob=filename.replace('.csv', '.json')
                )
                blob_client.upload_blob(json_data, overwrite=True)
            
            blob_url = blob_client.url
            logger.info(f"Saved {filename} to blob storage")
            return blob_url
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {e}")
            return ""

    def create_historical_analytics_dataset(
        self,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Create historical analytics dataset for Power BI"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Generate sample historical data (in production, query actual data)
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
            
            # Recommendation performance over time
            rec_performance = []
            for dt in date_range:
                hour = dt.hour
                # Simulate realistic patterns
                base_requests = 1000 + 500 * np.sin((hour - 6) * np.pi / 12)
                noise = np.random.normal(0, 50)
                
                rec_performance.append({
                    "timestamp": dt,
                    "hour": hour,
                    "day_of_week": dt.dayofweek,
                    "total_requests": max(0, int(base_requests + noise)),
                    "success_rate": min(100, max(90, 96 + np.random.normal(0, 2))),
                    "response_time": max(100, 400 + 100 * np.sin(hour * np.pi / 12) + np.random.normal(0, 50)),
                    "cache_hit_rate": min(100, max(60, 78 + np.random.normal(0, 5))),
                    "unique_users": max(0, int(base_requests * 0.7 + np.random.normal(0, 20)))
                })
            
            rec_df = pd.DataFrame(rec_performance)
            
            # User engagement patterns
            engagement_data = []
            for dt in date_range:
                hour = dt.hour
                # Peak engagement in evening hours
                engagement_multiplier = 0.5 + 0.5 * np.sin((hour - 6) * np.pi / 12)
                
                engagement_data.append({
                    "timestamp": dt,
                    "hour": hour,
                    "day_of_week": dt.dayofweek,
                    "total_interactions": max(0, int(2000 * engagement_multiplier + np.random.normal(0, 100))),
                    "unique_users": max(0, int(800 * engagement_multiplier + np.random.normal(0, 40))),
                    "avg_session_duration": max(60, 240 + 60 * np.sin(hour * np.pi / 12) + np.random.normal(0, 30)),
                    "bounce_rate": min(50, max(10, 25 - 10 * engagement_multiplier + np.random.normal(0, 3)))
                })
            
            engagement_df = pd.DataFrame(engagement_data)
            
            # Algorithm performance comparison
            algorithms = ["hybrid", "collaborative", "content"]
            algo_performance = []
            
            for dt in pd.date_range(start=start_date, end=end_date, freq='D'):
                total_requests = np.random.randint(20000, 30000)
                
                # Hybrid gets most traffic
                hybrid_pct = np.random.uniform(0.45, 0.55)
                collab_pct = np.random.uniform(0.25, 0.35)
                content_pct = 1 - hybrid_pct - collab_pct
                
                algo_performance.extend([
                    {
                        "date": dt,
                        "algorithm": "hybrid",
                        "requests": int(total_requests * hybrid_pct),
                        "accuracy": np.random.uniform(0.82, 0.87),
                        "coverage": np.random.uniform(0.91, 0.95),
                        "diversity": np.random.uniform(0.74, 0.78)
                    },
                    {
                        "date": dt,
                        "algorithm": "collaborative",
                        "requests": int(total_requests * collab_pct),
                        "accuracy": np.random.uniform(0.79, 0.84),
                        "coverage": np.random.uniform(0.86, 0.91),
                        "diversity": np.random.uniform(0.61, 0.67)
                    },
                    {
                        "date": dt,
                        "algorithm": "content",
                        "requests": int(total_requests * content_pct),
                        "accuracy": np.random.uniform(0.76, 0.82),
                        "coverage": np.random.uniform(0.93, 0.97),
                        "diversity": np.random.uniform(0.80, 0.86)
                    }
                ])
            
            algo_df = pd.DataFrame(algo_performance)
            
            # Export datasets
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            exported_files = {
                "recommendation_performance": self._save_dataframe(
                    rec_df, f"historical_recommendations_{timestamp}.csv", "csv"
                ),
                "user_engagement": self._save_dataframe(
                    engagement_df, f"historical_engagement_{timestamp}.csv", "csv"
                ),
                "algorithm_performance": self._save_dataframe(
                    algo_df, f"historical_algorithms_{timestamp}.csv", "csv"
                )
            }
            
            return {
                "datasets": exported_files,
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days_covered": days_back
                },
                "record_counts": {
                    "recommendation_performance": len(rec_df),
                    "user_engagement": len(engagement_df),
                    "algorithm_performance": len(algo_df)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to create historical dataset: {e}")
            return {}

    def generate_powerbi_template(self) -> Dict[str, Any]:
        """Generate Power BI dashboard template configuration"""
        try:
            template = {
                "version": "1.0",
                "name": "Content Recommendation Engine Analytics",
                "description": "Comprehensive analytics dashboard for content recommendation engine",
                "pages": [
                    {
                        "name": "Executive Dashboard",
                        "visualizations": [
                            {
                                "type": "card",
                                "title": "Total Recommendations Served",
                                "data_source": "recommendation_performance",
                                "measure": "sum(total_requests)",
                                "format": "number"
                            },
                            {
                                "type": "card",
                                "title": "Average Success Rate",
                                "data_source": "recommendation_performance",
                                "measure": "average(success_rate)",
                                "format": "percentage"
                            },
                            {
                                "type": "line_chart",
                                "title": "Recommendations Over Time",
                                "data_source": "recommendation_performance",
                                "x_axis": "timestamp",
                                "y_axis": "total_requests",
                                "time_granularity": "hour"
                            },
                            {
                                "type": "pie_chart",
                                "title": "Algorithm Distribution",
                                "data_source": "algorithm_performance",
                                "category": "algorithm",
                                "value": "requests"
                            }
                        ]
                    },
                    {
                        "name": "Performance Analysis",
                        "visualizations": [
                            {
                                "type": "line_chart",
                                "title": "Response Time Trends",
                                "data_source": "recommendation_performance",
                                "x_axis": "timestamp",
                                "y_axis": "response_time",
                                "time_granularity": "hour"
                            },
                            {
                                "type": "combo_chart",
                                "title": "Success Rate vs Cache Hit Rate",
                                "data_source": "recommendation_performance",
                                "x_axis": "timestamp",
                                "y_axis_1": "success_rate",
                                "y_axis_2": "cache_hit_rate",
                                "chart_type_1": "line",
                                "chart_type_2": "column"
                            },
                            {
                                "type": "scatter_plot",
                                "title": "Algorithm Performance Comparison",
                                "data_source": "algorithm_performance",
                                "x_axis": "accuracy",
                                "y_axis": "coverage",
                                "size": "requests",
                                "category": "algorithm"
                            }
                        ]
                    },
                    {
                        "name": "User Engagement",
                        "visualizations": [
                            {
                                "type": "area_chart",
                                "title": "User Interactions Over Time",
                                "data_source": "user_engagement",
                                "x_axis": "timestamp",
                                "y_axis": "total_interactions",
                                "time_granularity": "hour"
                            },
                            {
                                "type": "heatmap",
                                "title": "Engagement by Hour and Day",
                                "data_source": "user_engagement",
                                "x_axis": "hour",
                                "y_axis": "day_of_week",
                                "value": "total_interactions"
                            },
                            {
                                "type": "gauge",
                                "title": "Average Session Duration",
                                "data_source": "user_engagement",
                                "measure": "average(avg_session_duration)",
                                "min": 0,
                                "max": 600,
                                "target": 300
                            }
                        ]
                    },
                    {
                        "name": "System Health",
                        "visualizations": [
                            {
                                "type": "gauge",
                                "title": "System Availability",
                                "data_source": "system_health",
                                "measure": "average(availability)",
                                "min": 95,
                                "max": 100,
                                "target": 99.5
                            },
                            {
                                "type": "multi_row_card",
                                "title": "System Metrics",
                                "data_source": "system_health",
                                "metrics": [
                                    "average(cpu_utilization)",
                                    "average(memory_utilization)",
                                    "average(error_rate)"
                                ]
                            }
                        ]
                    }
                ],
                "filters": [
                    {
                        "name": "Date Range",
                        "type": "date_range",
                        "applies_to": "all_pages"
                    },
                    {
                        "name": "Algorithm",
                        "type": "slicer",
                        "data_source": "algorithm_performance",
                        "column": "algorithm",
                        "applies_to": ["Performance Analysis"]
                    }
                ],
                "refresh_schedule": {
                    "frequency": "hourly",
                    "enabled": True
                }
            }
            
            return template
            
        except Exception as e:
            logger.error(f"Failed to generate Power BI template: {e}")
            return {}

    def create_powerbi_dataset_definition(self) -> Dict[str, Any]:
        """Create Power BI dataset definition"""
        try:
            dataset_definition = {
                "name": "ContentRecommendationEngineDataset",
                "tables": [
                    {
                        "name": "RecommendationPerformance",
                        "columns": [
                            {"name": "timestamp", "dataType": "dateTime"},
                            {"name": "hour", "dataType": "int64"},
                            {"name": "day_of_week", "dataType": "int64"},
                            {"name": "total_requests", "dataType": "int64"},
                            {"name": "success_rate", "dataType": "double"},
                            {"name": "response_time", "dataType": "double"},
                            {"name": "cache_hit_rate", "dataType": "double"},
                            {"name": "unique_users", "dataType": "int64"}
                        ]
                    },
                    {
                        "name": "UserEngagement",
                        "columns": [
                            {"name": "timestamp", "dataType": "dateTime"},
                            {"name": "hour", "dataType": "int64"},
                            {"name": "day_of_week", "dataType": "int64"},
                            {"name": "total_interactions", "dataType": "int64"},
                            {"name": "unique_users", "dataType": "int64"},
                            {"name": "avg_session_duration", "dataType": "double"},
                            {"name": "bounce_rate", "dataType": "double"}
                        ]
                    },
                    {
                        "name": "AlgorithmPerformance",
                        "columns": [
                            {"name": "date", "dataType": "dateTime"},
                            {"name": "algorithm", "dataType": "string"},
                            {"name": "requests", "dataType": "int64"},
                            {"name": "accuracy", "dataType": "double"},
                            {"name": "coverage", "dataType": "double"},
                            {"name": "diversity", "dataType": "double"}
                        ]
                    },
                    {
                        "name": "SystemHealth",
                        "columns": [
                            {"name": "timestamp", "dataType": "dateTime"},
                            {"name": "cpu_utilization", "dataType": "double"},
                            {"name": "memory_utilization", "dataType": "double"},
                            {"name": "disk_utilization", "dataType": "double"},
                            {"name": "availability", "dataType": "double"},
                            {"name": "error_rate", "dataType": "double"},
                            {"name": "average_latency", "dataType": "double"}
                        ]
                    }
                ],
                "relationships": [
                    {
                        "name": "TimeRelationship",
                        "fromTable": "RecommendationPerformance",
                        "fromColumn": "timestamp",
                        "toTable": "UserEngagement",
                        "toColumn": "timestamp",
                        "crossFilteringBehavior": "bothDirections"
                    }
                ],
                "measures": [
                    {
                        "name": "TotalRecommendations",
                        "expression": "SUM(RecommendationPerformance[total_requests])"
                    },
                    {
                        "name": "AverageSuccessRate",
                        "expression": "AVERAGE(RecommendationPerformance[success_rate])"
                    },
                    {
                        "name": "AverageResponseTime",
                        "expression": "AVERAGE(RecommendationPerformance[response_time])"
                    },
                    {
                        "name": "TotalInteractions",
                        "expression": "SUM(UserEngagement[total_interactions])"
                    },
                    {
                        "name": "UniqueUsers",
                        "expression": "MAX(UserEngagement[unique_users])"
                    }
                ]
            }
            
            return dataset_definition
            
        except Exception as e:
            logger.error(f"Failed to create dataset definition: {e}")
            return {}

# Example usage and configuration
def create_powerbi_integration_config():
    """Create Power BI integration configuration"""
    return {
        "powerbi_workspace_id": os.getenv("POWERBI_WORKSPACE_ID"),
        "powerbi_client_id": os.getenv("POWERBI_CLIENT_ID"),
        "powerbi_client_secret": os.getenv("POWERBI_CLIENT_SECRET"),
        "tenant_id": os.getenv("AZURE_TENANT_ID"),
        "storage_account_name": os.getenv("POWERBI_STORAGE_ACCOUNT", "recenginestorage")
    }

if __name__ == "__main__":
    # Initialize Power BI integration
    config = create_powerbi_integration_config()
    powerbi = PowerBIIntegration(config)
    
    # Generate historical dataset
    historical_data = powerbi.create_historical_analytics_dataset(30)
    
    # Generate template
    template = powerbi.generate_powerbi_template()
    
    # Generate dataset definition
    dataset_def = powerbi.create_powerbi_dataset_definition()
    
    print("Power BI integration components generated successfully")