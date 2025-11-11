"""
Azure Functions Integration for Monitoring API
==============================================

Azure Functions endpoints for monitoring and analytics APIs.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import azure.functions as func
import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Import monitoring service
from monitoring_service import MonitoringService, MonitoringServiceFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize monitoring service
monitoring_service = None

def init_monitoring_service():
    """Initialize monitoring service"""
    global monitoring_service
    
    try:
        monitoring_service = MonitoringServiceFactory.create_from_config()
        logger.info("Monitoring service initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize monitoring service: {e}")
        return False

# Initialize on startup
init_monitoring_service()

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main entry point for monitoring API"""
    
    # Get the function route
    route = req.route_params.get('route', '')
    method = req.method
    
    try:
        # Route requests to appropriate handlers
        if route == 'dashboard' and method == 'GET':
            return handle_dashboard(req)
        elif route == 'metrics' and method == 'GET':
            return handle_metrics(req)
        elif route == 'alerts' and method == 'GET':
            return handle_alerts(req)
        elif route == 'analytics-report' and method == 'GET':
            return handle_analytics_report(req)
        elif route == 'health-check' and method == 'GET':
            return handle_health_check(req)
        elif route == 'charts' and method == 'GET':
            return handle_charts(req)
        elif route == 'real-time' and method == 'GET':
            return handle_real_time_metrics(req)
        else:
            return func.HttpResponse(
                json.dumps({"error": "Not found"}),
                status_code=404,
                mimetype="application/json"
            )
            
    except Exception as e:
        logger.error(f"Request handling failed: {e}")
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

def handle_dashboard(req: func.HttpRequest) -> func.HttpResponse:
    """Handle dashboard data requests"""
    try:
        if not monitoring_service:
            return func.HttpResponse(
                json.dumps({"error": "Monitoring service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Parse query parameters
        time_range_hours = int(req.params.get('hours', 24))
        
        # Generate dashboard data
        dashboard_data = monitoring_service.generate_performance_dashboard(time_range_hours)
        
        return func.HttpResponse(
            json.dumps(dashboard_data, default=str),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Dashboard request failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_metrics(req: func.HttpRequest) -> func.HttpResponse:
    """Handle specific metrics requests"""
    try:
        if not monitoring_service:
            return func.HttpResponse(
                json.dumps({"error": "Monitoring service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        metric_type = req.params.get('type', 'all')
        time_range_hours = int(req.params.get('hours', 24))
        
        response_data = {}
        
        if metric_type in ['all', 'recommendations']:
            rec_metrics = monitoring_service.collect_recommendation_metrics(time_range_hours)
            response_data['recommendation_metrics'] = {
                "total_requests": rec_metrics.total_requests,
                "successful_requests": rec_metrics.successful_requests,
                "failed_requests": rec_metrics.failed_requests,
                "success_rate": (rec_metrics.successful_requests / max(rec_metrics.total_requests, 1)) * 100,
                "average_response_time": rec_metrics.average_response_time,
                "cache_hit_rate": rec_metrics.cache_hit_rate,
                "unique_users": rec_metrics.unique_users,
                "algorithm_distribution": rec_metrics.algorithm_distribution,
                "quality_score": rec_metrics.recommendation_quality_score
            }
        
        if metric_type in ['all', 'engagement']:
            eng_metrics = monitoring_service.collect_user_engagement_metrics(time_range_hours)
            response_data['user_engagement'] = {
                "total_interactions": eng_metrics.total_interactions,
                "unique_users": eng_metrics.unique_users,
                "average_session_duration": eng_metrics.average_session_duration,
                "bounce_rate": eng_metrics.bounce_rate,
                "interaction_types": eng_metrics.interaction_types,
                "content_consumption_rate": eng_metrics.content_consumption_rate,
                "user_retention_rate": eng_metrics.user_retention_rate
            }
        
        if metric_type in ['all', 'system']:
            sys_metrics = monitoring_service.collect_system_health_metrics(1)
            response_data['system_health'] = {
                "cpu_utilization": sys_metrics.cpu_utilization,
                "memory_utilization": sys_metrics.memory_utilization,
                "disk_utilization": sys_metrics.disk_utilization,
                "network_throughput": sys_metrics.network_throughput,
                "error_rate": sys_metrics.error_rate,
                "availability": sys_metrics.availability,
                "average_latency": sys_metrics.average_latency
            }
        
        response_data['timestamp'] = datetime.now().isoformat()
        response_data['time_range_hours'] = time_range_hours
        
        return func.HttpResponse(
            json.dumps(response_data, default=str),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Metrics request failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_alerts(req: func.HttpRequest) -> func.HttpResponse:
    """Handle alerts and notifications"""
    try:
        if not monitoring_service:
            return func.HttpResponse(
                json.dumps({"error": "Monitoring service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Collect current metrics
        rec_metrics = monitoring_service.collect_recommendation_metrics(1)
        eng_metrics = monitoring_service.collect_user_engagement_metrics(1)
        sys_metrics = monitoring_service.collect_system_health_metrics(1)
        
        # Check alert conditions
        alerts = monitoring_service.check_alert_conditions(rec_metrics, eng_metrics, sys_metrics)
        
        # Get alert severity filter
        severity_filter = req.params.get('severity')
        if severity_filter:
            alerts = [alert for alert in alerts if alert['severity'] == severity_filter]
        
        response_data = {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "alert_summary": {
                "critical": len([a for a in alerts if a['severity'] == 'critical']),
                "high": len([a for a in alerts if a['severity'] == 'high']),
                "medium": len([a for a in alerts if a['severity'] == 'medium']),
                "low": len([a for a in alerts if a['severity'] == 'low'])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(response_data, default=str),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Alerts request failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_analytics_report(req: func.HttpRequest) -> func.HttpResponse:
    """Handle analytics report generation"""
    try:
        if not monitoring_service:
            return func.HttpResponse(
                json.dumps({"error": "Monitoring service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Parse parameters
        time_range_days = int(req.params.get('days', 7))
        report_format = req.params.get('format', 'json')
        
        # Generate analytics report
        report = monitoring_service.generate_analytics_report(time_range_days)
        
        if report_format == 'json':
            return func.HttpResponse(
                json.dumps(report, default=str),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({"error": "Only JSON format supported currently"}),
                status_code=400,
                mimetype="application/json"
            )
        
    except Exception as e:
        logger.error(f"Analytics report request failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Handle monitoring service health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {}
        }
        
        # Check monitoring service
        if monitoring_service:
            health_status["components"]["monitoring_service"] = "healthy"
            
            # Test basic functionality
            try:
                dashboard_data = monitoring_service.generate_performance_dashboard(1)
                health_status["components"]["dashboard_generation"] = "healthy"
                health_status["sample_metrics"] = {
                    "recommendations_collected": len(dashboard_data.get("recommendation_metrics", {})),
                    "engagement_collected": len(dashboard_data.get("user_engagement", {})),
                    "system_collected": len(dashboard_data.get("system_health", {}))
                }
            except Exception as e:
                health_status["components"]["dashboard_generation"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
        else:
            health_status["components"]["monitoring_service"] = "not_initialized"
            health_status["status"] = "degraded"
        
        # Check dependencies
        dependencies = {
            "application_insights": monitoring_service.config.application_insights_connection_string if monitoring_service else None,
            "log_analytics": monitoring_service.config.log_analytics_workspace_id if monitoring_service else None,
            "storage_account": monitoring_service.config.storage_account_name if monitoring_service else None
        }
        
        for dep_name, dep_value in dependencies.items():
            if dep_value:
                health_status["components"][dep_name] = "configured"
            else:
                health_status["components"][dep_name] = "not_configured"
        
        return func.HttpResponse(
            json.dumps(health_status),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

def handle_charts(req: func.HttpRequest) -> func.HttpResponse:
    """Handle chart generation requests"""
    try:
        if not monitoring_service:
            return func.HttpResponse(
                json.dumps({"error": "Monitoring service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Parse parameters
        time_range_hours = int(req.params.get('hours', 24))
        chart_type = req.params.get('type', 'all')
        
        # Generate dashboard data
        dashboard_data = monitoring_service.generate_performance_dashboard(time_range_hours)
        
        # Generate charts
        charts = monitoring_service.create_plotly_charts(dashboard_data)
        
        # Filter charts if specific type requested
        if chart_type != 'all' and chart_type in charts:
            charts = {chart_type: charts[chart_type]}
        
        response_data = {
            "charts": charts,
            "chart_count": len(charts),
            "available_types": list(charts.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(response_data, default=str),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Charts request failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_real_time_metrics(req: func.HttpRequest) -> func.HttpResponse:
    """Handle real-time metrics streaming"""
    try:
        if not monitoring_service:
            return func.HttpResponse(
                json.dumps({"error": "Monitoring service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Collect real-time metrics (last 5 minutes)
        rec_metrics = monitoring_service.collect_recommendation_metrics(0.083)  # 5 minutes
        sys_metrics = monitoring_service.collect_system_health_metrics(0.083)
        
        # Real-time data
        real_time_data = {
            "timestamp": datetime.now().isoformat(),
            "requests_per_minute": rec_metrics.total_requests / 5 if rec_metrics.total_requests else 0,
            "success_rate": (rec_metrics.successful_requests / max(rec_metrics.total_requests, 1)) * 100,
            "average_response_time": rec_metrics.average_response_time,
            "cache_hit_rate": rec_metrics.cache_hit_rate,
            "cpu_utilization": sys_metrics.cpu_utilization,
            "memory_utilization": sys_metrics.memory_utilization,
            "error_rate": sys_metrics.error_rate,
            "availability": sys_metrics.availability,
            "active_connections": rec_metrics.unique_users
        }
        
        return func.HttpResponse(
            json.dumps(real_time_data, default=str),
            status_code=200,
            mimetype="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Real-time metrics request failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )