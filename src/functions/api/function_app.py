"""
Main API Azure Function App
Entry point for traditional and A/B tested recommendations

This is a thin entry point that imports from src packages.
All business logic is in src/recommendation/, src/ab_testing/, etc.
"""

import azure.functions as func
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import Config
from src.models.openai_models import OpenAIRecommendationRequest, UserProfile
from src.ab_testing.router import ABTestRouter

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize configuration
try:
    config = Config.load()
    if not config.validate():
        logger.error("Invalid configuration")
        raise RuntimeError("Configuration validation failed")
except Exception as e:
    logger.error(f"Configuration initialization failed: {e}")
    config = None

# Initialize A/B test router
ab_test_router = None
if config and config.ab_testing.enabled:
    try:
        ab_test_router = ABTestRouter()
        logger.info("A/B test router initialized")
    except Exception as e:
        logger.error(f"Failed to initialize A/B test router: {e}")

# Create Function App
app = func.FunctionApp()


@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint
    
    Returns:
        JSON response with service health status
    """
    try:
        health_data = {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "config": config is not None,
                "ab_testing": ab_test_router is not None
            }
        }
        
        return func.HttpResponse(
            json.dumps(health_data),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return func.HttpResponse(
            json.dumps({"status": "unhealthy", "error": str(e)}),
            mimetype="application/json",
            status_code=503
        )


@app.route(route="recommendations", methods=["GET"])
def get_recommendations_get(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get recommendations via GET request (legacy support)
    
    Query Parameters:
        user_id: str - User identifier (required)
        algorithm: str - Algorithm type (collaborative, content, hybrid)
        num_recommendations: int - Number of recommendations (default: 10)
    
    Returns:
        JSON response with recommendations
    """
    try:
        # Parse query parameters
        user_id = req.params.get('user_id')
        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "user_id parameter is required"}),
                mimetype="application/json",
                status_code=400
            )
        
        algorithm = req.params.get('algorithm', 'hybrid')
        num_recommendations = int(req.params.get('num_recommendations', 10))
        
        # Import traditional engine here to avoid circular imports
        try:
            from src.recommendation.traditional_engine import get_user_recommendations
        except ImportError:
            # Fallback if module doesn't exist yet
            logger.warning("Traditional engine not available, using fallback")
            return _get_fallback_recommendations(user_id, num_recommendations, algorithm)
        
        # Get traditional recommendations
        traditional_request = {
            "user_id": user_id,
            "algorithm": algorithm,
            "num_recommendations": num_recommendations
        }
        
        recommendations = get_user_recommendations(traditional_request)
        
        response_data = {
            "user_id": user_id,
            "algorithm": algorithm,
            "recommendations": recommendations.get("recommendations", []),
            "generated_at": datetime.utcnow().isoformat(),
            "total_count": len(recommendations.get("recommendations", [])),
            "ab_test_info": {
                "enabled": False,
                "reason": "get_method_legacy"
            }
        }
        
        return func.HttpResponse(
            json.dumps(response_data),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error in GET recommendations: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="recommendations", methods=["POST"])
async def get_recommendations_post(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get recommendations via POST request with A/B testing support
    
    Request Body:
        {
            "user_id": str,
            "user_profile": dict (optional),
            "num_recommendations": int (default: 10),
            "context": dict (optional),
            "exclude_items": list (optional),
            "enable_ab_test": bool (default: false),
            "test_name": str (optional)
        }
    
    Returns:
        JSON response with recommendations and A/B test info
    """
    start_time = datetime.utcnow()
    
    try:
        # Parse request body
        try:
            request_data = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                mimetype="application/json",
                status_code=400
            )
        
        if not request_data:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Validate required fields
        user_id = request_data.get('user_id')
        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "user_id is required"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Extract parameters
        user_profile = request_data.get('user_profile', {})
        num_recommendations = request_data.get('num_recommendations', 10)
        context = request_data.get('context', {})
        exclude_items = request_data.get('exclude_items', [])
        enable_ab_test = request_data.get('enable_ab_test', False)
        test_name = request_data.get('test_name')
        
        # Add session ID to context if provided
        context['session_id'] = request_data.get('session_id', context.get('session_id'))
        
        # Check if A/B testing is enabled and router is available
        if enable_ab_test and ab_test_router is not None:
            try:
                # Initialize A/B test router if needed
                if not hasattr(ab_test_router, 'openai_engine') or ab_test_router.openai_engine is None:
                    await ab_test_router.initialize()
                
                # Create OpenAI recommendation request
                openai_request = OpenAIRecommendationRequest(
                    user_id=user_id,
                    user_profile=UserProfile(**user_profile) if user_profile else UserProfile(),
                    num_recommendations=num_recommendations,
                    context=context,
                    exclude_items=exclude_items
                )
                
                # Route through A/B test
                ab_result = await ab_test_router.route_recommendation_request(
                    openai_request, test_name
                )
                
                # Convert recommendations to API format
                recommendations_data = []
                for rec in ab_result.recommendations:
                    rec_dict = rec.dict()
                    recommendations_data.append(rec_dict)
                
                # Calculate response time
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Format response
                response_data = {
                    "user_id": user_id,
                    "recommendations": recommendations_data,
                    "generated_at": datetime.utcnow().isoformat(),
                    "total_count": len(recommendations_data),
                    "response_time_ms": response_time,
                    "ab_test_info": {
                        "enabled": True,
                        "test_name": ab_result.test_name,
                        "variant": ab_result.variant,
                        "algorithm_used": ab_result.algorithm_used.value,
                        "session_id": ab_result.session_id
                    }
                }
                
                return func.HttpResponse(
                    json.dumps(response_data, default=str),
                    mimetype="application/json",
                    status_code=200
                )
                
            except Exception as e:
                logger.error(f"A/B test failed, falling back to traditional: {e}")
                # Fall through to traditional recommendations
        
        # Traditional recommendations (fallback or when A/B testing is disabled)
        try:
            from src.recommendation.traditional_engine import get_user_recommendations
        except ImportError:
            logger.warning("Traditional engine not available, using fallback")
            return _get_fallback_recommendations(
                user_id, num_recommendations, 
                request_data.get('algorithm', 'hybrid')
            )
        
        algorithm = request_data.get('algorithm', 'hybrid')
        
        traditional_request = {
            "user_id": user_id,
            "algorithm": algorithm,
            "num_recommendations": num_recommendations,
            "filters": {
                "exclude_items": exclude_items
            }
        }
        
        recommendations = get_user_recommendations(traditional_request)
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response_data = {
            "user_id": user_id,
            "algorithm": algorithm,
            "recommendations": recommendations.get("recommendations", []),
            "generated_at": datetime.utcnow().isoformat(),
            "total_count": len(recommendations.get("recommendations", [])),
            "response_time_ms": response_time,
            "ab_test_info": {
                "enabled": False,
                "reason": "traditional_only" if not enable_ab_test else "ab_test_unavailable"
            }
        }
        
        return func.HttpResponse(
            json.dumps(response_data),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="ab-test/status", methods=["GET"])
def get_ab_test_status(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get A/B test status and metrics
    
    Query Parameters:
        test_name: str - Test name (optional)
        days_back: int - Number of days to look back for metrics (default: 7)
    
    Returns:
        JSON response with A/B test metrics
    """
    try:
        # Check if A/B test router is available
        if ab_test_router is None:
            return func.HttpResponse(
                json.dumps({
                    "error": "A/B testing is not available",
                    "available": False
                }),
                mimetype="application/json",
                status_code=503
            )
        
        # Get query parameters
        test_name = req.params.get('test_name')
        days_back = int(req.params.get('days_back', 7))
        
        # Get test metrics
        metrics = ab_test_router.calculate_test_metrics(test_name, days_back)
        
        # Get active tests
        active_tests = ab_test_router.get_active_tests()
        
        response_data = {
            "available": True,
            "metrics": metrics,
            "active_tests": active_tests,
            "query_parameters": {
                "test_name": test_name,
                "days_back": days_back
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(response_data, default=str),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error getting A/B test status: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


def _get_fallback_recommendations(
    user_id: str, 
    num_recommendations: int, 
    algorithm: str
) -> func.HttpResponse:
    """
    Fallback recommendations when traditional engine is not available
    
    Args:
        user_id: User identifier
        num_recommendations: Number of recommendations
        algorithm: Algorithm type
    
    Returns:
        HTTP response with fallback recommendations
    """
    import random
    
    fallback_items = []
    for i in range(num_recommendations):
        fallback_items.append({
            "item_id": f"item_{random.randint(1000, 9999)}",
            "title": f"Recommended Item {i+1}",
            "score": round(random.uniform(0.5, 0.95), 3),
            "rank": i + 1,
            "source": "fallback"
        })
    
    response_data = {
        "user_id": user_id,
        "algorithm": algorithm,
        "recommendations": fallback_items,
        "generated_at": datetime.utcnow().isoformat(),
        "total_count": len(fallback_items),
        "warning": "Using fallback recommendations - traditional engine not available"
    }
    
    return func.HttpResponse(
        json.dumps(response_data),
        mimetype="application/json",
        status_code=200
    )
