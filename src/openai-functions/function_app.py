"""
Azure Function App for OpenAI-powered recommendation system
Provides HTTP endpoints for OpenAI recommendations, A/B testing, and data generation
"""

import azure.functions as func
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import OpenAI recommendation system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.openai_models import (
    OpenAIRecommendationRequest, ABTestConfig, UserProfile, ContentItem
)
from openai.openai_recommendation_engine import OpenAIRecommendationEngine
from openai.data_generator import OpenAIDataGenerator
from api.ab_test_router import ABTestRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global instances (will be properly initialized on first request)
openai_engine = None
data_generator = None
ab_router = None

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

async def initialize_services():
    """Initialize OpenAI services if not already initialized"""
    global openai_engine, data_generator, ab_router
    
    try:
        if openai_engine is None:
            openai_engine = OpenAIRecommendationEngine()
            await openai_engine.initialize()
            logger.info("OpenAI recommendation engine initialized")
        
        if data_generator is None:
            data_generator = OpenAIDataGenerator()
            await data_generator.initialize()
            logger.info("OpenAI data generator initialized")
        
        if ab_router is None:
            ab_router = ABTestRouter()
            await ab_router.initialize()
            logger.info("A/B test router initialized")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        return False

def create_error_response(message: str, status_code: int = 500) -> func.HttpResponse:
    """Create standardized error response"""
    return func.HttpResponse(
        json.dumps({
            "error": message,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "error"
        }),
        status_code=status_code,
        headers={"Content-Type": "application/json"}
    )

def create_success_response(data: Dict[str, Any], status_code: int = 200) -> func.HttpResponse:
    """Create standardized success response"""
    response_data = {
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "success"
    }
    return func.HttpResponse(
        json.dumps(response_data, default=str),
        status_code=status_code,
        headers={"Content-Type": "application/json"}
    )

@app.route(route="health", methods=["GET"])
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    try:
        # Check if services are initialized
        services_status = {
            "openai_engine": openai_engine is not None,
            "data_generator": data_generator is not None,
            "ab_router": ab_router is not None
        }
        
        return create_success_response({
            "message": "OpenAI Function App is healthy",
            "services": services_status,
            "version": "1.0.0"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return create_error_response(f"Health check failed: {e}")

@app.route(route="openai/recommendations", methods=["POST"])
async def get_openai_recommendations(req: func.HttpRequest) -> func.HttpResponse:
    """Get recommendations using OpenAI engine"""
    try:
        # Initialize services
        if not await initialize_services():
            return create_error_response("Failed to initialize OpenAI services", 503)
        
        # Parse request
        try:
            request_data = req.get_json()
        except ValueError:
            return create_error_response("Invalid JSON in request body", 400)
        
        if not request_data:
            return create_error_response("Request body is required", 400)
        
        # Validate required fields
        required_fields = ["user_id"]
        for field in required_fields:
            if field not in request_data:
                return create_error_response(f"Missing required field: {field}", 400)
        
        # Create request object
        try:
            recommendation_request = OpenAIRecommendationRequest(
                user_id=request_data["user_id"],
                user_profile=UserProfile(**request_data.get("user_profile", {})),
                num_recommendations=request_data.get("num_recommendations", 10),
                context=request_data.get("context", {}),
                exclude_items=request_data.get("exclude_items", [])
            )
        except Exception as e:
            return create_error_response(f"Invalid request format: {e}", 400)
        
        # Get recommendations
        try:
            recommendations = await openai_engine.get_recommendations(
                user_id=recommendation_request.user_id,
                user_profile=recommendation_request.user_profile.dict(),
                num_recommendations=recommendation_request.num_recommendations,
                context=recommendation_request.context,
                exclude_items=recommendation_request.exclude_items
            )
            
            return create_success_response({
                "recommendations": recommendations,
                "algorithm": "openai",
                "user_id": recommendation_request.user_id,
                "request_id": request_data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error getting OpenAI recommendations: {e}")
            return create_error_response(f"Failed to get recommendations: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI recommendations: {e}")
        return create_error_response(f"Internal server error: {e}")

@app.route(route="openai/ab-test", methods=["POST"])
async def get_ab_test_recommendations(req: func.HttpRequest) -> func.HttpResponse:
    """Get recommendations through A/B testing router"""
    try:
        # Initialize services
        if not await initialize_services():
            return create_error_response("Failed to initialize A/B testing services", 503)
        
        # Parse request
        try:
            request_data = req.get_json()
        except ValueError:
            return create_error_response("Invalid JSON in request body", 400)
        
        if not request_data:
            return create_error_response("Request body is required", 400)
        
        # Validate required fields
        required_fields = ["user_id"]
        for field in required_fields:
            if field not in request_data:
                return create_error_response(f"Missing required field: {field}", 400)
        
        # Create request object
        try:
            recommendation_request = OpenAIRecommendationRequest(
                user_id=request_data["user_id"],
                user_profile=UserProfile(**request_data.get("user_profile", {})),
                num_recommendations=request_data.get("num_recommendations", 10),
                context=request_data.get("context", {}),
                exclude_items=request_data.get("exclude_items", [])
            )
        except Exception as e:
            return create_error_response(f"Invalid request format: {e}", 400)
        
        # Route through A/B test
        try:
            test_name = request_data.get("test_name")
            ab_result = await ab_router.route_recommendation_request(
                recommendation_request, 
                test_name
            )
            
            # Convert recommendations to serializable format
            recommendations_data = []
            for rec in ab_result.recommendations:
                rec_dict = rec.dict()
                recommendations_data.append(rec_dict)
            
            return create_success_response({
                "recommendations": recommendations_data,
                "ab_test_info": {
                    "test_name": ab_result.test_name,
                    "variant": ab_result.variant,
                    "algorithm_used": ab_result.algorithm_used.value,
                    "session_id": ab_result.session_id,
                    "response_time_ms": ab_result.response_time_ms
                },
                "user_id": ab_result.user_id,
                "request_id": request_data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error in A/B test routing: {e}")
            return create_error_response(f"Failed to get A/B test recommendations: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in A/B test recommendations: {e}")
        return create_error_response(f"Internal server error: {e}")

@app.route(route="openai/generate-data", methods=["POST"])
async def generate_synthetic_data(req: func.HttpRequest) -> func.HttpResponse:
    """Generate synthetic data using OpenAI"""
    try:
        # Initialize services
        if not await initialize_services():
            return create_error_response("Failed to initialize data generation services", 503)
        
        # Parse request
        try:
            request_data = req.get_json()
        except ValueError:
            return create_error_response("Invalid JSON in request body", 400)
        
        if not request_data:
            return create_error_response("Request body is required", 400)
        
        # Get generation parameters
        data_type = request_data.get("data_type", "users")  # users, content, interactions
        count = min(request_data.get("count", 10), 100)  # Limit to 100 items
        
        try:
            if data_type == "users":
                generated_data = await data_generator.generate_user_profiles(count)
            elif data_type == "content":
                content_type = request_data.get("content_type", "movie")
                generated_data = await data_generator.generate_content_items(count, content_type)
            elif data_type == "interactions":
                user_ids = request_data.get("user_ids", [])
                content_ids = request_data.get("content_ids", [])
                if not user_ids or not content_ids:
                    return create_error_response(
                        "user_ids and content_ids are required for interaction generation", 400
                    )
                generated_data = await data_generator.generate_user_interactions(
                    user_ids, content_ids, count
                )
            else:
                return create_error_response(
                    f"Invalid data_type. Must be one of: users, content, interactions", 400
                )
            
            return create_success_response({
                "generated_data": generated_data,
                "data_type": data_type,
                "count": len(generated_data),
                "request_id": request_data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return create_error_response(f"Failed to generate data: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in data generation: {e}")
        return create_error_response(f"Internal server error: {e}")

@app.route(route="ab-test/configure", methods=["POST"])
async def configure_ab_test(req: func.HttpRequest) -> func.HttpResponse:
    """Configure A/B test parameters"""
    try:
        # Initialize services
        if not await initialize_services():
            return create_error_response("Failed to initialize A/B testing services", 503)
        
        # Parse request
        try:
            request_data = req.get_json()
        except ValueError:
            return create_error_response("Invalid JSON in request body", 400)
        
        if not request_data:
            return create_error_response("Request body is required", 400)
        
        # Create AB test configuration
        try:
            config = ABTestConfig(**request_data)
        except Exception as e:
            return create_error_response(f"Invalid configuration format: {e}", 400)
        
        # Configure the test
        try:
            success = ab_router.configure_test(config)
            if success:
                return create_success_response({
                    "message": f"A/B test '{config.test_name}' configured successfully",
                    "test_config": config.dict()
                })
            else:
                return create_error_response("Failed to configure A/B test", 400)
        except Exception as e:
            logger.error(f"Error configuring A/B test: {e}")
            return create_error_response(f"Failed to configure A/B test: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in A/B test configuration: {e}")
        return create_error_response(f"Internal server error: {e}")

@app.route(route="ab-test/results", methods=["GET"])
async def get_ab_test_results(req: func.HttpRequest) -> func.HttpResponse:
    """Get A/B test results and metrics"""
    try:
        # Initialize services
        if not await initialize_services():
            return create_error_response("Failed to initialize A/B testing services", 503)
        
        # Get query parameters
        test_name = req.params.get("test_name")
        days_back = int(req.params.get("days_back", 7))
        
        try:
            # Get test metrics
            metrics = ab_router.calculate_test_metrics(test_name, days_back)
            
            # Get active tests
            active_tests = ab_router.get_active_tests()
            
            return create_success_response({
                "metrics": metrics,
                "active_tests": active_tests,
                "query_parameters": {
                    "test_name": test_name,
                    "days_back": days_back
                }
            })
        except Exception as e:
            logger.error(f"Error getting A/B test results: {e}")
            return create_error_response(f"Failed to get A/B test results: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in A/B test results: {e}")
        return create_error_response(f"Internal server error: {e}")

@app.route(route="ab-test/stop", methods=["POST"])
async def stop_ab_test(req: func.HttpRequest) -> func.HttpResponse:
    """Stop an active A/B test"""
    try:
        # Initialize services
        if not await initialize_services():
            return create_error_response("Failed to initialize A/B testing services", 503)
        
        # Parse request
        try:
            request_data = req.get_json()
        except ValueError:
            return create_error_response("Invalid JSON in request body", 400)
        
        if not request_data:
            return create_error_response("Request body is required", 400)
        
        test_name = request_data.get("test_name")
        if not test_name:
            return create_error_response("test_name is required", 400)
        
        try:
            success = ab_router.stop_test(test_name)
            if success:
                return create_success_response({
                    "message": f"A/B test '{test_name}' stopped successfully",
                    "test_name": test_name
                })
            else:
                return create_error_response(f"Test '{test_name}' not found or already stopped", 404)
        except Exception as e:
            logger.error(f"Error stopping A/B test: {e}")
            return create_error_response(f"Failed to stop A/B test: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error stopping A/B test: {e}")
        return create_error_response(f"Internal server error: {e}")

@app.route(route="openai/analyze-content", methods=["POST"])
async def analyze_content(req: func.HttpRequest) -> func.HttpResponse:
    """Analyze content using OpenAI for insights and categorization"""
    try:
        # Initialize services
        if not await initialize_services():
            return create_error_response("Failed to initialize OpenAI services", 503)
        
        # Parse request
        try:
            request_data = req.get_json()
        except ValueError:
            return create_error_response("Invalid JSON in request body", 400)
        
        if not request_data:
            return create_error_response("Request body is required", 400)
        
        # Validate required fields
        content_text = request_data.get("content")
        if not content_text:
            return create_error_response("content field is required", 400)
        
        try:
            # Analyze content using OpenAI engine
            analysis = await openai_engine.openai_service.analyze_content(content_text)
            
            return create_success_response({
                "analysis": analysis,
                "content_length": len(content_text),
                "request_id": request_data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return create_error_response(f"Failed to analyze content: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in content analysis: {e}")
        return create_error_response(f"Internal server error: {e}")

if __name__ == "__main__":
    # For local development
    app.run()