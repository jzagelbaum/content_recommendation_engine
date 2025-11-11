"""
OpenAI API Azure Function App
Entry point for OpenAI-powered recommendations

This Function App provides endpoints for:
- OpenAI-powered recommendations
- A/B test configuration
- Synthetic data generation
- Content analysis
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
from src.models.openai_models import (
    OpenAIRecommendationRequest, 
    ABTestConfig, 
    UserProfile, 
    ContentItem
)
from src.openai.engine import OpenAIRecommendationEngine
from src.openai.data_generator import OpenAIDataGenerator
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

# Initialize global services (lazy initialization)
openai_engine = None
data_generator = None
ab_router = None

# Create Function App
app = func.FunctionApp()


async def _ensure_services_initialized():
    """Ensure OpenAI services are initialized"""
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


@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    try:
        services_status = {
            "openai_engine": openai_engine is not None,
            "data_generator": data_generator is not None,
            "ab_router": ab_router is not None,
            "config": config is not None
        }
        
        health_data = {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status
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


@app.route(route="openai/recommendations", methods=["POST"])
async def get_openai_recommendations(req: func.HttpRequest) -> func.HttpResponse:
    """Get recommendations using OpenAI engine"""
    try:
        # Initialize services
        if not await _ensure_services_initialized():
            return func.HttpResponse(
                json.dumps({"error": "Failed to initialize OpenAI services"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Parse request
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
        
        # Create request object
        try:
            recommendation_request = OpenAIRecommendationRequest(
                user_id=user_id,
                user_profile=UserProfile(**request_data.get("user_profile", {})),
                num_recommendations=request_data.get("num_recommendations", 10),
                context=request_data.get("context", {}),
                exclude_items=request_data.get("exclude_items", [])
            )
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid request format: {e}"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Get recommendations
        try:
            recommendations = await openai_engine.get_recommendations(
                user_id=recommendation_request.user_id,
                user_profile=recommendation_request.user_profile.dict(),
                num_recommendations=recommendation_request.num_recommendations,
                context=recommendation_request.context,
                exclude_items=recommendation_request.exclude_items
            )
            
            response_data = {
                "recommendations": recommendations,
                "algorithm": "openai",
                "user_id": recommendation_request.user_id,
                "generated_at": datetime.utcnow().isoformat(),
                "request_id": request_data.get("request_id")
            }
            
            return func.HttpResponse(
                json.dumps(response_data, default=str),
                mimetype="application/json",
                status_code=200
            )
        except Exception as e:
            logger.error(f"Error getting OpenAI recommendations: {e}")
            return func.HttpResponse(
                json.dumps({"error": f"Failed to get recommendations: {e}"}),
                mimetype="application/json",
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI recommendations: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {e}"}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="openai/generate-data", methods=["POST"])
async def generate_synthetic_data(req: func.HttpRequest) -> func.HttpResponse:
    """Generate synthetic data using OpenAI"""
    try:
        # Initialize services
        if not await _ensure_services_initialized():
            return func.HttpResponse(
                json.dumps({"error": "Failed to initialize data generation services"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Parse request
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
        
        # Get generation parameters
        data_type = request_data.get("data_type", "users")
        count = min(request_data.get("count", 10), 100)  # Limit to 100
        
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
                    return func.HttpResponse(
                        json.dumps({"error": "user_ids and content_ids required for interactions"}),
                        mimetype="application/json",
                        status_code=400
                    )
                generated_data = await data_generator.generate_user_interactions(
                    user_ids, content_ids, count
                )
            else:
                return func.HttpResponse(
                    json.dumps({"error": f"Invalid data_type: {data_type}"}),
                    mimetype="application/json",
                    status_code=400
                )
            
            response_data = {
                "generated_data": generated_data,
                "data_type": data_type,
                "count": len(generated_data),
                "generated_at": datetime.utcnow().isoformat(),
                "request_id": request_data.get("request_id")
            }
            
            return func.HttpResponse(
                json.dumps(response_data, default=str),
                mimetype="application/json",
                status_code=200
            )
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return func.HttpResponse(
                json.dumps({"error": f"Failed to generate data: {e}"}),
                mimetype="application/json",
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in data generation: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {e}"}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="ab-test/configure", methods=["POST"])
async def configure_ab_test(req: func.HttpRequest) -> func.HttpResponse:
    """Configure A/B test parameters"""
    try:
        # Initialize services
        if not await _ensure_services_initialized():
            return func.HttpResponse(
                json.dumps({"error": "Failed to initialize A/B testing services"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Parse request
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
        
        # Create AB test configuration
        try:
            test_config = ABTestConfig(**request_data)
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid configuration format: {e}"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Configure the test
        try:
            success = ab_router.configure_test(test_config)
            if success:
                response_data = {
                    "message": f"A/B test '{test_config.test_name}' configured successfully",
                    "test_config": test_config.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }
                return func.HttpResponse(
                    json.dumps(response_data, default=str),
                    mimetype="application/json",
                    status_code=200
                )
            else:
                return func.HttpResponse(
                    json.dumps({"error": "Failed to configure A/B test"}),
                    mimetype="application/json",
                    status_code=400
                )
        except Exception as e:
            logger.error(f"Error configuring A/B test: {e}")
            return func.HttpResponse(
                json.dumps({"error": f"Failed to configure A/B test: {e}"}),
                mimetype="application/json",
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in A/B test configuration: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {e}"}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="ab-test/results", methods=["GET"])
async def get_ab_test_results(req: func.HttpRequest) -> func.HttpResponse:
    """Get A/B test results and metrics"""
    try:
        # Initialize services
        if not await _ensure_services_initialized():
            return func.HttpResponse(
                json.dumps({"error": "Failed to initialize A/B testing services"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Get query parameters
        test_name = req.params.get("test_name")
        days_back = int(req.params.get("days_back", 7))
        
        try:
            # Get test metrics
            metrics = ab_router.calculate_test_metrics(test_name, days_back)
            
            # Get active tests
            active_tests = ab_router.get_active_tests()
            
            response_data = {
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
            logger.error(f"Error getting A/B test results: {e}")
            return func.HttpResponse(
                json.dumps({"error": f"Failed to get A/B test results: {e}"}),
                mimetype="application/json",
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in A/B test results: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {e}"}),
            mimetype="application/json",
            status_code=500
        )
