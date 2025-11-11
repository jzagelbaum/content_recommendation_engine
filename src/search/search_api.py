"""
Search API Integration
======================

Azure Functions integration for search and discovery endpoints.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import azure.functions as func
import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

# Import our search and discovery services
from search_service import ContentSearchService, SearchServiceFactory
from discovery_service import ContentDiscoveryService, DiscoveryConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
search_service = None
discovery_service = None

def init_services():
    """Initialize search and discovery services"""
    global search_service, discovery_service
    
    try:
        # Initialize search service
        search_service = SearchServiceFactory.create_from_config()
        logger.info("Search service initialized")
        
        # Initialize discovery service
        discovery_config = DiscoveryConfiguration()
        discovery_service = ContentDiscoveryService(discovery_config)
        logger.info("Discovery service initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

# Initialize services on startup
init_services()

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main entry point for search API"""
    
    # Get the function route
    route = req.route_params.get('route', '')
    method = req.method
    
    try:
        # Route requests to appropriate handlers
        if route == 'search' and method == 'GET':
            return handle_search(req)
        elif route == 'suggest' and method == 'GET':
            return handle_suggestions(req)
        elif route == 'autocomplete' and method == 'GET':
            return handle_autocomplete(req)
        elif route == 'discover' and method == 'GET':
            return handle_discovery(req)
        elif route == 'facets' and method == 'GET':
            return handle_facets(req)
        elif route == 'similar' and method == 'GET':
            return handle_similar_content(req)
        elif route == 'trending-searches' and method == 'GET':
            return handle_trending_searches(req)
        elif route == 'seasonal' and method == 'GET':
            return handle_seasonal_content(req)
        elif route == 'collections' and method == 'GET':
            return handle_content_collections(req)
        elif route == 'index-content' and method == 'POST':
            return handle_index_content(req)
        elif route == 'health' and method == 'GET':
            return handle_health_check(req)
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

def handle_search(req: func.HttpRequest) -> func.HttpResponse:
    """Handle content search requests"""
    try:
        if not search_service:
            return func.HttpResponse(
                json.dumps({"error": "Search service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Parse query parameters
        query = req.params.get('q', req.params.get('query', ''))
        user_id = req.params.get('user_id')
        top = int(req.params.get('top', 20))
        skip = int(req.params.get('skip', 0))
        
        # Parse filters
        filters = {}
        filter_params = ['category', 'genre', 'language', 'year_min', 'year_max', 'rating_min']
        for param in filter_params:
            value = req.params.get(param)
            if value:
                if param in ['year_min', 'year_max']:
                    filters[param] = int(value)
                elif param == 'rating_min':
                    filters[param] = float(value)
                else:
                    filters[param] = value
        
        # Parse facets
        facets = req.params.get('facets')
        if facets:
            facets = facets.split(',')
        else:
            facets = ['category', 'genre', 'language', 'year']
        
        # Search options
        enable_semantic = req.params.get('semantic', 'true').lower() == 'true'
        enable_vector = req.params.get('vector', 'true').lower() == 'true'
        
        # Perform search
        results = search_service.search_content(
            query=query,
            user_id=int(user_id) if user_id else None,
            filters=filters if filters else None,
            facets=facets,
            top=top,
            skip=skip,
            enable_semantic=enable_semantic,
            enable_vector=enable_vector
        )
        
        return func.HttpResponse(
            json.dumps(results, default=str),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_suggestions(req: func.HttpRequest) -> func.HttpResponse:
    """Handle search suggestions"""
    try:
        if not search_service:
            return func.HttpResponse(
                json.dumps({"error": "Search service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        query = req.params.get('q', req.params.get('query', ''))
        top = int(req.params.get('top', 5))
        
        suggestions = search_service.suggest_content(query, top=top)
        
        return func.HttpResponse(
            json.dumps({"suggestions": suggestions}),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_autocomplete(req: func.HttpRequest) -> func.HttpResponse:
    """Handle autocomplete requests"""
    try:
        if not search_service:
            return func.HttpResponse(
                json.dumps({"error": "Search service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        query = req.params.get('q', req.params.get('query', ''))
        top = int(req.params.get('top', 5))
        mode = req.params.get('mode', 'oneTerm')
        
        completions = search_service.autocomplete(query, mode=mode, top=top)
        
        return func.HttpResponse(
            json.dumps({"completions": completions}),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Autocomplete failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_discovery(req: func.HttpRequest) -> func.HttpResponse:
    """Handle content discovery requests"""
    try:
        if not discovery_service:
            return func.HttpResponse(
                json.dumps({"error": "Discovery service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        user_id = req.params.get('user_id')
        discovery_mode = req.params.get('mode', 'mixed')
        sort_by = req.params.get('sort_by', 'popularity')
        sort_order = req.params.get('sort_order', 'desc')
        limit = int(req.params.get('limit', 20))
        offset = int(req.params.get('offset', 0))
        
        # Parse filters
        filters = {}
        filter_params = ['category', 'genre', 'language', 'year_min', 'year_max', 'rating_min']
        for param in filter_params:
            value = req.params.get(param)
            if value:
                if param in ['year_min', 'year_max']:
                    filters[param] = int(value)
                elif param == 'rating_min':
                    filters[param] = float(value)
                else:
                    filters[param] = value
        
        results = discovery_service.discover_content(
            user_id=int(user_id) if user_id else None,
            filters=filters if filters else None,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
            discovery_mode=discovery_mode
        )
        
        return func.HttpResponse(
            json.dumps(results, default=str),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_facets(req: func.HttpRequest) -> func.HttpResponse:
    """Handle facets requests"""
    try:
        if not search_service:
            return func.HttpResponse(
                json.dumps({"error": "Search service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Parse filters for facet refinement
        filters = {}
        filter_params = ['category', 'genre', 'language', 'year_min', 'year_max']
        for param in filter_params:
            value = req.params.get(param)
            if value:
                if param in ['year_min', 'year_max']:
                    filters[param] = int(value)
                else:
                    filters[param] = value
        
        facets = search_service.get_facets(filters if filters else None)
        
        return func.HttpResponse(
            json.dumps({"facets": facets}),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Facets request failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_similar_content(req: func.HttpRequest) -> func.HttpResponse:
    """Handle similar content requests"""
    try:
        item_id = req.params.get('item_id')
        if not item_id:
            return func.HttpResponse(
                json.dumps({"error": "item_id parameter required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        top = int(req.params.get('top', 10))
        similarity_threshold = float(req.params.get('threshold', 0.1))
        
        # Try search service first (vector similarity)
        similar_items = []
        if search_service:
            similar_items = search_service.get_similar_content(
                item_id=item_id,
                top=top,
                similarity_threshold=similarity_threshold
            )
        
        # Fallback to discovery service (content-based similarity)
        if not similar_items and discovery_service:
            similar_items = discovery_service.get_content_recommendations_by_similarity(
                item_id=int(item_id),
                limit=top,
                similarity_threshold=similarity_threshold
            )
        
        return func.HttpResponse(
            json.dumps({
                "item_id": item_id,
                "similar_items": similar_items,
                "count": len(similar_items)
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Similar content request failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_trending_searches(req: func.HttpRequest) -> func.HttpResponse:
    """Handle trending searches requests"""
    try:
        if not search_service:
            return func.HttpResponse(
                json.dumps({"error": "Search service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        time_window = int(req.params.get('hours', 24))
        top = int(req.params.get('top', 10))
        
        trending = search_service.get_trending_searches(
            time_window_hours=time_window,
            top=top
        )
        
        return func.HttpResponse(
            json.dumps({
                "trending_searches": trending,
                "time_window_hours": time_window,
                "generated_at": datetime.now().isoformat()
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Trending searches failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_seasonal_content(req: func.HttpRequest) -> func.HttpResponse:
    """Handle seasonal content requests"""
    try:
        if not discovery_service:
            return func.HttpResponse(
                json.dumps({"error": "Discovery service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        season = req.params.get('season')  # winter, spring, summer, fall
        limit = int(req.params.get('limit', 20))
        
        seasonal_content = discovery_service.get_seasonal_content(
            season=season,
            limit=limit
        )
        
        return func.HttpResponse(
            json.dumps({
                "seasonal_content": seasonal_content,
                "season": season,
                "count": len(seasonal_content)
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Seasonal content failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_content_collections(req: func.HttpRequest) -> func.HttpResponse:
    """Handle content collections requests"""
    try:
        if not discovery_service:
            return func.HttpResponse(
                json.dumps({"error": "Discovery service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        collections = discovery_service.get_content_collections()
        
        return func.HttpResponse(
            json.dumps({
                "collections": collections,
                "count": len(collections)
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Content collections failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_index_content(req: func.HttpRequest) -> func.HttpResponse:
    """Handle content indexing requests"""
    try:
        if not search_service:
            return func.HttpResponse(
                json.dumps({"error": "Search service not available"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Parse request body
        try:
            request_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if not request_body or 'content_items' not in request_body:
            return func.HttpResponse(
                json.dumps({"error": "content_items required in request body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        content_items = request_body['content_items']
        
        # Index content
        success = search_service.index_content(content_items)
        
        if success:
            return func.HttpResponse(
                json.dumps({
                    "status": "success",
                    "indexed_count": len(content_items),
                    "timestamp": datetime.now().isoformat()
                }),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({"error": "Indexing failed"}),
                status_code=500,
                mimetype="application/json"
            )
        
    except Exception as e:
        logger.error(f"Content indexing failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

def handle_health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Handle health check requests"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {}
        }
        
        # Check search service
        if search_service:
            try:
                stats = search_service.get_index_statistics()
                health_status["components"]["search_service"] = "healthy"
                health_status["search_stats"] = stats
            except Exception as e:
                health_status["components"]["search_service"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
        else:
            health_status["components"]["search_service"] = "not_initialized"
            health_status["status"] = "degraded"
        
        # Check discovery service
        if discovery_service:
            health_status["components"]["discovery_service"] = "healthy"
            health_status["content_count"] = len(discovery_service.content_items)
        else:
            health_status["components"]["discovery_service"] = "not_initialized"
            health_status["status"] = "degraded"
        
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