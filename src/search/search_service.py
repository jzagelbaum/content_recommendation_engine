"""
Search and Discovery Service
============================

Azure Cognitive Search integration for content discovery, semantic search,
and faceted navigation capabilities.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import (
    VectorizedQuery, 
    QueryType, 
    QueryCaptionType, 
    QueryAnswerType
)
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    ComplexField,
    CorsOptions,
    SearchIndexerSkillset,
    SearchIndexerDataSourceConnection,
    SearchIndexer,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmKind,
    HnswAlgorithmConfiguration
)
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import openai
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchConfiguration:
    """Search service configuration"""
    service_name: str
    api_key: str
    index_name: str = "content-index"
    vector_dimension: int = 1536
    enable_semantic_search: bool = True
    enable_vector_search: bool = True

class ContentSearchService:
    """
    Azure Cognitive Search service for content discovery and semantic search
    """
    
    def __init__(self, config: SearchConfiguration):
        """Initialize the search service"""
        self.config = config
        self.endpoint = f"https://{config.service_name}.search.windows.net"
        self.credential = AzureKeyCredential(config.api_key)
        
        # Initialize clients
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=config.index_name,
            credential=self.credential
        )
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        
        # Initialize Text Analytics and OpenAI for embeddings
        self._init_text_analytics()
        self._init_openai()
        
        logger.info(f"Initialized ContentSearchService for {config.service_name}")

    def _init_text_analytics(self):
        """Initialize Text Analytics client"""
        try:
            text_analytics_endpoint = os.getenv("TEXT_ANALYTICS_ENDPOINT")
            text_analytics_key = os.getenv("TEXT_ANALYTICS_KEY")
            
            if text_analytics_endpoint and text_analytics_key:
                self.text_analytics_client = TextAnalyticsClient(
                    endpoint=text_analytics_endpoint,
                    credential=AzureKeyCredential(text_analytics_key)
                )
                logger.info("Text Analytics client initialized")
            else:
                self.text_analytics_client = None
                logger.warning("Text Analytics credentials not found")
        except Exception as e:
            logger.error(f"Failed to initialize Text Analytics: {e}")
            self.text_analytics_client = None

    def _init_openai(self):
        """Initialize OpenAI client for embeddings"""
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                openai.api_key = openai_key
                self.openai_enabled = True
                logger.info("OpenAI client initialized")
            else:
                self.openai_enabled = False
                logger.warning("OpenAI API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.openai_enabled = False

    def create_index(self) -> bool:
        """Create the search index with vector search capabilities"""
        try:
            # Define search fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="title", type=SearchFieldDataType.String, 
                              analyzer_name="en.microsoft"),
                SearchableField(name="description", type=SearchFieldDataType.String,
                              analyzer_name="en.microsoft"),
                SearchableField(name="content", type=SearchFieldDataType.String,
                              analyzer_name="en.microsoft"),
                SimpleField(name="category", type=SearchFieldDataType.String, 
                           facetable=True, filterable=True),
                SimpleField(name="genre", type=SearchFieldDataType.String,
                           facetable=True, filterable=True),
                SimpleField(name="year", type=SearchFieldDataType.Int32,
                           facetable=True, filterable=True),
                SimpleField(name="duration", type=SearchFieldDataType.Int32,
                           filterable=True),
                SimpleField(name="rating", type=SearchFieldDataType.Double,
                           filterable=True, sortable=True),
                SimpleField(name="popularity", type=SearchFieldDataType.Double,
                           filterable=True, sortable=True),
                SimpleField(name="language", type=SearchFieldDataType.String,
                           facetable=True, filterable=True),
                SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset,
                           filterable=True, sortable=True),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.config.vector_dimension,
                    vector_search_profile_name="content-vector-profile"
                ),
                ComplexField(name="metadata", fields=[
                    SimpleField(name="director", type=SearchFieldDataType.String),
                    SimpleField(name="cast", type=SearchFieldDataType.String),
                    SimpleField(name="keywords", type=SearchFieldDataType.String),
                ])
            ]

            # Vector search configuration
            vector_search = None
            if self.config.enable_vector_search:
                vector_search = VectorSearch(
                    profiles=[
                        VectorSearchProfile(
                            name="content-vector-profile",
                            algorithm_configuration_name="content-algorithm-config"
                        )
                    ],
                    algorithms=[
                        HnswAlgorithmConfiguration(
                            name="content-algorithm-config",
                            kind=VectorSearchAlgorithmKind.HNSW,
                            parameters={
                                "m": 4,
                                "efConstruction": 400,
                                "efSearch": 500,
                                "metric": "cosine"
                            }
                        )
                    ]
                )

            # Create index
            index = SearchIndex(
                name=self.config.index_name,
                fields=fields,
                vector_search=vector_search,
                cors_options=CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
            )

            # Create or update index
            result = self.index_client.create_or_update_index(index)
            logger.info(f"Created/updated search index: {result.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create search index: {e}")
            return False

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate text embedding using OpenAI"""
        if not self.openai_enabled:
            return None
            
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def index_content(self, content_items: List[Dict[str, Any]]) -> bool:
        """Index content items with vector embeddings"""
        try:
            documents = []
            
            for item in content_items:
                # Generate content text for embedding
                content_text = f"{item.get('title', '')} {item.get('description', '')}"
                
                # Generate embedding
                embedding = None
                if self.config.enable_vector_search:
                    embedding = self.generate_embedding(content_text)
                
                # Prepare document
                doc = {
                    "id": str(item["id"]),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", ""),
                    "category": item.get("category", ""),
                    "genre": item.get("genre", ""),
                    "year": item.get("year"),
                    "duration": item.get("duration"),
                    "rating": item.get("rating"),
                    "popularity": item.get("popularity", 0.0),
                    "language": item.get("language", "en"),
                    "created_date": item.get("created_date", datetime.now().isoformat()),
                    "metadata": item.get("metadata", {})
                }
                
                if embedding:
                    doc["content_vector"] = embedding
                
                documents.append(doc)
            
            # Upload documents
            result = self.search_client.upload_documents(documents)
            
            success_count = sum(1 for r in result if r.succeeded)
            logger.info(f"Indexed {success_count}/{len(documents)} content items")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to index content: {e}")
            return False

    def search_content(
        self,
        query: str,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        facets: Optional[List[str]] = None,
        top: int = 20,
        skip: int = 0,
        enable_semantic: bool = True,
        enable_vector: bool = True
    ) -> Dict[str, Any]:
        """
        Search content with hybrid search (text + vector + semantic)
        """
        try:
            search_parameters = {
                "search_text": query,
                "top": top,
                "skip": skip,
                "include_total_count": True,
                "query_type": QueryType.FULL if not enable_semantic else QueryType.SEMANTIC,
                "semantic_configuration_name": "default" if enable_semantic else None,
                "query_caption": QueryCaptionType.EXTRACTIVE if enable_semantic else None,
                "query_answer": QueryAnswerType.EXTRACTIVE if enable_semantic else None,
            }

            # Add filters
            if filters:
                filter_expressions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        values = "','".join(str(v) for v in value)
                        filter_expressions.append(f"{key} in ('{values}')")
                    else:
                        filter_expressions.append(f"{key} eq '{value}'")
                
                if filter_expressions:
                    search_parameters["filter"] = " and ".join(filter_expressions)

            # Add facets
            if facets:
                search_parameters["facets"] = facets

            # Add vector search
            vector_queries = []
            if enable_vector and self.config.enable_vector_search:
                query_embedding = self.generate_embedding(query)
                if query_embedding:
                    vector_queries.append(
                        VectorizedQuery(
                            vector=query_embedding,
                            k_nearest_neighbors=50,
                            fields="content_vector"
                        )
                    )
                    search_parameters["vector_queries"] = vector_queries

            # Perform search
            results = self.search_client.search(**search_parameters)
            
            # Process results
            search_results = []
            for result in results:
                item = {
                    "id": result["id"],
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                    "category": result.get("category", ""),
                    "genre": result.get("genre", ""),
                    "year": result.get("year"),
                    "rating": result.get("rating"),
                    "score": result.get("@search.score", 0.0),
                    "highlights": result.get("@search.highlights", {}),
                    "captions": result.get("@search.captions", []),
                    "metadata": result.get("metadata", {})
                }
                search_results.append(item)

            return {
                "results": search_results,
                "total_count": getattr(results, 'get_count', lambda: 0)(),
                "facets": getattr(results, 'get_facets', lambda: {})(),
                "answers": getattr(results, 'get_answers', lambda: [])(),
                "query": query,
                "top": top,
                "skip": skip
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "results": [],
                "total_count": 0,
                "facets": {},
                "answers": [],
                "query": query,
                "error": str(e)
            }

    def get_similar_content(
        self,
        item_id: str,
        top: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar content using vector similarity"""
        try:
            # Get the source item's vector
            source_doc = self.search_client.get_document(key=item_id)
            if not source_doc or "content_vector" not in source_doc:
                return []

            source_vector = source_doc["content_vector"]
            
            # Vector similarity search
            vector_query = VectorizedQuery(
                vector=source_vector,
                k_nearest_neighbors=top + 1,  # +1 to exclude the source item
                fields="content_vector"
            )

            results = self.search_client.search(
                search_text="*",
                vector_queries=[vector_query],
                top=top + 1,
                filter=f"id ne '{item_id}'"  # Exclude source item
            )

            similar_items = []
            for result in results:
                if result["id"] != item_id and result.get("@search.score", 0) >= similarity_threshold:
                    similar_items.append({
                        "id": result["id"],
                        "title": result.get("title", ""),
                        "category": result.get("category", ""),
                        "similarity_score": result.get("@search.score", 0.0),
                        "metadata": result.get("metadata", {})
                    })

            return similar_items[:top]

        except Exception as e:
            logger.error(f"Similar content search failed: {e}")
            return []

    def get_facets(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get faceted navigation data"""
        try:
            facet_fields = ["category", "genre", "year", "language", "rating"]
            
            search_parameters = {
                "search_text": "*",
                "facets": facet_fields,
                "top": 0  # We only want facets, not results
            }

            if filters:
                filter_expressions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        values = "','".join(str(v) for v in value)
                        filter_expressions.append(f"{key} in ('{values}')")
                    else:
                        filter_expressions.append(f"{key} eq '{value}'")
                
                if filter_expressions:
                    search_parameters["filter"] = " and ".join(filter_expressions)

            results = self.search_client.search(**search_parameters)
            return results.get_facets()

        except Exception as e:
            logger.error(f"Facets retrieval failed: {e}")
            return {}

    def suggest_content(
        self,
        query: str,
        suggester_name: str = "content-suggester",
        top: int = 5
    ) -> List[Dict[str, Any]]:
        """Get search suggestions"""
        try:
            results = self.search_client.suggest(
                search_text=query,
                suggester_name=suggester_name,
                top=top
            )

            suggestions = []
            for result in results:
                suggestions.append({
                    "text": result["text"],
                    "id": result["id"],
                    "title": result.get("title", ""),
                    "category": result.get("category", "")
                })

            return suggestions

        except Exception as e:
            logger.error(f"Suggestions failed: {e}")
            return []

    def autocomplete(
        self,
        query: str,
        suggester_name: str = "content-suggester",
        mode: str = "oneTerm",
        top: int = 5
    ) -> List[str]:
        """Get autocomplete suggestions"""
        try:
            results = self.search_client.autocomplete(
                search_text=query,
                suggester_name=suggester_name,
                autocomplete_mode=mode,
                top=top
            )

            return [result["text"] for result in results]

        except Exception as e:
            logger.error(f"Autocomplete failed: {e}")
            return []

    def get_trending_searches(
        self,
        time_window_hours: int = 24,
        top: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending search queries (requires custom analytics)"""
        # This would typically integrate with Application Insights or custom analytics
        # For now, return mock data
        trending_queries = [
            {"query": "action movies", "count": 1250, "growth": 15.2},
            {"query": "comedy series", "count": 987, "growth": 8.7},
            {"query": "sci-fi", "count": 756, "growth": 22.1},
            {"query": "documentaries", "count": 654, "growth": 5.3},
            {"query": "thriller", "count": 543, "growth": 12.8}
        ]
        
        return trending_queries[:top]

    def delete_content(self, content_ids: List[str]) -> bool:
        """Delete content from search index"""
        try:
            documents = [{"id": content_id} for content_id in content_ids]
            result = self.search_client.delete_documents(documents)
            
            success_count = sum(1 for r in result if r.succeeded)
            logger.info(f"Deleted {success_count}/{len(content_ids)} content items")
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to delete content: {e}")
            return False

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get search index statistics"""
        try:
            stats = self.index_client.get_search_index_statistics(self.config.index_name)
            return {
                "document_count": stats.document_count,
                "storage_size": stats.storage_size,
                "index_name": self.config.index_name,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get index statistics: {e}")
            return {}

class SearchServiceFactory:
    """Factory for creating search service instances"""
    
    @staticmethod
    def create_from_config(config_path: str = None) -> ContentSearchService:
        """Create search service from configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            # Use environment variables
            config_data = {
                "service_name": os.getenv("AZURE_SEARCH_SERVICE_NAME"),
                "api_key": os.getenv("AZURE_SEARCH_API_KEY"),
                "index_name": os.getenv("AZURE_SEARCH_INDEX_NAME", "content-index"),
                "vector_dimension": int(os.getenv("VECTOR_DIMENSION", "1536")),
                "enable_semantic_search": os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() == "true",
                "enable_vector_search": os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"
            }

        config = SearchConfiguration(**config_data)
        return ContentSearchService(config)

# Example usage and testing
if __name__ == "__main__":
    # Initialize search service
    search_service = SearchServiceFactory.create_from_config()
    
    # Create index
    search_service.create_index()
    
    # Example content for indexing
    sample_content = [
        {
            "id": 1,
            "title": "The Matrix",
            "description": "A computer hacker learns about the true nature of his reality",
            "category": "movie",
            "genre": "sci-fi",
            "year": 1999,
            "duration": 136,
            "rating": 8.7,
            "language": "en",
            "metadata": {"director": "Wachowski Sisters", "cast": "Keanu Reeves"}
        }
    ]
    
    # Index content
    search_service.index_content(sample_content)
    
    # Search content
    results = search_service.search_content("sci-fi movies", top=10)
    print(f"Found {results['total_count']} results")