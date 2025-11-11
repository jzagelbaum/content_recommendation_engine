"""
Embedding Service for Vector Search
Manages embeddings and vector search using Azure AI Search
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from .openai_service import AzureOpenAIService
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for managing embeddings and vector search"""
    
    def __init__(self):
        """Initialize embedding service with Azure AI Search"""
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "content-recommendations-index")
        
        if not self.search_endpoint or not self.search_key:
            raise ValueError("Azure Search endpoint and API key must be provided")
        
        self.credential = AzureKeyCredential(self.search_key)
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        self.index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=self.credential
        )
        
        self.openai_service = AzureOpenAIService()
        self.vector_dimensions = 1536  # text-embedding-ada-002 dimensions
    
    async def initialize_index(self) -> bool:
        """Initialize the search index for content embeddings"""
        try:
            # Check if index already exists
            try:
                await self.index_client.get_index(self.index_name)
                logger.info(f"Index {self.index_name} already exists")
                return True
            except Exception:
                logger.info(f"Creating new index: {self.index_name}")
            
            # Define the index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="description", type=SearchFieldDataType.String),
                SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="genre", type=SearchFieldDataType.Collection(SearchFieldDataType.String), 
                           filterable=True, facetable=True),
                SimpleField(name="release_year", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
                SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="duration_minutes", type=SearchFieldDataType.Int32, filterable=True),
                SimpleField(name="content_rating", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="cast", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
                SearchableField(name="director", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.vector_dimensions,
                    vector_search_profile_name="content-vector-profile"
                ),
                SearchableField(name="combined_features", type=SearchFieldDataType.String),
                SimpleField(name="popularity_score", type=SearchFieldDataType.Double, sortable=True),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, sortable=True),
                SimpleField(name="updated_at", type=SearchFieldDataType.DateTimeOffset, sortable=True)
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="content-vector-profile",
                        algorithm_configuration_name="content-hnsw-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="content-hnsw-config",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ]
            )
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            await self.index_client.create_index(index)
            logger.info(f"Successfully created index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating search index: {e}")
            return False
    
    async def index_content(self, content_items: List[Dict[str, Any]]) -> bool:
        """Index content items with embeddings"""
        try:
            # Prepare text for embedding generation
            texts_for_embedding = []
            documents = []
            
            for item in content_items:
                # Combine relevant text fields for embedding
                combined_text = self._build_combined_text(item)
                texts_for_embedding.append(combined_text)
                
                # Prepare document for indexing
                doc = {
                    "id": str(item.get("id", item.get("item_id", ""))),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "category": item.get("category", ""),
                    "genre": item.get("genre", []) if isinstance(item.get("genre"), list) else [item.get("genre", "")],
                    "release_year": item.get("release_year", item.get("year", 0)),
                    "rating": float(item.get("rating", 0.0)),
                    "duration_minutes": item.get("duration_minutes", item.get("duration", 0)),
                    "content_rating": item.get("content_rating", ""),
                    "cast": item.get("cast", []) if isinstance(item.get("cast"), list) else [],
                    "director": item.get("director", ""),
                    "combined_features": combined_text,
                    "popularity_score": float(item.get("popularity_score", 0.0)),
                    "created_at": item.get("created_at", "2024-01-01T00:00:00Z"),
                    "updated_at": item.get("updated_at", "2024-01-01T00:00:00Z")
                }
                documents.append(doc)
            
            # Generate embeddings in batches
            logger.info(f"Generating embeddings for {len(texts_for_embedding)} content items")
            embeddings = await self.openai_service.generate_embeddings_batch(texts_for_embedding)
            
            # Add embeddings to documents
            for i, embedding in enumerate(embeddings):
                documents[i]["content_vector"] = embedding
            
            # Upload documents to search index
            logger.info(f"Uploading {len(documents)} documents to search index")
            result = await self.search_client.upload_documents(documents)
            
            # Check for errors
            succeeded = sum(1 for r in result if r.succeeded)
            failed = len(result) - succeeded
            
            if failed > 0:
                logger.warning(f"Failed to index {failed} documents")
            
            logger.info(f"Successfully indexed {succeeded} content items")
            return succeeded > 0
        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            return False
    
    async def find_similar_content(
        self, 
        query_text: str, 
        top_k: int = 10,
        filters: Optional[str] = None,
        include_semantic: bool = True
    ) -> List[Dict[str, Any]]:
        """Find similar content using vector and semantic search"""
        try:
            # Generate embedding for query
            query_embedding = await self.openai_service.generate_embedding(query_text)
            
            # Create vectorized query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # Perform search
            search_text = query_text if include_semantic else ""
            
            results = await self.search_client.search(
                search_text=search_text,
                vector_queries=[vector_query],
                filter=filters,
                top=top_k,
                select=["id", "title", "description", "category", "genre", "rating", 
                       "release_year", "duration_minutes", "popularity_score"]
            )
            
            # Format results
            similar_items = []
            async for result in results:
                item = {
                    "id": result["id"],
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                    "category": result.get("category", ""),
                    "genre": result.get("genre", []),
                    "rating": result.get("rating", 0.0),
                    "release_year": result.get("release_year", 0),
                    "duration_minutes": result.get("duration_minutes", 0),
                    "popularity_score": result.get("popularity_score", 0.0),
                    "similarity_score": result.get("@search.score", 0.0),
                    "search_highlights": result.get("@search.highlights", {})
                }
                similar_items.append(item)
            
            return similar_items
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []
    
    async def find_content_by_user_preferences(
        self,
        user_preferences: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find content based on user preferences using vector search"""
        try:
            # Build query text from user preferences
            query_parts = []
            
            if "preferences" in user_preferences:
                query_parts.append(f"Genres: {', '.join(user_preferences['preferences'])}")
            
            if "viewing_history" in user_preferences:
                recent_content = user_preferences["viewing_history"][-5:]  # Last 5 items
                query_parts.append(f"Similar to: {', '.join(recent_content)}")
            
            if "mood" in user_preferences:
                query_parts.append(f"Mood: {user_preferences['mood']}")
            
            query_text = " ".join(query_parts)
            
            # Build filters based on preferences
            filters = []
            
            if "preferred_genres" in user_preferences:
                genre_filter = " or ".join([f"genre/any(g: g eq '{genre}')" 
                                          for genre in user_preferences["preferred_genres"]])
                filters.append(f"({genre_filter})")
            
            if "min_rating" in user_preferences:
                filters.append(f"rating ge {user_preferences['min_rating']}")
            
            if "max_duration" in user_preferences:
                filters.append(f"duration_minutes le {user_preferences['max_duration']}")
            
            filter_string = " and ".join(filters) if filters else None
            
            return await self.find_similar_content(
                query_text=query_text,
                top_k=top_k,
                filters=filter_string,
                include_semantic=True
            )
        except Exception as e:
            logger.error(f"Error finding content by user preferences: {e}")
            return []
    
    async def get_content_recommendations_hybrid(
        self,
        user_query: str,
        user_preferences: Dict[str, Any],
        top_k: int = 10,
        vector_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get hybrid recommendations combining vector and preference-based search"""
        try:
            # Get vector-based recommendations
            vector_results = await self.find_similar_content(
                query_text=user_query,
                top_k=int(top_k * 1.5),  # Get more for better hybrid results
                include_semantic=True
            )
            
            # Get preference-based recommendations
            preference_results = await self.find_content_by_user_preferences(
                user_preferences=user_preferences,
                top_k=int(top_k * 1.5)
            )
            
            # Combine and re-rank results
            combined_results = self._combine_recommendation_results(
                vector_results,
                preference_results,
                vector_weight
            )
            
            return combined_results[:top_k]
        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {e}")
            return []
    
    def _build_combined_text(self, item: Dict[str, Any]) -> str:
        """Build combined text for embedding generation"""
        text_parts = []
        
        # Title and description
        if title := item.get("title"):
            text_parts.append(f"Title: {title}")
        
        if description := item.get("description"):
            text_parts.append(f"Description: {description}")
        
        # Genre and category
        if genre := item.get("genre"):
            if isinstance(genre, list):
                text_parts.append(f"Genres: {', '.join(genre)}")
            else:
                text_parts.append(f"Genre: {genre}")
        
        if category := item.get("category"):
            text_parts.append(f"Category: {category}")
        
        # Cast and director
        if cast := item.get("cast"):
            if isinstance(cast, list):
                text_parts.append(f"Cast: {', '.join(cast[:5])}")  # Limit to first 5
            else:
                text_parts.append(f"Cast: {cast}")
        
        if director := item.get("director"):
            text_parts.append(f"Director: {director}")
        
        return " ".join(text_parts)
    
    def _combine_recommendation_results(
        self,
        vector_results: List[Dict[str, Any]],
        preference_results: List[Dict[str, Any]],
        vector_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine and re-rank results from different sources"""
        # Create a dictionary to merge results by ID
        combined = {}
        preference_weight = 1.0 - vector_weight
        
        # Add vector results
        for i, item in enumerate(vector_results):
            item_id = item["id"]
            vector_score = item.get("similarity_score", 0.0)
            position_bonus = (len(vector_results) - i) / len(vector_results) * 0.1
            
            combined[item_id] = {
                **item,
                "vector_score": vector_score * vector_weight + position_bonus,
                "preference_score": 0.0,
                "source": "vector"
            }
        
        # Add preference results
        for i, item in enumerate(preference_results):
            item_id = item["id"]
            preference_score = item.get("similarity_score", 0.0)
            position_bonus = (len(preference_results) - i) / len(preference_results) * 0.1
            
            if item_id in combined:
                combined[item_id]["preference_score"] = preference_score * preference_weight + position_bonus
                combined[item_id]["source"] = "hybrid"
            else:
                combined[item_id] = {
                    **item,
                    "vector_score": 0.0,
                    "preference_score": preference_score * preference_weight + position_bonus,
                    "source": "preference"
                }
        
        # Calculate final scores and sort
        for item in combined.values():
            item["final_score"] = item["vector_score"] + item["preference_score"]
        
        return sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)
    
    async def close(self):
        """Close the search clients"""
        if hasattr(self.search_client, 'close'):
            await self.search_client.close()
        if hasattr(self.index_client, 'close'):
            await self.index_client.close()
        
        await self.openai_service.close()