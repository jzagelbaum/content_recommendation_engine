"""
Content Discovery and Faceted Navigation
========================================

Advanced content discovery features including faceted navigation,
personalized discovery, and content exploration capabilities.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from azure.storage.blob import BlobServiceClient
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiscoveryConfiguration:
    """Configuration for content discovery service"""
    enable_personalization: bool = True
    enable_trending: bool = True
    enable_seasonal: bool = True
    cache_ttl_minutes: int = 30
    min_content_score: float = 0.1
    max_recommendations: int = 50

@dataclass
class ContentItem:
    """Content item data structure"""
    item_id: int
    title: str
    description: str
    category: str
    genre: str
    year: int
    duration: Optional[int] = None
    rating: Optional[float] = None
    popularity: float = 0.0
    language: str = "en"
    keywords: List[str] = field(default_factory=list)
    cast: List[str] = field(default_factory=list)
    director: Optional[str] = None
    created_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FacetValue:
    """Facet value with count and metadata"""
    value: str
    count: int
    percentage: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Facet:
    """Facet definition with values"""
    name: str
    display_name: str
    values: List[FacetValue]
    facet_type: str = "terms"  # terms, range, date_range
    is_filterable: bool = True
    is_sortable: bool = True

class ContentDiscoveryService:
    """
    Advanced content discovery service with faceted navigation and personalization
    """
    
    def __init__(self, config: DiscoveryConfiguration):
        """Initialize the discovery service"""
        self.config = config
        self.content_items: Dict[int, ContentItem] = {}
        self.facets: Dict[str, Facet] = {}
        
        # Initialize cache
        self._init_cache()
        
        # Initialize content analysis components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        logger.info("ContentDiscoveryService initialized")

    def _init_cache(self):
        """Initialize Redis cache if available"""
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD")
            
            self.cache = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True
            )
            
            # Test connection
            self.cache.ping()
            self.cache_enabled = True
            logger.info("Redis cache initialized")
            
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.cache_enabled = False
            self.cache = None

    def load_content(self, content_data: List[Dict[str, Any]]):
        """Load content items for discovery"""
        try:
            self.content_items = {}
            
            for item_data in content_data:
                content_item = ContentItem(
                    item_id=item_data["id"],
                    title=item_data.get("title", ""),
                    description=item_data.get("description", ""),
                    category=item_data.get("category", ""),
                    genre=item_data.get("genre", ""),
                    year=item_data.get("year", 0),
                    duration=item_data.get("duration"),
                    rating=item_data.get("rating"),
                    popularity=item_data.get("popularity", 0.0),
                    language=item_data.get("language", "en"),
                    keywords=item_data.get("keywords", []),
                    cast=item_data.get("cast", []),
                    director=item_data.get("director"),
                    created_date=item_data.get("created_date"),
                    metadata=item_data.get("metadata", {})
                )
                
                self.content_items[content_item.item_id] = content_item
            
            # Build facets
            self._build_facets()
            
            # Build content similarity matrix
            self._build_similarity_matrix()
            
            logger.info(f"Loaded {len(self.content_items)} content items")
            
        except Exception as e:
            logger.error(f"Failed to load content: {e}")

    def _build_facets(self):
        """Build faceted navigation structure"""
        try:
            facet_data = defaultdict(Counter)
            total_items = len(self.content_items)
            
            # Count facet values
            for item in self.content_items.values():
                facet_data["category"][item.category] += 1
                facet_data["genre"][item.genre] += 1
                facet_data["language"][item.language] += 1
                
                # Year ranges
                decade = (item.year // 10) * 10
                facet_data["decade"][f"{decade}s"] += 1
                
                # Rating ranges
                if item.rating:
                    rating_range = f"{int(item.rating)}-{int(item.rating)+1}"
                    facet_data["rating_range"][rating_range] += 1
                
                # Duration ranges
                if item.duration:
                    if item.duration < 60:
                        duration_range = "Under 1 hour"
                    elif item.duration < 120:
                        duration_range = "1-2 hours"
                    elif item.duration < 180:
                        duration_range = "2-3 hours"
                    else:
                        duration_range = "Over 3 hours"
                    facet_data["duration_range"][duration_range] += 1
            
            # Build facet objects
            self.facets = {}
            
            facet_configs = {
                "category": {"display_name": "Category", "type": "terms"},
                "genre": {"display_name": "Genre", "type": "terms"},
                "language": {"display_name": "Language", "type": "terms"},
                "decade": {"display_name": "Decade", "type": "terms"},
                "rating_range": {"display_name": "Rating", "type": "range"},
                "duration_range": {"display_name": "Duration", "type": "range"}
            }
            
            for facet_name, config in facet_configs.items():
                if facet_name in facet_data:
                    values = []
                    for value, count in facet_data[facet_name].most_common():
                        percentage = (count / total_items) * 100
                        facet_value = FacetValue(
                            value=value,
                            count=count,
                            percentage=percentage
                        )
                        values.append(facet_value)
                    
                    self.facets[facet_name] = Facet(
                        name=facet_name,
                        display_name=config["display_name"],
                        values=values,
                        facet_type=config["type"]
                    )
            
            logger.info(f"Built {len(self.facets)} facets")
            
        except Exception as e:
            logger.error(f"Failed to build facets: {e}")

    def _build_similarity_matrix(self):
        """Build content similarity matrix for discovery"""
        try:
            if not self.content_items:
                return
            
            # Prepare text data for similarity calculation
            content_texts = []
            item_ids = []
            
            for item in self.content_items.values():
                text = f"{item.title} {item.description} {item.genre} {' '.join(item.keywords)}"
                content_texts.append(text)
                item_ids.append(item.item_id)
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_texts)
            
            # Calculate similarity matrix
            self.similarity_matrix = cosine_similarity(tfidf_matrix)
            self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
            
            logger.info("Built content similarity matrix")
            
        except Exception as e:
            logger.error(f"Failed to build similarity matrix: {e}")
            self.similarity_matrix = None

    def discover_content(
        self,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "popularity",
        sort_order: str = "desc",
        limit: int = 20,
        offset: int = 0,
        discovery_mode: str = "mixed"  # trending, popular, recent, similar, mixed
    ) -> Dict[str, Any]:
        """
        Discover content with various discovery modes and filtering
        """
        try:
            # Apply filters
            filtered_items = self._apply_filters(filters)
            
            # Apply discovery mode
            if discovery_mode == "trending":
                scored_items = self._get_trending_content(filtered_items)
            elif discovery_mode == "popular":
                scored_items = self._get_popular_content(filtered_items)
            elif discovery_mode == "recent":
                scored_items = self._get_recent_content(filtered_items)
            elif discovery_mode == "similar" and user_id:
                scored_items = self._get_similar_content(filtered_items, user_id)
            else:  # mixed
                scored_items = self._get_mixed_discovery(filtered_items, user_id)
            
            # Sort results
            sorted_items = self._sort_content(scored_items, sort_by, sort_order)
            
            # Paginate
            paginated_items = sorted_items[offset:offset + limit]
            
            # Prepare response
            discovery_results = []
            for item, score in paginated_items:
                discovery_results.append({
                    "item_id": item.item_id,
                    "title": item.title,
                    "description": item.description,
                    "category": item.category,
                    "genre": item.genre,
                    "year": item.year,
                    "rating": item.rating,
                    "popularity": item.popularity,
                    "discovery_score": score,
                    "metadata": item.metadata
                })
            
            return {
                "results": discovery_results,
                "total_count": len(sorted_items),
                "facets": self._get_filtered_facets(filters),
                "discovery_mode": discovery_mode,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "offset": offset,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"Content discovery failed: {e}")
            return {
                "results": [],
                "total_count": 0,
                "facets": {},
                "error": str(e)
            }

    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> List[ContentItem]:
        """Apply filters to content items"""
        if not filters:
            return list(self.content_items.values())
        
        filtered_items = []
        
        for item in self.content_items.values():
            include_item = True
            
            for filter_key, filter_value in filters.items():
                if filter_key == "category" and item.category != filter_value:
                    include_item = False
                    break
                elif filter_key == "genre" and item.genre != filter_value:
                    include_item = False
                    break
                elif filter_key == "language" and item.language != filter_value:
                    include_item = False
                    break
                elif filter_key == "year_min" and item.year < filter_value:
                    include_item = False
                    break
                elif filter_key == "year_max" and item.year > filter_value:
                    include_item = False
                    break
                elif filter_key == "rating_min" and (not item.rating or item.rating < filter_value):
                    include_item = False
                    break
                elif filter_key == "duration_max" and (item.duration and item.duration > filter_value):
                    include_item = False
                    break
            
            if include_item:
                filtered_items.append(item)
        
        return filtered_items

    def _get_trending_content(self, items: List[ContentItem]) -> List[Tuple[ContentItem, float]]:
        """Get trending content based on recent popularity"""
        # Mock trending calculation - in real implementation, use analytics data
        current_time = datetime.now()
        scored_items = []
        
        for item in items:
            # Calculate trending score based on popularity and recency
            recency_boost = 1.0
            if item.created_date:
                days_old = (current_time - item.created_date).days
                recency_boost = max(0.1, 1.0 - (days_old / 365))  # Decay over time
            
            trending_score = item.popularity * recency_boost
            scored_items.append((item, trending_score))
        
        return scored_items

    def _get_popular_content(self, items: List[ContentItem]) -> List[Tuple[ContentItem, float]]:
        """Get popular content based on overall popularity"""
        return [(item, item.popularity) for item in items]

    def _get_recent_content(self, items: List[ContentItem]) -> List[Tuple[ContentItem, float]]:
        """Get recently added content"""
        current_time = datetime.now()
        scored_items = []
        
        for item in items:
            if item.created_date:
                days_old = (current_time - item.created_date).days
                recency_score = max(0.0, 1.0 - (days_old / 30))  # 30-day window
            else:
                recency_score = 0.0
            
            scored_items.append((item, recency_score))
        
        return scored_items

    def _get_similar_content(self, items: List[ContentItem], user_id: int) -> List[Tuple[ContentItem, float]]:
        """Get content similar to user's preferences"""
        # This would typically use user preference data
        # For now, return based on popularity
        return [(item, item.popularity) for item in items]

    def _get_mixed_discovery(self, items: List[ContentItem], user_id: Optional[int]) -> List[Tuple[ContentItem, float]]:
        """Get mixed discovery results combining multiple signals"""
        # Combine trending, popular, and recent signals
        trending_items = dict(self._get_trending_content(items))
        popular_items = dict(self._get_popular_content(items))
        recent_items = dict(self._get_recent_content(items))
        
        scored_items = []
        
        for item in items:
            # Weighted combination of different signals
            mixed_score = (
                0.4 * trending_items.get(item, 0.0) +
                0.3 * popular_items.get(item, 0.0) +
                0.3 * recent_items.get(item, 0.0)
            )
            
            scored_items.append((item, mixed_score))
        
        return scored_items

    def _sort_content(
        self,
        scored_items: List[Tuple[ContentItem, float]],
        sort_by: str,
        sort_order: str
    ) -> List[Tuple[ContentItem, float]]:
        """Sort content items by specified criteria"""
        try:
            reverse = sort_order.lower() == "desc"
            
            if sort_by == "discovery_score":
                return sorted(scored_items, key=lambda x: x[1], reverse=reverse)
            elif sort_by == "popularity":
                return sorted(scored_items, key=lambda x: x[0].popularity, reverse=reverse)
            elif sort_by == "rating":
                return sorted(scored_items, key=lambda x: x[0].rating or 0, reverse=reverse)
            elif sort_by == "year":
                return sorted(scored_items, key=lambda x: x[0].year, reverse=reverse)
            elif sort_by == "title":
                return sorted(scored_items, key=lambda x: x[0].title, reverse=reverse)
            else:
                # Default to discovery score
                return sorted(scored_items, key=lambda x: x[1], reverse=reverse)
                
        except Exception as e:
            logger.error(f"Sorting failed: {e}")
            return scored_items

    def _get_filtered_facets(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get facets adjusted for current filters"""
        if not filters:
            return {name: {
                "display_name": facet.display_name,
                "type": facet.facet_type,
                "values": [{"value": v.value, "count": v.count, "percentage": v.percentage} 
                          for v in facet.values]
            } for name, facet in self.facets.items()}
        
        # For filtered results, we'd need to recalculate facet counts
        # This is a simplified version
        return {name: {
            "display_name": facet.display_name,
            "type": facet.facet_type,
            "values": [{"value": v.value, "count": v.count, "percentage": v.percentage} 
                      for v in facet.values]
        } for name, facet in self.facets.items()}

    def get_content_recommendations_by_similarity(
        self,
        item_id: int,
        limit: int = 10,
        similarity_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Get content recommendations based on similarity"""
        try:
            if not self.similarity_matrix or item_id not in self.item_id_to_index:
                return []
            
            item_index = self.item_id_to_index[item_id]
            similarities = self.similarity_matrix[item_index]
            
            # Get similar items
            similar_indices = np.argsort(similarities)[::-1][1:limit+1]  # Exclude self
            
            recommendations = []
            for idx in similar_indices:
                similarity_score = similarities[idx]
                if similarity_score >= similarity_threshold:
                    similar_item_id = list(self.item_id_to_index.keys())[idx]
                    similar_item = self.content_items[similar_item_id]
                    
                    recommendations.append({
                        "item_id": similar_item.item_id,
                        "title": similar_item.title,
                        "category": similar_item.category,
                        "genre": similar_item.genre,
                        "similarity_score": float(similarity_score),
                        "metadata": similar_item.metadata
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Similarity recommendations failed: {e}")
            return []

    def get_seasonal_content(
        self,
        season: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get seasonal content recommendations"""
        try:
            current_date = datetime.now()
            
            if not season:
                # Determine current season
                month = current_date.month
                if month in [12, 1, 2]:
                    season = "winter"
                elif month in [3, 4, 5]:
                    season = "spring"
                elif month in [6, 7, 8]:
                    season = "summer"
                else:
                    season = "fall"
            
            # Define seasonal keywords and themes
            seasonal_themes = {
                "winter": ["christmas", "holiday", "snow", "winter", "cozy"],
                "spring": ["romance", "renewal", "fresh", "growth"],
                "summer": ["adventure", "action", "beach", "vacation", "outdoors"],
                "fall": ["horror", "thriller", "dark", "mystery", "autumn"]
            }
            
            keywords = seasonal_themes.get(season, [])
            seasonal_content = []
            
            for item in self.content_items.values():
                # Check if content matches seasonal themes
                content_text = f"{item.title} {item.description} {' '.join(item.keywords)}".lower()
                
                seasonal_score = 0.0
                for keyword in keywords:
                    if keyword in content_text:
                        seasonal_score += 1.0
                
                if seasonal_score > 0:
                    seasonal_content.append({
                        "item_id": item.item_id,
                        "title": item.title,
                        "category": item.category,
                        "genre": item.genre,
                        "seasonal_score": seasonal_score,
                        "season": season,
                        "metadata": item.metadata
                    })
            
            # Sort by seasonal score and limit
            seasonal_content.sort(key=lambda x: x["seasonal_score"], reverse=True)
            return seasonal_content[:limit]
            
        except Exception as e:
            logger.error(f"Seasonal content failed: {e}")
            return []

    def get_content_collections(self) -> List[Dict[str, Any]]:
        """Get curated content collections"""
        try:
            collections = []
            
            # Genre-based collections
            genre_collections = defaultdict(list)
            for item in self.content_items.values():
                genre_collections[item.genre].append(item)
            
            for genre, items in genre_collections.items():
                if len(items) >= 5:  # Minimum items for a collection
                    collections.append({
                        "id": f"genre-{genre.lower().replace(' ', '-')}",
                        "title": f"Best of {genre}",
                        "description": f"Top {genre} content",
                        "type": "genre",
                        "item_count": len(items),
                        "items": [item.item_id for item in sorted(items, key=lambda x: x.popularity, reverse=True)[:10]]
                    })
            
            # Decade-based collections
            decade_collections = defaultdict(list)
            for item in self.content_items.values():
                decade = (item.year // 10) * 10
                decade_collections[decade].append(item)
            
            for decade, items in decade_collections.items():
                if len(items) >= 5:
                    collections.append({
                        "id": f"decade-{decade}s",
                        "title": f"{decade}s Classics",
                        "description": f"Best content from the {decade}s",
                        "type": "decade",
                        "item_count": len(items),
                        "items": [item.item_id for item in sorted(items, key=lambda x: x.popularity, reverse=True)[:10]]
                    })
            
            return collections
            
        except Exception as e:
            logger.error(f"Content collections failed: {e}")
            return []

    def export_facets_config(self, output_path: str) -> bool:
        """Export facets configuration for use in search service"""
        try:
            facets_config = {}
            
            for name, facet in self.facets.items():
                facets_config[name] = {
                    "display_name": facet.display_name,
                    "type": facet.facet_type,
                    "is_filterable": facet.is_filterable,
                    "is_sortable": facet.is_sortable,
                    "values": [
                        {
                            "value": v.value,
                            "count": v.count,
                            "percentage": v.percentage
                        }
                        for v in facet.values
                    ]
                }
            
            with open(output_path, 'w') as f:
                json.dump(facets_config, f, indent=2)
            
            logger.info(f"Exported facets configuration to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export facets config: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize discovery service
    config = DiscoveryConfiguration()
    discovery_service = ContentDiscoveryService(config)
    
    # Load sample content
    sample_content = [
        {
            "id": 1,
            "title": "The Matrix",
            "description": "A computer hacker learns about reality",
            "category": "movie",
            "genre": "sci-fi",
            "year": 1999,
            "rating": 8.7,
            "popularity": 0.9,
            "keywords": ["cyberpunk", "philosophy", "action"]
        }
    ]
    
    discovery_service.load_content(sample_content)
    
    # Discover content
    results = discovery_service.discover_content(
        discovery_mode="mixed",
        limit=10
    )
    
    print(f"Discovered {len(results['results'])} content items")