"""
OpenAI Data Generator
Generates synthetic data for testing and development using Azure OpenAI
"""

import asyncio
import json
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .openai_service import AzureOpenAIService
import logging

logger = logging.getLogger(__name__)

class OpenAIDataGenerator:
    """Generate synthetic data using OpenAI for testing and development"""
    
    def __init__(self):
        """Initialize the data generator"""
        self.openai_service = AzureOpenAIService()
        self.content_categories = [
            "Action", "Comedy", "Drama", "Horror", "Sci-Fi", 
            "Romance", "Documentary", "Thriller", "Animation", "Fantasy"
        ]
    
    async def generate_users(
        self, 
        num_users: int = 100,
        demographic_distribution: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate synthetic user profiles"""
        try:
            if demographic_distribution is None:
                demographic_distribution = {
                    "age_ranges": ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
                    "regions": ["North America", "Europe", "Asia", "South America", "Australia"],
                    "subscription_types": ["basic", "premium", "family"]
                }
            
            # Generate users in batches to avoid token limits
            batch_size = 20
            all_users = []
            
            for i in range(0, num_users, batch_size):
                batch_num_users = min(batch_size, num_users - i)
                
                prompt = f"""
                Generate {batch_num_users} diverse synthetic user profiles for a streaming content recommendation system.
                
                User demographic distribution:
                - Age ranges: {', '.join(demographic_distribution['age_ranges'])}
                - Regions: {', '.join(demographic_distribution['regions'])}
                - Subscription types: {', '.join(demographic_distribution['subscription_types'])}
                - Content categories: {', '.join(self.content_categories)}
                
                For each user, provide a JSON object with:
                - user_id: unique identifier (string, format: "user_XXXXX")
                - age: specific age in years (integer)
                - region: user's region (string)
                - preferences: array of 2-4 preferred content categories
                - viewing_history: array of 5-8 realistic content titles they've watched
                - viewing_patterns: object with avg_session_duration_minutes (30-180), preferred_time_of_day ("morning", "afternoon", "evening", "night"), and weekly_hours (5-40)
                - subscription_type: subscription level
                - personality_traits: array of 2-3 traits that affect viewing preferences (e.g., "adventurous", "family-oriented", "analytical")
                - device_preferences: array of preferred devices ("mobile", "tv", "tablet", "laptop")
                
                Return as a valid JSON array of user objects. Ensure realistic diversity in preferences and demographics.
                """
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are a data generator that creates realistic synthetic user profiles for testing recommendation systems. Always return valid JSON arrays."
                    },
                    {"role": "user", "content": prompt}
                ]
                
                response_text = await self.openai_service.generate_completion(
                    messages, 
                    max_tokens=2000, 
                    temperature=0.8
                )
                
                try:
                    batch_users = json.loads(response_text)
                    if isinstance(batch_users, list):
                        all_users.extend(batch_users)
                    else:
                        logger.warning(f"Invalid JSON structure in batch {i}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON in batch {i}: {e}")
                
                # Small delay between batches to respect rate limits
                if i + batch_size < num_users:
                    await asyncio.sleep(1)
            
            # Fill in any missing users with template data if needed
            while len(all_users) < num_users:
                template_user = self._create_template_user(len(all_users))
                all_users.append(template_user)
            
            return all_users[:num_users]
        except Exception as e:
            logger.error(f"Error generating users: {e}")
            return []
    
    async def generate_content_items(
        self, 
        num_items: int = 500,
        content_distribution: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate synthetic content items"""
        try:
            if content_distribution is None:
                content_distribution = {
                    "types": ["movie", "tv_series", "documentary", "short"],
                    "release_years": list(range(1990, 2025)),
                    "ratings": ["G", "PG", "PG-13", "R", "TV-Y", "TV-PG", "TV-14", "TV-MA"]
                }
            
            batch_size = 15  # Smaller batches for content generation
            all_content = []
            
            for i in range(0, num_items, batch_size):
                batch_num_items = min(batch_size, num_items - i)
                
                prompt = f"""
                Generate {batch_num_items} diverse synthetic content items for a streaming platform.
                
                Content distribution:
                - Types: {', '.join(content_distribution['types'])}
                - Genres: {', '.join(self.content_categories)}
                - Content ratings: {', '.join(content_distribution['ratings'])}
                - Release years: {min(content_distribution['release_years'])}-{max(content_distribution['release_years'])}
                
                For each content item, provide a JSON object with:
                - id: unique identifier (string, format: "content_XXXXX")
                - title: realistic and engaging title (string)
                - description: compelling 2-3 sentence description (string, 100-200 chars)
                - type: content type from the list above
                - genre: array of 1-3 relevant genres
                - release_year: realistic release year (integer)
                - duration_minutes: appropriate duration (movies: 80-180, series episodes: 20-60, documentaries: 45-120)
                - rating: content rating from the list above
                - cast: array of 3-5 realistic actor names (can be fictional)
                - director: realistic director name (can be fictional)
                - language: primary language ("English", "Spanish", "French", etc.)
                - country: country of origin
                - imdb_rating: realistic rating 4.0-9.5 (float)
                - popularity_score: popularity metric 0.1-1.0 (float)
                
                Ensure titles and descriptions are creative and realistic for their genres.
                Return as a valid JSON array of content objects.
                """
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are a content generator that creates realistic synthetic content metadata for streaming platforms. Always return valid JSON arrays with creative, engaging content."
                    },
                    {"role": "user", "content": prompt}
                ]
                
                response_text = await self.openai_service.generate_completion(
                    messages, 
                    max_tokens=2500, 
                    temperature=0.9
                )
                
                try:
                    batch_content = json.loads(response_text)
                    if isinstance(batch_content, list):
                        # Enhance with additional computed fields
                        for item in batch_content:
                            item = self._enhance_content_item(item)
                        all_content.extend(batch_content)
                    else:
                        logger.warning(f"Invalid JSON structure in content batch {i}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON in content batch {i}: {e}")
                
                # Delay between batches
                if i + batch_size < num_items:
                    await asyncio.sleep(1)
            
            # Fill in missing items if needed
            while len(all_content) < num_items:
                template_item = self._create_template_content(len(all_content))
                all_content.append(template_item)
            
            return all_content[:num_items]
        except Exception as e:
            logger.error(f"Error generating content items: {e}")
            return []
    
    async def generate_interactions(
        self,
        users: List[Dict[str, Any]],
        content_items: List[Dict[str, Any]],
        interactions_per_user: int = 50,
        time_span_days: int = 365
    ) -> List[Dict[str, Any]]:
        """Generate realistic user-content interactions"""
        try:
            all_interactions = []
            
            for user in users:
                user_preferences = user.get('preferences', [])
                user_personality = user.get('personality_traits', [])
                
                # Generate interactions for this user
                user_interactions = await self._generate_user_interactions(
                    user,
                    content_items,
                    interactions_per_user,
                    time_span_days
                )
                
                all_interactions.extend(user_interactions)
            
            return all_interactions
        except Exception as e:
            logger.error(f"Error generating interactions: {e}")
            return []
    
    async def generate_content_features_with_ai(
        self,
        content_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance content items with AI-generated features"""
        try:
            enhanced_items = []
            batch_size = 10
            
            for i in range(0, len(content_items), batch_size):
                batch = content_items[i:i + batch_size]
                
                for item in batch:
                    # Generate enhanced features using AI
                    description = item.get('description', '')
                    title = item.get('title', '')
                    genre = item.get('genre', [])
                    
                    content_text = f"Title: {title}. Description: {description}. Genres: {', '.join(genre)}"
                    
                    features = await self.openai_service.analyze_content_for_features(content_text)
                    
                    # Merge AI features with existing item
                    enhanced_item = {**item, **features}
                    enhanced_items.append(enhanced_item)
                
                # Rate limiting
                if i + batch_size < len(content_items):
                    await asyncio.sleep(0.5)
            
            return enhanced_items
        except Exception as e:
            logger.error(f"Error enhancing content with AI: {e}")
            return content_items
    
    async def _generate_user_interactions(
        self,
        user: Dict[str, Any],
        content_items: List[Dict[str, Any]],
        num_interactions: int,
        time_span_days: int
    ) -> List[Dict[str, Any]]:
        """Generate interactions for a single user"""
        interactions = []
        user_preferences = user.get('preferences', [])
        user_id = user.get('user_id')
        
        # Select content items that match user preferences
        preferred_content = []
        other_content = []
        
        for item in content_items:
            item_genres = item.get('genre', [])
            if any(genre in user_preferences for genre in item_genres):
                preferred_content.append(item)
            else:
                other_content.append(item)
        
        # Generate interactions with preference bias
        for i in range(num_interactions):
            # 70% chance to interact with preferred content
            if random.random() < 0.7 and preferred_content:
                selected_item = random.choice(preferred_content)
            elif other_content:
                selected_item = random.choice(other_content)
            else:
                continue
            
            # Generate interaction details
            interaction = self._create_interaction(
                user,
                selected_item,
                time_span_days
            )
            interactions.append(interaction)
        
        return interactions
    
    def _create_interaction(
        self,
        user: Dict[str, Any],
        content_item: Dict[str, Any],
        time_span_days: int
    ) -> Dict[str, Any]:
        """Create a single interaction record"""
        # Random timestamp within time span
        days_ago = random.randint(0, time_span_days)
        interaction_time = datetime.utcnow() - timedelta(days=days_ago)
        
        # Determine interaction type and rating based on user preferences
        user_preferences = user.get('preferences', [])
        item_genres = content_item.get('genre', [])
        
        # Higher ratings for preferred genres
        if any(genre in user_preferences for genre in item_genres):
            rating = random.uniform(3.5, 5.0)
            completion_rate = random.uniform(0.7, 1.0)
        else:
            rating = random.uniform(1.0, 4.0)
            completion_rate = random.uniform(0.2, 0.8)
        
        # Watch duration based on completion rate
        total_duration = content_item.get('duration_minutes', 90)
        watch_duration = int(total_duration * completion_rate)
        
        return {
            "interaction_id": f"int_{user['user_id']}_{content_item['id']}_{int(interaction_time.timestamp())}",
            "user_id": user['user_id'],
            "content_id": content_item['id'],
            "interaction_type": "view",
            "rating": round(rating, 1),
            "watch_duration_minutes": watch_duration,
            "completion_rate": round(completion_rate, 2),
            "timestamp": interaction_time.isoformat(),
            "device": random.choice(user.get('device_preferences', ['tv'])),
            "session_id": f"session_{random.randint(100000, 999999)}",
            "context": {
                "time_of_day": random.choice(["morning", "afternoon", "evening", "night"]),
                "day_of_week": random.choice(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]),
                "binge_session": random.choice([True, False])
            }
        }
    
    def _enhance_content_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance content item with computed fields"""
        # Add metadata
        item['created_at'] = datetime.utcnow().isoformat()
        item['updated_at'] = datetime.utcnow().isoformat()
        
        # Ensure required fields exist
        if 'popularity_score' not in item:
            item['popularity_score'] = random.uniform(0.1, 1.0)
        
        if 'imdb_rating' not in item:
            item['imdb_rating'] = random.uniform(4.0, 9.5)
        
        return item
    
    def _create_template_user(self, index: int) -> Dict[str, Any]:
        """Create a template user for fallback"""
        return {
            "user_id": f"user_{index:05d}",
            "age": random.randint(18, 65),
            "region": random.choice(["North America", "Europe", "Asia"]),
            "preferences": random.sample(self.content_categories, random.randint(2, 4)),
            "viewing_history": [f"Template Content {i}" for i in range(random.randint(3, 6))],
            "viewing_patterns": {
                "avg_session_duration_minutes": random.randint(30, 180),
                "preferred_time_of_day": random.choice(["morning", "afternoon", "evening", "night"]),
                "weekly_hours": random.randint(5, 40)
            },
            "subscription_type": random.choice(["basic", "premium", "family"]),
            "personality_traits": random.sample(["adventurous", "family-oriented", "analytical", "social"], 2),
            "device_preferences": random.sample(["mobile", "tv", "tablet", "laptop"], random.randint(1, 3))
        }
    
    def _create_template_content(self, index: int) -> Dict[str, Any]:
        """Create template content for fallback"""
        genre = random.choice(self.content_categories)
        content_type = random.choice(["movie", "tv_series", "documentary"])
        
        return {
            "id": f"content_{index:05d}",
            "title": f"Template {genre} {content_type.title()} {index}",
            "description": f"A {genre.lower()} {content_type} with engaging storyline and great characters.",
            "type": content_type,
            "genre": [genre],
            "release_year": random.randint(2000, 2024),
            "duration_minutes": random.randint(80, 180) if content_type == "movie" else random.randint(20, 60),
            "rating": random.choice(["PG", "PG-13", "R", "TV-PG", "TV-14"]),
            "cast": [f"Actor {i}" for i in range(random.randint(3, 5))],
            "director": f"Director {index}",
            "language": "English",
            "country": "USA",
            "imdb_rating": round(random.uniform(4.0, 9.0), 1),
            "popularity_score": round(random.uniform(0.1, 1.0), 2),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    async def close(self):
        """Close the data generator"""
        await self.openai_service.close()