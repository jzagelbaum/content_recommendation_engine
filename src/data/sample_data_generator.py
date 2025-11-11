"""
Sample Data Generator
====================

Generate realistic sample datasets for testing and demonstrating
the content recommendation engine capabilities.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import json
import csv
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import uuid
from faker import Faker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker for generating realistic data
fake = Faker()

class ContentDataGenerator:
    """Generate sample content data"""
    
    def __init__(self, seed: int = 42):
        """Initialize the content data generator"""
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        # Content categories and genres
        self.categories = ["movie", "tv_series", "documentary", "short", "anime"]
        self.genres = [
            "action", "adventure", "animation", "biography", "comedy", "crime",
            "documentary", "drama", "family", "fantasy", "history", "horror",
            "music", "mystery", "romance", "sci-fi", "sport", "thriller", "war", "western"
        ]
        
        # Languages
        self.languages = ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "hi"]
        
        # Directors and cast pools
        self.directors = [fake.name() for _ in range(200)]
        self.actors = [fake.name() for _ in range(1000)]
        
        logger.info("ContentDataGenerator initialized")

    def generate_content_items(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Generate sample content items"""
        content_items = []
        
        for i in range(count):
            category = random.choice(self.categories)
            genre = random.choice(self.genres)
            language = random.choice(self.languages)
            
            # Generate realistic titles
            if category == "movie":
                title = fake.catch_phrase()
            elif category == "tv_series":
                title = f"{fake.company()} {random.choice(['Chronicles', 'Adventures', 'Tales', 'Stories'])}"
            elif category == "documentary":
                title = f"The {fake.word().title()} {random.choice(['Story', 'Chronicles', 'Files', 'Truth'])}"
            else:
                title = fake.sentence(nb_words=3).replace('.', '')
            
            # Duration based on category
            if category == "movie":
                duration = random.randint(80, 180)
            elif category == "tv_series":
                duration = random.randint(30, 60)  # per episode
            elif category == "documentary":
                duration = random.randint(45, 120)
            else:
                duration = random.randint(10, 45)
            
            # Generate description
            description = fake.text(max_nb_chars=200)
            
            # Release year
            year = random.randint(1980, 2024)
            
            # Rating (1-10, with bias towards higher ratings)
            rating = max(1.0, min(10.0, np.random.normal(7.0, 1.5)))
            
            # Popularity score (0-1, with power law distribution)
            popularity = min(1.0, np.random.pareto(1.5) / 10)
            
            # Cast and crew
            director = random.choice(self.directors) if random.random() > 0.1 else None
            cast = random.sample(self.actors, min(random.randint(2, 8), len(self.actors)))
            
            # Keywords
            keywords = random.sample([
                "adventure", "friendship", "love", "betrayal", "mystery", "action",
                "comedy", "drama", "suspense", "family", "supernatural", "historical",
                "futuristic", "psychological", "emotional", "intense", "heartwarming"
            ], random.randint(2, 5))
            
            # Created date (realistic distribution)
            days_ago = int(np.random.exponential(100))
            created_date = datetime.now() - timedelta(days=days_ago)
            
            content_item = {
                "id": i + 1,
                "title": title,
                "description": description,
                "category": category,
                "genre": genre,
                "year": year,
                "duration": duration,
                "rating": round(rating, 1),
                "popularity": round(popularity, 3),
                "language": language,
                "director": director,
                "cast": cast,
                "keywords": keywords,
                "created_date": created_date.isoformat(),
                "metadata": {
                    "content_id": str(uuid.uuid4()),
                    "source": "sample_generator",
                    "quality": random.choice(["HD", "4K", "SD"]),
                    "age_rating": random.choice(["G", "PG", "PG-13", "R", "NC-17"])
                }
            }
            
            content_items.append(content_item)
        
        logger.info(f"Generated {len(content_items)} content items")
        return content_items

class UserDataGenerator:
    """Generate sample user data"""
    
    def __init__(self, seed: int = 42):
        """Initialize the user data generator"""
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        # User demographics
        self.age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        self.countries = ["US", "CA", "UK", "DE", "FR", "AU", "JP", "BR", "IN", "MX"]
        self.devices = ["mobile", "desktop", "tablet", "smart_tv", "gaming_console"]
        
        logger.info("UserDataGenerator initialized")

    def generate_users(self, count: int = 5000) -> List[Dict[str, Any]]:
        """Generate sample users"""
        users = []
        
        for i in range(count):
            age_group = random.choice(self.age_groups)
            country = random.choice(self.countries)
            
            # User preferences based on demographics
            if age_group in ["18-24", "25-34"]:
                preferred_genres = random.sample(
                    ["action", "comedy", "sci-fi", "horror", "romance", "animation"], 
                    random.randint(2, 4)
                )
            elif age_group in ["35-44", "45-54"]:
                preferred_genres = random.sample(
                    ["drama", "comedy", "thriller", "documentary", "biography"], 
                    random.randint(2, 3)
                )
            else:
                preferred_genres = random.sample(
                    ["drama", "documentary", "biography", "history", "comedy"], 
                    random.randint(1, 3)
                )
            
            # Account creation date
            days_ago = int(np.random.exponential(200))
            created_date = datetime.now() - timedelta(days=days_ago)
            
            # Last activity
            last_active_days = random.randint(0, 30)
            last_active = datetime.now() - timedelta(days=last_active_days)
            
            user = {
                "user_id": i + 1,
                "username": fake.user_name(),
                "email": fake.email(),
                "age_group": age_group,
                "country": country,
                "preferred_language": random.choice(["en", "es", "fr", "de"]),
                "preferred_genres": preferred_genres,
                "preferred_devices": random.sample(self.devices, random.randint(1, 3)),
                "subscription_tier": random.choice(["free", "premium", "family"]),
                "created_date": created_date.isoformat(),
                "last_active": last_active.isoformat(),
                "is_active": last_active_days < 7,
                "preferences": {
                    "explicit_content": random.choice([True, False]),
                    "preferred_duration": random.choice(["short", "medium", "long", "any"]),
                    "autoplay": random.choice([True, False]),
                    "hd_quality": random.choice([True, False])
                }
            }
            
            users.append(user)
        
        logger.info(f"Generated {len(users)} users")
        return users

class InteractionDataGenerator:
    """Generate sample user interaction data"""
    
    def __init__(self, users: List[Dict], content_items: List[Dict], seed: int = 42):
        """Initialize the interaction data generator"""
        random.seed(seed)
        np.random.seed(seed)
        
        self.users = users
        self.content_items = content_items
        self.interaction_types = ["view", "like", "share", "rating", "bookmark", "purchase"]
        
        logger.info("InteractionDataGenerator initialized")

    def generate_interactions(self, count: int = 50000) -> List[Dict[str, Any]]:
        """Generate sample user interactions"""
        interactions = []
        
        for i in range(count):
            user = random.choice(self.users)
            content = random.choice(self.content_items)
            
            # Interaction type probability based on user behavior
            interaction_weights = {
                "view": 0.5,
                "like": 0.2,
                "rating": 0.15,
                "share": 0.08,
                "bookmark": 0.05,
                "purchase": 0.02
            }
            
            interaction_type = random.choices(
                list(interaction_weights.keys()),
                weights=list(interaction_weights.values())
            )[0]
            
            # Generate timestamp (more recent interactions more likely)
            days_ago = int(np.random.exponential(30))
            hours_offset = random.randint(0, 23)
            minutes_offset = random.randint(0, 59)
            
            timestamp = (datetime.now() - timedelta(days=days_ago, hours=hours_offset, minutes=minutes_offset))
            
            # Rating value for rating interactions
            rating = None
            if interaction_type == "rating":
                # Bias towards higher ratings with some variation
                rating = max(1.0, min(5.0, np.random.normal(4.0, 1.0)))
                rating = round(rating, 1)
            
            # Session information
            session_id = str(uuid.uuid4())
            device_type = random.choice(user["preferred_devices"])
            platform = random.choice(["web", "mobile_app", "tv_app"])
            
            # Content consumption details for view interactions
            watch_duration = None
            completion_rate = None
            if interaction_type == "view":
                # Realistic viewing patterns
                total_duration = content["duration"]
                if random.random() < 0.3:  # 30% abandon early
                    watch_duration = random.randint(1, min(10, total_duration))
                elif random.random() < 0.6:  # 30% watch partially
                    watch_duration = random.randint(10, int(total_duration * 0.8))
                else:  # 40% watch completely
                    watch_duration = total_duration + random.randint(-2, 5)
                
                completion_rate = min(1.0, watch_duration / total_duration)
            
            interaction = {
                "interaction_id": str(uuid.uuid4()),
                "user_id": user["user_id"],
                "item_id": content["id"],
                "interaction_type": interaction_type,
                "rating": rating,
                "timestamp": timestamp.isoformat(),
                "session_id": session_id,
                "device_type": device_type,
                "platform": platform,
                "watch_duration": watch_duration,
                "completion_rate": round(completion_rate, 3) if completion_rate else None,
                "metadata": {
                    "user_age_group": user["age_group"],
                    "user_country": user["country"],
                    "content_genre": content["genre"],
                    "content_category": content["category"],
                    "content_year": content["year"]
                }
            }
            
            interactions.append(interaction)
        
        logger.info(f"Generated {len(interactions)} interactions")
        return interactions

class DatasetExporter:
    """Export generated datasets to various formats"""
    
    def __init__(self, output_dir: str = "sample_data"):
        """Initialize the dataset exporter"""
        self.output_dir = output_dir
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"DatasetExporter initialized with output directory: {output_dir}")

    def export_to_json(self, data: List[Dict], filename: str) -> str:
        """Export data to JSON format"""
        filepath = f"{self.output_dir}/{filename}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Exported {len(data)} records to {filepath}")
        return filepath

    def export_to_csv(self, data: List[Dict], filename: str) -> str:
        """Export data to CSV format"""
        filepath = f"{self.output_dir}/{filename}.csv"
        
        if not data:
            logger.warning(f"No data to export for {filename}")
            return filepath
        
        # Convert to DataFrame for easier CSV export
        df = pd.DataFrame(data)
        
        # Handle nested columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains lists or dicts
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, (list, dict)):
                    df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.info(f"Exported {len(data)} records to {filepath}")
        return filepath

    def export_to_parquet(self, data: List[Dict], filename: str) -> str:
        """Export data to Parquet format"""
        filepath = f"{self.output_dir}/{filename}.parquet"
        
        if not data:
            logger.warning(f"No data to export for {filename}")
            return filepath
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Handle nested columns for Parquet
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, (list, dict)):
                    df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
        
        df.to_parquet(filepath, index=False, engine='pyarrow')
        
        logger.info(f"Exported {len(data)} records to {filepath}")
        return filepath

    def create_data_summary(self, datasets: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Create a summary of the generated datasets"""
        summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "datasets": {}
        }
        
        for dataset_name, data in datasets.items():
            if data:
                df = pd.DataFrame(data)
                
                dataset_summary = {
                    "record_count": len(data),
                    "columns": list(df.columns),
                    "column_types": {col: str(df[col].dtype) for col in df.columns},
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
                
                # Add specific insights based on dataset type
                if dataset_name == "content_items":
                    dataset_summary["insights"] = {
                        "categories": df['category'].value_counts().to_dict(),
                        "genres": df['genre'].value_counts().to_dict(),
                        "languages": df['language'].value_counts().to_dict(),
                        "year_range": f"{df['year'].min()}-{df['year'].max()}",
                        "avg_rating": round(df['rating'].mean(), 2),
                        "avg_duration": round(df['duration'].mean(), 1)
                    }
                elif dataset_name == "users":
                    dataset_summary["insights"] = {
                        "age_groups": df['age_group'].value_counts().to_dict(),
                        "countries": df['country'].value_counts().to_dict(),
                        "subscription_tiers": df['subscription_tier'].value_counts().to_dict(),
                        "active_users": df['is_active'].sum(),
                        "active_percentage": round(df['is_active'].mean() * 100, 1)
                    }
                elif dataset_name == "interactions":
                    dataset_summary["insights"] = {
                        "interaction_types": df['interaction_type'].value_counts().to_dict(),
                        "platforms": df['platform'].value_counts().to_dict(),
                        "devices": df['device_type'].value_counts().to_dict(),
                        "avg_rating": round(df['rating'].mean(), 2) if 'rating' in df.columns else None,
                        "total_watch_hours": round(df['watch_duration'].sum() / 60, 1) if 'watch_duration' in df.columns else None
                    }
                
                summary["datasets"][dataset_name] = dataset_summary
        
        # Save summary
        summary_path = f"{self.output_dir}/dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Created dataset summary: {summary_path}")
        return summary

def generate_complete_dataset(
    content_count: int = 1000,
    user_count: int = 5000,
    interaction_count: int = 50000,
    output_dir: str = "sample_data"
) -> Dict[str, Any]:
    """Generate complete sample dataset"""
    logger.info("Starting complete dataset generation...")
    
    # Initialize generators
    content_gen = ContentDataGenerator()
    user_gen = UserDataGenerator()
    
    # Generate content items
    logger.info("Generating content items...")
    content_items = content_gen.generate_content_items(content_count)
    
    # Generate users
    logger.info("Generating users...")
    users = user_gen.generate_users(user_count)
    
    # Generate interactions
    logger.info("Generating interactions...")
    interaction_gen = InteractionDataGenerator(users, content_items)
    interactions = interaction_gen.generate_interactions(interaction_count)
    
    # Prepare datasets
    datasets = {
        "content_items": content_items,
        "users": users,
        "interactions": interactions
    }
    
    # Export datasets
    logger.info("Exporting datasets...")
    exporter = DatasetExporter(output_dir)
    
    exported_files = {}
    for dataset_name, data in datasets.items():
        exported_files[dataset_name] = {
            "json": exporter.export_to_json(data, dataset_name),
            "csv": exporter.export_to_csv(data, dataset_name),
            "parquet": exporter.export_to_parquet(data, dataset_name)
        }
    
    # Create summary
    summary = exporter.create_data_summary(datasets)
    
    logger.info("Dataset generation completed successfully!")
    
    return {
        "datasets": datasets,
        "exported_files": exported_files,
        "summary": summary
    }

if __name__ == "__main__":
    # Generate complete sample dataset
    result = generate_complete_dataset(
        content_count=1000,
        user_count=5000,
        interaction_count=50000,
        output_dir="c:/Git/capstone/sample_data"
    )
    
    print("Sample dataset generation completed!")
    print(f"Generated datasets:")
    for dataset_name, summary in result["summary"]["datasets"].items():
        print(f"  {dataset_name}: {summary['record_count']} records")
    
    print(f"\nExported files:")
    for dataset_name, files in result["exported_files"].items():
        print(f"  {dataset_name}:")
        for format_type, filepath in files.items():
            print(f"    {format_type}: {filepath}")