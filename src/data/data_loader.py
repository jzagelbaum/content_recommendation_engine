"""
Data Loader for Content Recommendation Engine
=============================================

This module handles data loading from various sources including:
- Azure Data Lake Storage Gen2
- Azure SQL Database
- Local files
- Sample data generation

Author: Content Recommendation Engine Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import os
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for the content recommendation engine
    """
    
    def __init__(self, config: Any):
        """
        Initialize the data loader
        
        Args:
            config: Configuration object with data source settings
        """
        self.config = config
        self.credential = DefaultAzureCredential()
        self.data_lake_client = None
        self._setup_data_lake_client()
    
    def _setup_data_lake_client(self):
        """Setup Azure Data Lake client"""
        try:
            if hasattr(self.config, 'data_lake_account_name'):
                account_url = f"https://{self.config.data_lake_account_name}.dfs.core.windows.net"
                self.data_lake_client = DataLakeServiceClient(
                    account_url=account_url,
                    credential=self.credential
                )
                logger.info("Data Lake client initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Data Lake client: {e}")
    
    def load_interactions_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load user-item interaction data
        
        Args:
            file_path: Optional specific file path
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        logger.info("Loading interactions data...")
        
        if file_path and os.path.exists(file_path):
            # Load from local file
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format")
        else:
            # Try to load from Data Lake
            df = self._load_from_data_lake("raw-data/interactions.parquet")
            
            if df is None:
                # Generate sample data if no data source available
                logger.warning("No interaction data found, generating sample data")
                df = self._generate_sample_interactions()
        
        # Validate and clean data
        df = self._validate_interactions_data(df)
        
        logger.info(f"Loaded {len(df)} interaction records")
        return df
    
    def load_content_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load content metadata
        
        Args:
            file_path: Optional specific file path
            
        Returns:
            DataFrame with content features
        """
        logger.info("Loading content data...")
        
        if file_path and os.path.exists(file_path):
            # Load from local file
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format")
        else:
            # Try to load from Data Lake
            df = self._load_from_data_lake("raw-data/content.parquet")
            
            if df is None:
                # Generate sample data if no data source available
                logger.warning("No content data found, generating sample data")
                df = self._generate_sample_content()
        
        # Validate and clean data
        df = self._validate_content_data(df)
        
        logger.info(f"Loaded {len(df)} content items")
        return df
    
    def load_user_profiles(self, file_path: str = None) -> pd.DataFrame:
        """
        Load user profile data
        
        Args:
            file_path: Optional specific file path
            
        Returns:
            DataFrame with user features
        """
        logger.info("Loading user profile data...")
        
        if file_path and os.path.exists(file_path):
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format")
        else:
            # Try to load from Data Lake
            df = self._load_from_data_lake("raw-data/users.parquet")
            
            if df is None:
                # Generate sample data
                logger.warning("No user profile data found, generating sample data")
                df = self._generate_sample_users()
        
        logger.info(f"Loaded {len(df)} user profiles")
        return df
    
    def _load_from_data_lake(self, blob_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from Azure Data Lake Storage
        
        Args:
            blob_path: Path to the blob in Data Lake
            
        Returns:
            DataFrame or None if not available
        """
        if not self.data_lake_client:
            return None
        
        try:
            file_system_name = getattr(self.config, 'data_lake_container', 'contentrec')
            file_system_client = self.data_lake_client.get_file_system_client(file_system_name)
            file_client = file_system_client.get_file_client(blob_path)
            
            # Download file content
            download = file_client.download_file()
            content = download.readall()
            
            # Load as DataFrame
            if blob_path.endswith('.parquet'):
                import io
                df = pd.read_parquet(io.BytesIO(content))
            elif blob_path.endswith('.csv'):
                import io
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            else:
                raise ValueError(f"Unsupported file format: {blob_path}")
            
            logger.info(f"Successfully loaded {blob_path} from Data Lake")
            return df
            
        except Exception as e:
            logger.warning(f"Could not load {blob_path} from Data Lake: {e}")
            return None
    
    def _validate_interactions_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean interactions data
        
        Args:
            df: Raw interactions DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Required columns
        required_columns = ['user_id', 'item_id', 'rating']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add timestamp if missing
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        
        # Convert data types
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        df['rating'] = df['rating'].astype(float)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id', 'item_id'])
        
        # Filter valid ratings (assuming 1-5 scale)
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        
        # Remove users/items with too few interactions
        min_interactions = getattr(self.config, 'min_interactions', 5)
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        logger.info(f"After validation: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        
        return df
    
    def _validate_content_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean content data
        
        Args:
            df: Raw content DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Required columns
        if 'item_id' not in df.columns:
            raise ValueError("Missing required column: item_id")
        
        # Ensure required text columns exist
        text_columns = ['title', 'description', 'genre', 'tags']
        for col in text_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Convert data types
        df['item_id'] = df['item_id'].astype(int)
        
        # Fill missing values
        for col in text_columns:
            df[col] = df[col].fillna('').astype(str)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['item_id'])
        
        return df
    
    def _generate_sample_interactions(self, num_users: int = 1000, num_items: int = 500, 
                                    num_interactions: int = 50000) -> pd.DataFrame:
        """
        Generate sample interaction data for development/testing
        
        Args:
            num_users: Number of users to generate
            num_items: Number of items to generate
            num_interactions: Number of interactions to generate
            
        Returns:
            Sample interactions DataFrame
        """
        logger.info("Generating sample interaction data...")
        
        np.random.seed(42)
        random.seed(42)
        
        # Generate interactions with realistic patterns
        interactions = []
        
        for _ in range(num_interactions):
            user_id = random.randint(1, num_users)
            item_id = random.randint(1, num_items)
            
            # Create realistic rating distribution (skewed toward higher ratings)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
            
            # Generate timestamp (last 2 years)
            days_ago = random.randint(0, 730)
            timestamp = datetime.now() - timedelta(days=days_ago)
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(interactions)
        return df
    
    def _generate_sample_content(self, num_items: int = 500) -> pd.DataFrame:
        """
        Generate sample content data for development/testing
        
        Args:
            num_items: Number of content items to generate
            
        Returns:
            Sample content DataFrame
        """
        logger.info("Generating sample content data...")
        
        np.random.seed(42)
        random.seed(42)
        
        # Sample data for content generation
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary']
        content_types = ['Movie', 'TV Show', 'Documentary', 'Short Film']
        adjectives = ['Amazing', 'Incredible', 'Fantastic', 'Brilliant', 'Epic', 'Stunning', 'Powerful', 'Beautiful']
        nouns = ['Adventure', 'Journey', 'Story', 'Tale', 'Experience', 'Mystery', 'Romance', 'Battle']
        
        content_data = []
        
        for item_id in range(1, num_items + 1):
            # Generate realistic content metadata
            title = f"{random.choice(adjectives)} {random.choice(nouns)} {item_id}"
            genre = random.choice(genres)
            content_type = random.choice(content_types)
            
            # Generate description
            description = f"A {genre.lower()} {content_type.lower()} about {random.choice(nouns).lower()}. " \
                         f"This {random.choice(adjectives).lower()} production features outstanding performances."
            
            # Generate tags
            num_tags = random.randint(2, 5)
            available_tags = ['award-winning', 'popular', 'trending', 'classic', 'new-release', 
                            'family-friendly', 'international', 'indie', 'blockbuster', 'acclaimed']
            tags = ', '.join(random.sample(available_tags, min(num_tags, len(available_tags))))
            
            # Generate additional metadata
            duration = random.randint(90, 180) if content_type == 'Movie' else random.randint(20, 60)
            release_year = random.randint(1980, 2025)
            rating = round(random.uniform(6.0, 9.5), 1)
            
            content_data.append({
                'item_id': item_id,
                'title': title,
                'description': description,
                'genre': genre,
                'content_type': content_type,
                'tags': tags,
                'duration_minutes': duration,
                'release_year': release_year,
                'imdb_rating': rating
            })
        
        df = pd.DataFrame(content_data)
        return df
    
    def _generate_sample_users(self, num_users: int = 1000) -> pd.DataFrame:
        """
        Generate sample user profile data for development/testing
        
        Args:
            num_users: Number of users to generate
            
        Returns:
            Sample users DataFrame
        """
        logger.info("Generating sample user data...")
        
        np.random.seed(42)
        random.seed(42)
        
        age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'ES', 'IT', 'BR', 'MX']
        preferred_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary']
        
        users_data = []
        
        for user_id in range(1, num_users + 1):
            # Generate user profile
            age_group = random.choice(age_groups)
            location = random.choice(locations)
            preferred_genre = random.choice(preferred_genres)
            
            # Generate usage patterns
            avg_rating = round(random.uniform(2.5, 4.5), 1)
            total_ratings = random.randint(10, 200)
            account_age_days = random.randint(30, 1825)  # 1 month to 5 years
            
            users_data.append({
                'user_id': user_id,
                'age_group': age_group,
                'location': location,
                'preferred_genre': preferred_genre,
                'avg_rating': avg_rating,
                'total_ratings': total_ratings,
                'account_age_days': account_age_days,
                'is_premium': random.choice([True, False])
            })
        
        df = pd.DataFrame(users_data)
        return df
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str, container: str = "processed-data"):
        """
        Save processed data to Data Lake or local storage
        
        Args:
            df: DataFrame to save
            file_path: Target file path
            container: Data Lake container name
        """
        try:
            if self.data_lake_client:
                # Save to Data Lake
                self._save_to_data_lake(df, f"{container}/{file_path}")
            else:
                # Save locally
                local_path = Path(f"./data/{container}/{file_path}")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                if file_path.endswith('.parquet'):
                    df.to_parquet(local_path, index=False)
                elif file_path.endswith('.csv'):
                    df.to_csv(local_path, index=False)
                
                logger.info(f"Saved data to {local_path}")
                
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def _save_to_data_lake(self, df: pd.DataFrame, blob_path: str):
        """
        Save DataFrame to Azure Data Lake Storage
        
        Args:
            df: DataFrame to save
            blob_path: Target blob path
        """
        try:
            file_system_name = getattr(self.config, 'data_lake_container', 'contentrec')
            file_system_client = self.data_lake_client.get_file_system_client(file_system_name)
            file_client = file_system_client.get_file_client(blob_path)
            
            # Convert DataFrame to bytes
            if blob_path.endswith('.parquet'):
                import io
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False)
                data = buffer.getvalue()
            elif blob_path.endswith('.csv'):
                data = df.to_csv(index=False).encode('utf-8')
            else:
                raise ValueError(f"Unsupported file format: {blob_path}")
            
            # Upload to Data Lake
            file_client.upload_data(data, overwrite=True)
            logger.info(f"Successfully saved {blob_path} to Data Lake")
            
        except Exception as e:
            logger.error(f"Error saving to Data Lake: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from utils.config import Config
    
    config = Config()
    data_loader = DataLoader(config)
    
    # Load sample data
    interactions_df = data_loader.load_interactions_data()
    content_df = data_loader.load_content_data()
    users_df = data_loader.load_user_profiles()
    
    print(f"Loaded {len(interactions_df)} interactions")
    print(f"Loaded {len(content_df)} content items")
    print(f"Loaded {len(users_df)} user profiles")