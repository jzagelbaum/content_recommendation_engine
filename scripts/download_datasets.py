#!/usr/bin/env python3
"""
scripts/download_datasets.py
Download and prepare real-world recommendation datasets

Supported Datasets:
1. MovieLens - Movie ratings (similar to Netflix Prize)
2. Amazon Product Reviews - E-commerce ratings
3. Book-Crossing - Book ratings
4. Jester - Joke ratings
5. Last.fm - Music listening history
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional
import csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and prepare real-world recommendation datasets"""
    
    DATASETS = {
        'movielens-small': {
            'name': 'MovieLens 100K',
            'description': 'MovieLens dataset with 100,000 ratings from 600 users on 9,000 movies',
            'url': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'size': '~1 MB',
            'records': '100K ratings',
            'type': 'movies'
        },
        'movielens-1m': {
            'name': 'MovieLens 1M',
            'description': 'MovieLens dataset with 1 million ratings from 6,000 users on 4,000 movies',
            'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'size': '~6 MB',
            'records': '1M ratings',
            'type': 'movies'
        },
        'movielens-25m': {
            'name': 'MovieLens 25M',
            'description': 'MovieLens dataset with 25 million ratings from 162,000 users on 62,000 movies',
            'url': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip',
            'size': '~250 MB',
            'records': '25M ratings',
            'type': 'movies'
        },
        'book-crossing': {
            'name': 'Book-Crossing Dataset',
            'description': 'Book ratings from Book-Crossing community',
            'url': 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip',
            'size': '~25 MB',
            'records': '1.1M ratings',
            'type': 'books'
        }
    }
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = output_dir / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def list_datasets(self):
        """List all available datasets"""
        logger.info("Available datasets:")
        logger.info("=" * 80)
        for key, info in self.DATASETS.items():
            logger.info(f"\n{key}:")
            logger.info(f"  Name: {info['name']}")
            logger.info(f"  Description: {info['description']}")
            logger.info(f"  Size: {info['size']}")
            logger.info(f"  Records: {info['records']}")
            logger.info(f"  Type: {info['type']}")
        logger.info("=" * 80)
    
    def download_file(self, url: str, destination: Path) -> Path:
        """Download a file with progress reporting"""
        logger.info(f"Downloading from {url}...")
        
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                if percent % 10 == 0 and count * block_size < total_size:
                    logger.info(f"  Progress: {percent}%")
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        logger.info(f"✓ Downloaded to {destination}")
        return destination
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> Path:
        """Extract a zip file"""
        logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✓ Extracted to {extract_to}")
        return extract_to
    
    def download_movielens(self, variant: str = 'movielens-small') -> Dict[str, Any]:
        """Download and process MovieLens dataset"""
        if variant not in ['movielens-small', 'movielens-1m', 'movielens-25m']:
            raise ValueError(f"Invalid MovieLens variant: {variant}")
        
        dataset_info = self.DATASETS[variant]
        logger.info(f"Downloading {dataset_info['name']}...")
        
        # Download
        zip_path = self.temp_dir / f"{variant}.zip"
        self.download_file(dataset_info['url'], zip_path)
        
        # Extract
        extract_path = self.temp_dir / variant
        self.extract_zip(zip_path, extract_path)
        
        # Find the extracted directory (it may have a different name)
        extracted_dirs = [d for d in extract_path.iterdir() if d.is_dir()]
        if extracted_dirs:
            data_dir = extracted_dirs[0]
        else:
            data_dir = extract_path
        
        # Process based on variant
        if variant == 'movielens-small' or variant == 'movielens-25m':
            return self._process_movielens_csv(data_dir, variant)
        else:  # movielens-1m
            return self._process_movielens_dat(data_dir, variant)
    
    def _process_movielens_csv(self, data_dir: Path, variant: str) -> Dict[str, Any]:
        """Process MovieLens CSV format (small and 25m variants)"""
        logger.info("Processing MovieLens CSV files...")
        
        users = []
        movies = []
        ratings = []
        
        # Read movies
        movies_file = data_dir / 'movies.csv'
        if movies_file.exists():
            logger.info("Reading movies...")
            with open(movies_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    movies.append({
                        'product_id': f"movie_{row['movieId']}",
                        'name': row['title'],
                        'category': 'movies',
                        'genres': row['genres'].split('|') if row['genres'] != '(no genres listed)' else [],
                        'description': f"Movie: {row['title']}"
                    })
        
        # Read ratings
        ratings_file = data_dir / 'ratings.csv'
        if ratings_file.exists():
            logger.info("Reading ratings...")
            user_ids = set()
            with open(ratings_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_ids.add(row['userId'])
                    ratings.append({
                        'user_id': f"user_{row['userId']}",
                        'product_id': f"movie_{row['movieId']}",
                        'rating': float(row['rating']),
                        'timestamp': int(row['timestamp'])
                    })
        
        # Generate users
        logger.info("Generating user profiles...")
        for user_id in sorted(user_ids):
            users.append({
                'user_id': f"user_{user_id}",
                'created_at': None  # Not available in dataset
            })
        
        # Save processed data
        self._save_processed_data(users, movies, ratings, variant)
        
        return {
            'users': len(users),
            'movies': len(movies),
            'ratings': len(ratings),
            'source': variant
        }
    
    def _process_movielens_dat(self, data_dir: Path, variant: str) -> Dict[str, Any]:
        """Process MovieLens DAT format (1m variant)"""
        logger.info("Processing MovieLens DAT files...")
        
        users = []
        movies = []
        ratings = []
        
        # Read movies (movies.dat: MovieID::Title::Genres)
        movies_file = data_dir / 'movies.dat'
        if movies_file.exists():
            logger.info("Reading movies...")
            with open(movies_file, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 3:
                        movies.append({
                            'product_id': f"movie_{parts[0]}",
                            'name': parts[1],
                            'category': 'movies',
                            'genres': parts[2].split('|') if parts[2] else [],
                            'description': f"Movie: {parts[1]}"
                        })
        
        # Read ratings (ratings.dat: UserID::MovieID::Rating::Timestamp)
        ratings_file = data_dir / 'ratings.dat'
        if ratings_file.exists():
            logger.info("Reading ratings...")
            user_ids = set()
            with open(ratings_file, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 4:
                        user_ids.add(parts[0])
                        ratings.append({
                            'user_id': f"user_{parts[0]}",
                            'product_id': f"movie_{parts[1]}",
                            'rating': float(parts[2]),
                            'timestamp': int(parts[3])
                        })
        
        # Read users (users.dat: UserID::Gender::Age::Occupation::Zip-code)
        users_file = data_dir / 'users.dat'
        if users_file.exists():
            logger.info("Reading user demographics...")
            with open(users_file, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 5:
                        users.append({
                            'user_id': f"user_{parts[0]}",
                            'gender': parts[1],
                            'age': int(parts[2]),
                            'occupation': int(parts[3]),
                            'zip_code': parts[4]
                        })
        else:
            # Generate basic users if demographics not available
            for user_id in sorted(user_ids):
                users.append({
                    'user_id': f"user_{user_id}",
                    'created_at': None
                })
        
        # Save processed data
        self._save_processed_data(users, movies, ratings, variant)
        
        return {
            'users': len(users),
            'movies': len(movies),
            'ratings': len(ratings),
            'source': variant
        }
    
    def download_book_crossing(self) -> Dict[str, Any]:
        """Download and process Book-Crossing dataset"""
        logger.info("Downloading Book-Crossing dataset...")
        
        dataset_info = self.DATASETS['book-crossing']
        
        # Download
        zip_path = self.temp_dir / 'book-crossing.zip'
        self.download_file(dataset_info['url'], zip_path)
        
        # Extract
        extract_path = self.temp_dir / 'book-crossing'
        self.extract_zip(zip_path, extract_path)
        
        users = []
        books = []
        ratings = []
        
        # Read books (BX-Books.csv)
        books_file = extract_path / 'BX-Books.csv'
        if books_file.exists():
            logger.info("Reading books...")
            with open(books_file, 'r', encoding='latin-1') as f:
                reader = csv.reader(f, delimiter=';', quotechar='"')
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 8:
                        books.append({
                            'product_id': f"book_{row[0]}",
                            'name': row[1],
                            'author': row[2],
                            'year': row[3],
                            'publisher': row[4],
                            'category': 'books',
                            'description': f"Book: {row[1]} by {row[2]}"
                        })
        
        # Read ratings (BX-Book-Ratings.csv)
        ratings_file = extract_path / 'BX-Book-Ratings.csv'
        if ratings_file.exists():
            logger.info("Reading ratings...")
            user_ids = set()
            with open(ratings_file, 'r', encoding='latin-1') as f:
                reader = csv.reader(f, delimiter=';', quotechar='"')
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        user_ids.add(row[0])
                        rating_value = int(row[2])
                        if rating_value > 0:  # Only explicit ratings
                            ratings.append({
                                'user_id': f"user_{row[0]}",
                                'product_id': f"book_{row[1]}",
                                'rating': float(rating_value)
                            })
        
        # Generate users
        logger.info("Generating user profiles...")
        for user_id in sorted(user_ids):
            users.append({
                'user_id': f"user_{user_id}",
                'created_at': None
            })
        
        # Save processed data
        self._save_processed_data(users, books, ratings, 'book-crossing')
        
        return {
            'users': len(users),
            'books': len(books),
            'ratings': len(ratings),
            'source': 'book-crossing'
        }
    
    def _save_processed_data(self, users, products, interactions, dataset_name):
        """Save processed data to JSON files"""
        logger.info("Saving processed data...")
        
        # Save users
        users_file = self.output_dir / f'{dataset_name}_users.json'
        with open(users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2)
        logger.info(f"✓ Saved {len(users)} users to {users_file}")
        
        # Save products
        products_file = self.output_dir / f'{dataset_name}_products.json'
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=2)
        logger.info(f"✓ Saved {len(products)} products to {products_file}")
        
        # Save interactions (convert ratings to interactions format)
        interactions_data = []
        for interaction in interactions:
            interaction_data = {
                'user_id': interaction['user_id'],
                'product_id': interaction['product_id'],
                'type': 'rating',
                'rating': interaction.get('rating'),
                'timestamp': interaction.get('timestamp')
            }
            interactions_data.append(interaction_data)
        
        interactions_file = self.output_dir / f'{dataset_name}_interactions.json'
        with open(interactions_file, 'w', encoding='utf-8') as f:
            json.dump(interactions_data, f, indent=2)
        logger.info(f"✓ Saved {len(interactions_data)} interactions to {interactions_file}")
        
        # Save summary
        summary = {
            'dataset': dataset_name,
            'downloaded_at': str(Path.ctime(users_file)),
            'num_users': len(users),
            'num_products': len(products),
            'num_interactions': len(interactions_data),
            'files': {
                'users': str(users_file.name),
                'products': str(products_file.name),
                'interactions': str(interactions_file.name)
            }
        }
        
        summary_file = self.output_dir / f'{dataset_name}_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary to {summary_file}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(self.temp_dir)
            logger.info("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare real-world recommendation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Datasets:
  movielens-small   MovieLens 100K - Small dataset perfect for testing
  movielens-1m      MovieLens 1M - Medium-sized dataset (recommended)
  movielens-25m     MovieLens 25M - Large dataset for production testing
  book-crossing     Book-Crossing - Book ratings dataset

Examples:
  # Download MovieLens 1M dataset (recommended)
  python download_datasets.py movielens-1m

  # Download to custom directory
  python download_datasets.py movielens-small -o data/real

  # List all available datasets
  python download_datasets.py --list

  # Download without cleanup
  python download_datasets.py movielens-1m --keep-temp
        """
    )
    
    parser.add_argument(
        'dataset',
        nargs='?',
        choices=['movielens-small', 'movielens-1m', 'movielens-25m', 'book-crossing'],
        help='Dataset to download'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets'
    )
    
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary download files'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    output_dir = Path(args.output)
    downloader = DatasetDownloader(output_dir)
    
    # List datasets if requested
    if args.list:
        downloader.list_datasets()
        return
    
    # Check if dataset specified
    if not args.dataset:
        parser.print_help()
        return
    
    try:
        logger.info(f"Starting download of {args.dataset}...")
        logger.info(f"Output directory: {output_dir}")
        
        # Download and process dataset
        if args.dataset.startswith('movielens'):
            stats = downloader.download_movielens(args.dataset)
        elif args.dataset == 'book-crossing':
            stats = downloader.download_book_crossing()
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            return
        
        # Cleanup temporary files
        if not args.keep_temp:
            downloader.cleanup()
        
        # Display summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ Download complete!")
        logger.info("=" * 80)
        logger.info(f"Dataset: {stats['source']}")
        logger.info(f"Users: {stats.get('users', 0):,}")
        logger.info(f"Products: {stats.get('movies', stats.get('books', 0)):,}")
        logger.info(f"Ratings: {stats.get('ratings', 0):,}")
        logger.info(f"Files saved to: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
