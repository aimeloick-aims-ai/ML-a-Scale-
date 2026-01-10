import sqlite3
import csv
import os


class UserRatingsManager:
    """Database manager for user ratings"""
    
    def __init__(self, db_path="user_ratings.db", movies_csv_path="ml-32m/movies.csv"):
        self.db_path = db_path
        self.movies_csv_path = movies_csv_path
        self.conn = None
        self.init_database()
        self.load_movies_mapping()
    
    def init_database(self):
        """Initialize the SQLite database"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Ratings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, movie_id)
            )
        ''')
        
        # Liked movies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_likes (
                like_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, movie_id)
            )
        ''')
        
        # Watchlist table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_watchlist (
                watchlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, movie_id)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_ratings ON user_ratings(user_id, movie_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_likes ON user_likes(user_id, movie_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_watchlist ON user_watchlist(user_id, movie_id)')
        
        self.conn.commit()
        print("✓ Database initialized")
    
    def load_movies_mapping(self):
        """Load movieId -> title mapping from movies.csv"""
        self.movieid_to_title = {}
        self.title_to_movieid = {}
        
        if not os.path.exists(self.movies_csv_path):
            print(f"⚠️ File {self.movies_csv_path} not found")
            return
        
        with open(self.movies_csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                movie_id = int(row[0])
                title = row[1]
                self.movieid_to_title[movie_id] = title
                self.title_to_movieid[title] = movie_id
        
        print(f"✓ {len(self.movieid_to_title)} movies loaded into the mapping")
    
    
    def create_user(self, username="default_user"):
        """Create a new user"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
            self.conn.commit()
            user_id = cursor.lastrowid
            print(f"✓ User '{username}' created with ID: {user_id}")
            return user_id
        except sqlite3.IntegrityError:
            cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
            user_id = cursor.fetchone()[0]
            print(f"✓ User '{username}' already exists with ID: {user_id}")
            return user_id
    
    def get_user_id(self, username="default_user"):
        """Get a user's ID or create the user if it does not exist"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            return self.create_user(username)
    
    
    def add_rating(self, movie_id, rating, username="default_user"):
        """Add or update a rating"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO user_ratings (user_id, movie_id, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, movie_id) 
                DO UPDATE SET rating = ?, updated_at = CURRENT_TIMESTAMP
            ''', (user_id, movie_id, rating, rating))
            self.conn.commit()
            
            movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
            print(f"✓ Rating {rating}/5 added for '{movie_title}'")
            return True
        except Exception as e:
            print(f"❌ Error while adding rating: {e}")
            return False
    
    def get_user_ratings(self, username="default_user"):
        """Retrieve all ratings from a user"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT movie_id, rating, created_at, updated_at
            FROM user_ratings
            WHERE user_id = ?
            ORDER BY updated_at DESC
        ''', (user_id,))
        
        ratings = cursor.fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in ratings]
    
    def get_user_ratings_for_recommendation(self, username="default_user"):
        """Retrieve ratings as a list for the recommendation model"""
        ratings = self.get_user_ratings(username)
        # Format: [(movie_id, rating), ...]
        return [(movie_id, rating) for movie_id, rating, _, _ in ratings]
    
    def delete_rating(self, movie_id, username="default_user"):
        """Delete a rating"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            DELETE FROM user_ratings
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        self.conn.commit()
        
        movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
        print(f"✓ Rating deleted for '{movie_title}'")
        return True
    
    # ============================================================================
    # LIKES MANAGEMENT
    # ============================================================================
    
    def add_like(self, movie_id, username="default_user"):
        """Add a like to a movie"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO user_likes (user_id, movie_id)
                VALUES (?, ?)
            ''', (user_id, movie_id))
            self.conn.commit()
            
            movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
            print(f"✓ Like added for '{movie_title}'")
            return True
        except sqlite3.IntegrityError:
            print("⚠️ Movie already liked")
            return False
    
    def remove_like(self, movie_id, username="default_user"):
        """Remove a like"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            DELETE FROM user_likes
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        self.conn.commit()
        
        movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
        print(f"✓ Like removed for '{movie_title}'")
        return True
    
    def get_user_likes(self, username="default_user"):
        """Retrieve all liked movies for a user"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT movie_id, created_at
            FROM user_likes
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        likes = cursor.fetchall()
        return [like[0] for like in likes]
    
    def is_liked(self, movie_id, username="default_user"):
        """Check if a movie is liked"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM user_likes
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        
        return cursor.fetchone()[0] > 0
    
    # ============================================================================
    # WATCHLIST MANAGEMENT
    # ============================================================================
    
    def add_to_watchlist(self, movie_id, username="default_user"):
        """Add a movie to the watchlist"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO user_watchlist (user_id, movie_id)
                VALUES (?, ?)
            ''', (user_id, movie_id))
            self.conn.commit()
            
            movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
            print(f"✓ Movie added to watchlist: '{movie_title}'")
            return True
        except sqlite3.IntegrityError:
            print("⚠️ Movie already in watchlist")
            return False
    
    def remove_from_watchlist(self, movie_id, username="default_user"):
        """Remove a movie from the watchlist"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            DELETE FROM user_watchlist
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        self.conn.commit()
        
        movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
        print(f"✓ Movie removed from watchlist: '{movie_title}'")
        return True
    
    def get_user_watchlist(self, username="default_user"):
        """Retrieve a user's watchlist"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT movie_id, added_at
            FROM user_watchlist
            WHERE user_id = ?
            ORDER BY added_at DESC
        ''', (user_id,))
        
        watchlist = cursor.fetchall()
        return [item[0] for item in watchlist]
    
    def is_in_watchlist(self, movie_id, username="default_user"):
        """Check if a movie is in the watchlist"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM user_watchlist
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        
        return cursor.fetchone()[0] > 0
    
    # ============================================================================
    # STATISTICS AND UTILITIES
    # ============================================================================
    
    def get_user_stats(self, username="default_user"):
        """Retrieve user statistics"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        # Number of ratings
        cursor.execute('SELECT COUNT(*) FROM user_ratings WHERE user_id = ?', (user_id,))
        num_ratings = cursor.fetchone()[0]
        
        # Number of likes
        cursor.execute('SELECT COUNT(*) FROM user_likes WHERE user_id = ?', (user_id,))
        num_likes = cursor.fetchone()[0]
        
        # Number of movies in watchlist
        cursor.execute('SELECT COUNT(*) FROM user_watchlist WHERE user_id = ?', (user_id,))
        num_watchlist = cursor.fetchone()[0]
        
        # Average rating
        cursor.execute('SELECT AVG(rating) FROM user_ratings WHERE user_id = ?', (user_id,))
        avg_rating = cursor.fetchone()[0] or 0
        
        return {
            'num_ratings': num_ratings,
            'num_likes': num_likes,
            'num_watchlist': num_watchlist,
            'avg_rating': round(avg_rating, 2)
        }
    
    def get_recent_activity(self, username="default_user", limit=10):
        """Retrieve recent user activity"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()
        
        # Combine recent activities
        cursor.execute('''
            SELECT 'rating' as type, movie_id, rating as value, updated_at as timestamp
            FROM user_ratings
            WHERE user_id = ?
            UNION ALL
            SELECT 'like' as type, movie_id, NULL as value, created_at as timestamp
            FROM user_likes
            WHERE user_id = ?
            UNION ALL
            SELECT 'watchlist' as type, movie_id, NULL as value, added_at as timestamp
            FROM user_watchlist
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, user_id, user_id, limit))
        
        activities = cursor.fetchall()
        
        result = []
        for activity in activities:
            activity_type, movie_id, value, timestamp = activity
            movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
            result.append({
                'type': activity_type,
                'movie_id': movie_id,
                'movie_title': movie_title,
                'value': value,
                'timestamp': timestamp
            })
        
        return result
    
    def clear_database(self):
        """Delete all data and tables from the database"""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS user_ratings")
        cursor.execute("DROP TABLE IF EXISTS user_likes")
        cursor.execute("DROP TABLE IF EXISTS user_watchlist")
        cursor.execute("DROP TABLE IF EXISTS users")
        self.conn.commit()
        print("✓ Database cleared")
        # Reinitialize the DB to reuse tables
        self.init_database()
    
    def export_user_ratings_csv(self, username="default_user", output_path="my_ratings.csv"):
        """Export user ratings to a CSV file"""
        ratings = self.get_user_ratings(username)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['movieId', 'title', 'rating', 'created_at', 'updated_at'])
            
            for movie_id, rating, created_at, updated_at in ratings:
                title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
                writer.writerow([movie_id, title, rating, created_at, updated_at])
        
        print(f"✓ Ratings exported to {output_path}")
        return output_path
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize manager
    manager = UserRatingsManager()
    
    # Create a user
    user_id = manager.create_user("john_doe")
    
    # Add ratings
    manager.add_rating(movie_id=1, rating=5.0, username="john_doe")
    manager.add_rating(movie_id=2, rating=4.5, username="john_doe")
    manager.add_rating(movie_id=3, rating=3.5, username="john_doe")
    
    # Add likes
    manager.add_like(movie_id=1, username="john_doe")
    manager.add_like(movie_id=5, username="john_doe")
    
    # Add to watchlist
    manager.add_to_watchlist(movie_id=10, username="john_doe")
    manager.add_to_watchlist(movie_id=15, username="john_doe")
    
    # Retrieve ratings
    ratings = manager.get_user_ratings("john_doe")
    print(f"\n📊 Ratings for john_doe: {len(ratings)}")
    for movie_id, rating, created, updated in ratings:
        print(f"  - Movie {movie_id}: {rating}/5")
    
    # Retrieve stats
    stats = manager.get_user_stats("john_doe")
    print(f"  - Ratings: {stats['num_ratings']}")
    print(f"  - Likes: {stats['num_likes']}")
    print(f"  - Watchlist: {stats['num_watchlist']}")
    print(f"  - Average rating: {stats['avg_rating']}/5")
    
    # Recent activity
    activities = manager.get_recent_activity("john_doe", limit=5)
    for activity in activities:
        print(f"  - {activity['type']}: {activity['movie_title']} ({activity['timestamp']})")
    
    # Export ratings
    manager.export_user_ratings_csv("john_doe")
    manager.clear_database()
    manager.close()
