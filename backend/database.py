import sqlite3
import json
from datetime import datetime, timedelta
import random

class Database:
    def __init__(self, db_name="venue_recommendations.db"):
        self.db_name = db_name
        self.init_db()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                interests TEXT,
                bio TEXT,
                profile_pic TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Venues table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS venues (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                category TEXT,
                rating REAL,
                description TEXT,
                image_url TEXT,
                capacity INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Posts table (Instagram-like)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                venue_id INTEGER,
                caption TEXT,
                image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (venue_id) REFERENCES venues(id)
            )
        ''')
        
        # Post engagement tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS post_views (
                id INTEGER PRIMARY KEY,
                post_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                time_spent INTEGER DEFAULT 0,
                viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES posts(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Filter interactions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS filter_interactions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filter_type TEXT,
                filter_value TEXT,
                interaction_count INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, filter_type, filter_value),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Groups (Instagram-like communities)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                interest_tag TEXT,
                member_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Group memberships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS group_members (
                id INTEGER PRIMARY KEY,
                group_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(group_id, user_id),
                FOREIGN KEY (group_id) REFERENCES groups(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Bookings (Agent-generated)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                venue_id INTEGER NOT NULL,
                booking_date TIMESTAMP NOT NULL,
                party_size INTEGER,
                status TEXT DEFAULT 'pending',
                companion_ids TEXT,
                agent_generated BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (venue_id) REFERENCES venues(id)
            )
        ''')
        
        # Social connections
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS connections (
                id INTEGER PRIMARY KEY,
                user_id_1 INTEGER NOT NULL,
                user_id_2 INTEGER NOT NULL,
                compatibility_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id_1, user_id_2),
                FOREIGN KEY (user_id_1) REFERENCES users(id),
                FOREIGN KEY (user_id_2) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ==================== USER QUERIES ====================
    def add_user(self, username, latitude, longitude, interests, bio):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (username, latitude, longitude, interests, bio, profile_pic)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, latitude, longitude, json.dumps(interests), bio, f"https://i.pravatar.cc/150?u={username}"))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    
    def get_user(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        return user
    
    def get_all_users(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()
        conn.close()
        return users
    
    # ==================== VENUE QUERIES ====================
    def add_venue(self, name, latitude, longitude, category, rating, description):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO venues (name, latitude, longitude, category, rating, description, capacity, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, latitude, longitude, category, rating, description, random.randint(20, 200), 
              f"https://via.placeholder.com/400?text={name}"))
        conn.commit()
        venue_id = cursor.lastrowid
        conn.close()
        return venue_id
    
    def get_venue(self, venue_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM venues WHERE id = ?', (venue_id,))
        venue = cursor.fetchone()
        conn.close()
        return venue
    
    def get_all_venues(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM venues')
        venues = cursor.fetchall()
        conn.close()
        return venues
    
    # ==================== POST QUERIES ====================
    def add_post(self, user_id, venue_id, caption):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO posts (user_id, venue_id, caption, image_url)
            VALUES (?, ?, ?, ?)
        ''', (user_id, venue_id, caption, f"https://via.placeholder.com/600?text=Post"))
        conn.commit()
        post_id = cursor.lastrowid
        conn.close()
        return post_id
    
    def track_post_view(self, post_id, user_id, time_spent):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO post_views (post_id, user_id, time_spent, viewed_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (post_id, user_id, time_spent))
        conn.commit()
        conn.close()
    
    def get_user_posts(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM posts WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
        posts = cursor.fetchall()
        conn.close()
        return posts
    
    # ==================== FILTER INTERACTIONS ====================
    def track_filter_interaction(self, user_id, filter_type, filter_value):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO filter_interactions (user_id, filter_type, filter_value, interaction_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(user_id, filter_type, filter_value) DO UPDATE SET
            interaction_count = interaction_count + 1,
            last_used = CURRENT_TIMESTAMP
        ''', (user_id, filter_type, filter_value))
        conn.commit()
        conn.close()
    
    def get_user_filter_history(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT filter_type, filter_value, interaction_count 
            FROM filter_interactions 
            WHERE user_id = ? 
            ORDER BY interaction_count DESC
        ''', (user_id,))
        history = cursor.fetchall()
        conn.close()
        return history
    
    # ==================== GROUP QUERIES ====================
    def create_group(self, name, description, interest_tag):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO groups (name, description, interest_tag)
            VALUES (?, ?, ?)
        ''', (name, description, interest_tag))
        conn.commit()
        group_id = cursor.lastrowid
        conn.close()
        return group_id
    
    def add_group_member(self, group_id, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO group_members (group_id, user_id)
            VALUES (?, ?)
        ''', (group_id, user_id))
        cursor.execute('UPDATE groups SET member_count = member_count + 1 WHERE id = ?', (group_id,))
        conn.commit()
        conn.close()
    
    def get_user_groups(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT g.* FROM groups g
            JOIN group_members gm ON g.id = gm.group_id
            WHERE gm.user_id = ?
        ''', (user_id,))
        groups = cursor.fetchall()
        conn.close()
        return groups
    
    def get_all_groups(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM groups ORDER BY member_count DESC')
        groups = cursor.fetchall()
        conn.close()
        return groups
    
    # ==================== BOOKING QUERIES ====================
    def add_booking(self, user_id, venue_id, booking_date, party_size, companion_ids=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO bookings (user_id, venue_id, booking_date, party_size, companion_ids, agent_generated)
            VALUES (?, ?, ?, ?, ?, 1)
        ''', (user_id, venue_id, booking_date, party_size, json.dumps(companion_ids) if companion_ids else None))
        conn.commit()
        booking_id = cursor.lastrowid
        conn.close()
        return booking_id
    
    def get_user_bookings(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM bookings 
            WHERE user_id = ? 
            ORDER BY booking_date DESC
        ''', (user_id,))
        bookings = cursor.fetchall()
        conn.close()
        return bookings
    
    # ==================== CONNECTION QUERIES ====================
    def add_connection(self, user_id_1, user_id_2, compatibility_score):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO connections (user_id_1, user_id_2, compatibility_score)
                VALUES (?, ?, ?)
            ''', (user_id_1, user_id_2, compatibility_score))
            conn.commit()
        except:
            pass
        conn.close()
    
    def get_user_connections(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM connections 
            WHERE (user_id_1 = ? OR user_id_2 = ?)
            ORDER BY compatibility_score DESC
        ''', (user_id, user_id))
        connections = cursor.fetchall()
        conn.close()
        return connections