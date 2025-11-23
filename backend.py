import os
import json
import pickle
import hashlib
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from enum import Enum
import sqlite3
import qrcode
from io import BytesIO
import base64
import re
from collections import defaultdict, Counter
import math

from luna_agent import LunaAIAgent
# ML Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# AI Integration
import google.generativeai as genai

# After `import google.generativeai as genai`
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")  # or "gemini-2.5-flash"
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

# ========================= DATA MODELS =========================
class VenueCategory(Enum):
    CAFE = "Café"
    RESTAURANT = "Restaurant"
    NIGHTLIFE = "Nightlife"
    HIKING = "Hiking"
    ART = "Art Gallery"
    MOVIE = "Movie Theater"
    PARK = "Park"
    BEACH = "Beach"
    SHOPPING = "Shopping"
    GYM = "Gym"

class InteractionType(Enum):
    VIEW = "view"
    LIKE = "like"
    COMMENT = "comment"
    SAVE = "save"
    CLICK = "click"
    VISIT = "visit"

@dataclass
class User:
    user_id: str
    username: str
    email: str
    created_at: datetime
    location: Tuple[float, float]  # (lat, lon)
    bio: str = ""
    avatar_url: str = ""
    verified: bool = False
    interests: List[str] = field(default_factory=list)
    saved_venues: List[str] = field(default_factory=list)
    bookings: List[str] = field(default_factory=list)
    friends: List[str] = field(default_factory=list)
    blocked_users: List[str] = field(default_factory=list)
    
@dataclass
class Venue:
    venue_id: str
    name: str
    category: VenueCategory
    location: Tuple[float, float]
    description: str
    rating: float
    image_url: str
    address: str
    phone: str
    hours: str
    capacity: int
    website: str
    trending_score: float = 0.0
    
@dataclass
class Post:
    post_id: str
    user_id: str
    venue_id: Optional[str]
    content: str
    image_url: str
    likes: int = 0
    comments_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    tagged_users: List[str] = field(default_factory=list)
    
@dataclass
class Comment:
    comment_id: str
    post_id: str
    user_id: str
    content: str
    likes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass
class Interaction:
    interaction_id: str
    user_id: str
    venue_id: str
    interaction_type: InteractionType
    duration_seconds: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class Booking:
    booking_id: str
    user_id: str
    venue_id: str
    booking_date: datetime
    party_size: int
    special_notes: str = ""
    qr_code: str = ""
    status: str = "confirmed"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Group:
    group_id: str
    name: str
    description: str
    members: List[str]
    creator_id: str
    created_at: datetime
    venue_preferences: List[VenueCategory] = field(default_factory=list)
    planned_venues: List[Tuple[str, datetime]] = field(default_factory=list)

@dataclass
class Message:
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None if group message
    group_id: Optional[str]  # None if direct message
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    read: bool = False

# ========================= DATABASE LAYER =========================
class LunaDatabase:
    def __init__(self, db_path: str = "luna_social.db"):
        self.db_path = db_path
        self.init_db()

    def get_all_user_ids(self) -> List[str]:
        query = "SELECT user_id FROM users"
        results = self.execute_query(query, fetch=True)
        return [row[0] for row in results]
        
    def init_db(self):
        """Initialize SQLite database with all tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT,
            location TEXT,
            bio TEXT,
            avatar_url TEXT,
            verified BOOLEAN,
            interests TEXT,
            saved_venues TEXT,
            bookings TEXT,
            friends TEXT,
            blocked_users TEXT
        )''')
        
        # Venues table
        cursor.execute('''CREATE TABLE IF NOT EXISTS venues (
            venue_id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            location TEXT,
            description TEXT,
            rating REAL,
            image_url TEXT,
            address TEXT,
            phone TEXT,
            hours TEXT,
            capacity INTEGER,
            website TEXT,
            trending_score REAL
        )''')
        
        # Posts table
        cursor.execute('''CREATE TABLE IF NOT EXISTS posts (
            post_id TEXT PRIMARY KEY,
            user_id TEXT,
            venue_id TEXT,
            content TEXT,
            image_url TEXT,
            likes INTEGER,
            comments_count INTEGER,
            created_at TEXT,
            tagged_users TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (venue_id) REFERENCES venues (venue_id)
        )''')
        
        # Comments table
        cursor.execute('''CREATE TABLE IF NOT EXISTS comments (
            comment_id TEXT PRIMARY KEY,
            post_id TEXT,
            user_id TEXT,
            content TEXT,
            likes INTEGER,
            created_at TEXT,
            FOREIGN KEY (post_id) REFERENCES posts (post_id),
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )''')
        
        # Interactions table
        cursor.execute('''CREATE TABLE IF NOT EXISTS interactions (
            interaction_id TEXT PRIMARY KEY,
            user_id TEXT,
            venue_id TEXT,
            interaction_type TEXT,
            duration_seconds INTEGER,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (venue_id) REFERENCES venues (venue_id)
        )''')
        
        # Bookings table
        cursor.execute('''CREATE TABLE IF NOT EXISTS bookings (
            booking_id TEXT PRIMARY KEY,
            user_id TEXT,
            venue_id TEXT,
            booking_date TEXT,
            party_size INTEGER,
            special_notes TEXT,
            qr_code TEXT,
            status TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (venue_id) REFERENCES venues (venue_id)
        )''')
        
        # Groups table
        cursor.execute('''CREATE TABLE IF NOT EXISTS groups (
            group_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            members TEXT,
            creator_id TEXT,
            created_at TEXT,
            venue_preferences TEXT,
            planned_venues TEXT,
            FOREIGN KEY (creator_id) REFERENCES users (user_id)
        )''')
        
        # Messages table
        cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            sender_id TEXT,
            receiver_id TEXT,
            group_id TEXT,
            content TEXT,
            timestamp TEXT,
            read BOOLEAN,
            FOREIGN KEY (sender_id) REFERENCES users (user_id),
            FOREIGN KEY (receiver_id) REFERENCES users (user_id),
            FOREIGN KEY (group_id) REFERENCES groups (group_id)
        )''')
        
        # User behavior profiles table (for ML)
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            behavior_vector TEXT,
            preference_tags TEXT,
            category_scores TEXT,
            last_updated TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )''')
        
        conn.commit()
        conn.close()
        
    def execute_query(self, query: str, params: tuple = (), fetch: bool = False):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        result = cursor.fetchall() if fetch else None
        conn.close()
        return result
        
    def insert_user(self, user: User):
        query = '''INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            user.user_id, user.username, user.email, "", 
            user.created_at.isoformat(), json.dumps(user.location),
            user.bio, user.avatar_url, user.verified,
            json.dumps(user.interests), json.dumps(user.saved_venues),
            json.dumps(user.bookings), json.dumps(user.friends),
            json.dumps(user.blocked_users)
        )
        self.execute_query(query, params)
        
    def get_user(self, user_id: str) -> Optional[User]:
        query = 'SELECT * FROM users WHERE user_id = ?'
        result = self.execute_query(query, (user_id,), fetch=True)
        if result:
            row = result[0]
            return User(
                user_id=row[0], username=row[1], email=row[2],
                created_at=datetime.fromisoformat(row[4]),
                location=tuple(json.loads(row[5])),
                bio=row[6], avatar_url=row[7], verified=row[8],
                interests=json.loads(row[9]), saved_venues=json.loads(row[10]),
                bookings=json.loads(row[11]), friends=json.loads(row[12]),
                blocked_users=json.loads(row[13])
            )
        return None
        
    def insert_venue(self, venue: Venue):
        query = '''INSERT OR REPLACE INTO venues VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            venue.venue_id, venue.name, venue.category.value,
            json.dumps(venue.location), venue.description, venue.rating,
            venue.image_url, venue.address, venue.phone, venue.hours,
            venue.capacity, venue.website, venue.trending_score
        )
        self.execute_query(query, params)
        
    def get_venue(self, venue_id: str) -> Optional[Venue]:
        query = 'SELECT * FROM venues WHERE venue_id = ?'
        result = self.execute_query(query, (venue_id,), fetch=True)
        if result:
            row = result[0]
            return Venue(
                venue_id=row[0], name=row[1],
                category=VenueCategory(row[2]),
                location=tuple(json.loads(row[3])),
                description=row[4], rating=row[5],
                image_url=row[6], address=row[7], phone=row[8],
                hours=row[9], capacity=row[10], website=row[11],
                trending_score=row[12]
            )
        return None
        
    def get_all_venues(self) -> List[Venue]:
        query = 'SELECT * FROM venues'
        results = self.execute_query(query, fetch=True)
        venues = []
        for row in results:
            venues.append(Venue(
                venue_id=row[0], name=row[1],
                category=VenueCategory(row[2]),
                location=tuple(json.loads(row[3])),
                description=row[4], rating=row[5],
                image_url=row[6], address=row[7], phone=row[8],
                hours=row[9], capacity=row[10], website=row[11],
                trending_score=row[12]
            ))
        return venues
        
    def insert_post(self, post: Post):
        query = '''INSERT OR REPLACE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            post.post_id, post.user_id, post.venue_id, post.content,
            post.image_url, post.likes, post.comments_count,
            post.created_at.isoformat(), json.dumps(post.tagged_users)
        )
        self.execute_query(query, params)
        
    def get_posts(self, limit: int = 50, offset: int = 0) -> List[Post]:
        query = 'SELECT * FROM posts ORDER BY created_at DESC LIMIT ? OFFSET ?'
        results = self.execute_query(query, (limit, offset), fetch=True)
        posts = []
        for row in results:
            posts.append(Post(
                post_id=row[0], user_id=row[1], venue_id=row[2],
                content=row[3], image_url=row[4], likes=row[5],
                comments_count=row[6],
                created_at=datetime.fromisoformat(row[7]),
                tagged_users=json.loads(row[8])
            ))
        return posts
        
    def insert_comment(self, comment: Comment):
        query = '''INSERT OR REPLACE INTO comments VALUES (?, ?, ?, ?, ?, ?)'''
        params = (
            comment.comment_id, comment.post_id, comment.user_id,
            comment.content, comment.likes, comment.created_at.isoformat()
        )
        self.execute_query(query, params)
        
    def get_comments(self, post_id: str) -> List[Comment]:
        query = 'SELECT * FROM comments WHERE post_id = ? ORDER BY created_at DESC'
        results = self.execute_query(query, (post_id,), fetch=True)
        comments = []
        for row in results:
            comments.append(Comment(
                comment_id=row[0], post_id=row[1], user_id=row[2],
                content=row[3], likes=row[4],
                created_at=datetime.fromisoformat(row[5])
            ))
        return comments
        
    def insert_interaction(self, interaction: Interaction):
        query = '''INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?, ?)'''
        params = (
            interaction.interaction_id, interaction.user_id,
            interaction.venue_id, interaction.interaction_type.value,
            interaction.duration_seconds, interaction.timestamp.isoformat()
        )
        self.execute_query(query, params)
        
    def get_user_interactions(self, user_id: str) -> List[Interaction]:
        query = 'SELECT * FROM interactions WHERE user_id = ? ORDER BY timestamp DESC'
        results = self.execute_query(query, (user_id,), fetch=True)
        interactions = []
        for row in results:
            interactions.append(Interaction(
                interaction_id=row[0], user_id=row[1], venue_id=row[2],
                interaction_type=InteractionType(row[3]),
                duration_seconds=row[4],
                timestamp=datetime.fromisoformat(row[5])
            ))
        return interactions
        
    def insert_booking(self, booking: Booking):
        query = '''INSERT OR REPLACE INTO bookings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            booking.booking_id, booking.user_id, booking.venue_id,
            booking.booking_date.isoformat(), booking.party_size,
            booking.special_notes, booking.qr_code, booking.status,
            booking.created_at.isoformat()
        )
        self.execute_query(query, params)
        
    def get_user_bookings(self, user_id: str) -> List[Booking]:
        query = 'SELECT * FROM bookings WHERE user_id = ? ORDER BY booking_date DESC'
        results = self.execute_query(query, (user_id,), fetch=True)
        bookings = []
        for row in results:
            bookings.append(Booking(
                booking_id=row[0], user_id=row[1], venue_id=row[2],
                booking_date=datetime.fromisoformat(row[3]),
                party_size=row[4], special_notes=row[5],
                qr_code=row[6], status=row[7],
                created_at=datetime.fromisoformat(row[8])
            ))
        return bookings
        
    def insert_group(self, group: Group):
        query = '''INSERT OR REPLACE INTO groups VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            group.group_id, group.name, group.description,
            json.dumps(group.members), group.creator_id,
            group.created_at.isoformat(),
            json.dumps([cat.value for cat in group.venue_preferences]),
            json.dumps(group.planned_venues)
        )
        self.execute_query(query, params)
        
    def get_group(self, group_id: str) -> Optional[Group]:
        query = 'SELECT * FROM groups WHERE group_id = ?'
        result = self.execute_query(query, (group_id,), fetch=True)
        if result:
            row = result[0]
            return Group(
                group_id=row[0], name=row[1], description=row[2],
                members=json.loads(row[3]), creator_id=row[4],
                created_at=datetime.fromisoformat(row[5]),
                venue_preferences=[VenueCategory(cat) for cat in json.loads(row[6])],
                planned_venues=json.loads(row[7])
            )
        return None
        
    def get_user_groups(self, user_id: str) -> List[Group]:
        query = 'SELECT * FROM groups WHERE creator_id = ? OR members LIKE ?'
        results = self.execute_query(query, (user_id, f'%{user_id}%'), fetch=True)
        groups = []
        for row in results:
            groups.append(Group(
                group_id=row[0], name=row[1], description=row[2],
                members=json.loads(row[3]), creator_id=row[4],
                created_at=datetime.fromisoformat(row[5]),
                venue_preferences=[VenueCategory(cat) for cat in json.loads(row[6])],
                planned_venues=json.loads(row[7])
            ))
        return groups
        
    def insert_message(self, message: Message):
        query = '''INSERT OR REPLACE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)'''
        params = (
            message.message_id, message.sender_id, message.receiver_id,
            message.group_id, message.content, message.timestamp.isoformat(),
            message.read
        )
        self.execute_query(query, params)
        
    def get_messages(self, user_id: str, other_user_id: Optional[str] = None, 
                    group_id: Optional[str] = None, limit: int = 50) -> List[Message]:
        if other_user_id:
            query = '''SELECT * FROM messages WHERE 
                      ((sender_id = ? AND receiver_id = ?) OR 
                       (sender_id = ? AND receiver_id = ?))
                      ORDER BY timestamp DESC LIMIT ?'''
            results = self.execute_query(query, (user_id, other_user_id, other_user_id, user_id, limit), fetch=True)
        elif group_id:
            query = '''SELECT * FROM messages WHERE group_id = ? 
                      ORDER BY timestamp DESC LIMIT ?'''
            results = self.execute_query(query, (group_id, limit), fetch=True)
        else:
            return []
            
        messages = []
        for row in results:
            messages.append(Message(
                message_id=row[0], sender_id=row[1], receiver_id=row[2],
                group_id=row[3], content=row[4],
                timestamp=datetime.fromisoformat(row[5]),
                read=row[6]
            ))
        return messages

# ========================= DATASET GENERATOR =========================
class DatasetGenerator:
    def __init__(self):
        self.user_locations = [
            (40.7128, -74.0060),  # NYC
            (40.7580, -73.9855),  # Times Square
            (40.7489, -73.9680),  # Grand Central
            (40.7549, -73.9840),  # Central Park
            (40.7282, -73.7949),  # Flushing
        ]
        
    def generate_users(self, count: int = 20) -> List[User]:
        users = []
        first_names = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Avery", "Quinn", "Sam", "Blake"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        interests_pool = ["Italian", "Hiking", "Rooftop Bars", "Coffee", "Art", "Music", "Nightlife", "Beach", "Fitness", "Movies"]
        
        for i in range(count):
            user_id = f"user_{i:03d}"
            username = f"{random.choice(first_names)}_{random.choice(last_names)}{i}".lower()
            email = f"{username}@example.com"
            location = random.choice(self.user_locations)
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                created_at=datetime.now() - timedelta(days=random.randint(1, 365)),
                location=location,
                bio=f"Exploring NYC and discovering new places!",
                avatar_url=f"https://i.pravatar.cc/150?img={i}",
                verified=random.random() > 0.7,
                interests=random.sample(interests_pool, random.randint(2, 5))
            )
            users.append(user)
        return users
        
    def generate_venues(self) -> List[Venue]:
        venues_data = [
            # Cafés
            ("ven_001", "Brew Haven", VenueCategory.CAFE, (40.7580, -73.9855), "Cozy café with excellent espresso", 4.8, "https://images.unsplash.com/photo-1495521821757-a1efb6729352?w=400", "123 5th Ave", "(212) 555-0101", "7AM-10PM", 50),
            ("ven_002", "The Daily Grind", VenueCategory.CAFE, (40.7489, -73.9680), "Modern café in Grand Central area", 4.6, "https://images.unsplash.com/photo-1511920170033-f8396924c348?w=400", "456 E 42nd St", "(212) 555-0102", "6AM-9PM", 40),
            ("ven_003", "Caffeine Dreams", VenueCategory.CAFE, (40.7549, -73.9840), "Trendy café near Central Park", 4.7, "https://images.unsplash.com/photo-1514432324607-2e1907c20b15?w=400", "789 Central Park W", "(212) 555-0103", "7AM-11PM", 60),
            
            # Restaurants
            ("ven_004", "Pasta Perfetto", VenueCategory.RESTAURANT, (40.7128, -74.0060), "Authentic Italian restaurant", 4.9, "https://images.unsplash.com/photo-1517457373614-b7152f800fd1?w=400", "101 Mulberry St", "(212) 555-0104", "5PM-11PM", 100),
            ("ven_005", "Sushi Paradise", VenueCategory.RESTAURANT, (40.7580, -73.9855), "Premium Japanese cuisine", 4.8, "https://images.unsplash.com/photo-1553639074-98eeb64c6a62?w=400", "202 Park Ave", "(212) 555-0105", "5:30PM-11:30PM", 80),
            ("ven_006", "Spice Route", VenueCategory.RESTAURANT, (40.7489, -73.9680), "Indian cuisine with modern twist", 4.7, "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400", "303 Lexington Ave", "(212) 555-0106", "5PM-10PM", 120),
            
            # Nightlife
            ("ven_007", "Skyline Rooftop", VenueCategory.NIGHTLIFE, (40.7549, -73.9840), "Rooftop bar with Manhattan views", 4.8, "https://images.unsplash.com/photo-1514432324607-2e1907c20b15?w=400", "404 Times Square", "(212) 555-0107", "8PM-4AM", 200),
            ("ven_008", "Electric Nights", VenueCategory.NIGHTLIFE, (40.7128, -74.0060), "EDM club with live DJs", 4.6, "https://images.unsplash.com/photo-1516637090002-7f51b3c8c3b9?w=400", "505 Houston St", "(212) 555-0108", "9PM-5AM", 500),
            ("ven_009", "Jazz Corner", VenueCategory.NIGHTLIFE, (40.7580, -73.9855), "Intimate jazz lounge", 4.7, "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=400", "606 5th Ave", "(212) 555-0109", "7PM-2AM", 150),
            
            # Hiking
            ("ven_010", "Central Park Trails", VenueCategory.HIKING, (40.7829, -73.9654), "Urban hiking trails", 4.9, "https://images.unsplash.com/photo-1551528781-e94e99454814?w=400", "Central Park", "(212) 555-0110", "Sunrise-Sunset", 1000),
            ("ven_011", "Hudson Valley Hikes", VenueCategory.HIKING, (41.0534, -74.1502), "Scenic mountain trails", 4.8, "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400", "Hudson Valley NY", "(845) 555-0111", "Sunrise-Sunset", 500),
            
            # Art Galleries
            ("ven_012", "Modern Art Hub", VenueCategory.ART, (40.7614, -73.9776), "Contemporary art gallery", 4.7, "https://images.unsplash.com/photo-1578926078328-123456789012?w=400", "707 Park Ave S", "(212) 555-0112", "10AM-6PM", 200),
            ("ven_013", "Street Art Canvas", VenueCategory.ART, (40.7196, -74.0022), "Street art and urban culture", 4.6, "https://images.unsplash.com/photo-1577720643272-265ab3809d59?w=400", "808 Bowery", "(212) 555-0113", "11AM-8PM", 150),
            
            # Parks
            ("ven_014", "Washington Park", VenueCategory.PARK, (40.7331, -74.0029), "Green space in the city", 4.8, "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=400", "Washington Park", "(212) 555-0114", "24 Hours", 5000),
            
            # Beaches
            ("ven_015", "Coney Island Beach", VenueCategory.BEACH, (40.5755, -73.9822), "Classic NYC beach", 4.7, "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400", "Coney Island", "(718) 555-0115", "24 Hours", 10000),
        ]
        
        venues = []
        for venue_data in venues_data:
            venue = Venue(
                venue_id=venue_data[0],
                name=venue_data[1],
                category=venue_data[2],
                location=venue_data[3],
                description=venue_data[4],
                rating=venue_data[5],
                image_url=venue_data[6],
                address=venue_data[7],
                phone=venue_data[8],
                hours=venue_data[9],
                capacity=venue_data[10],
                website=f"https://{venue_data[1].lower().replace(' ', '')}.com",
                trending_score=random.uniform(0.5, 1.0)
            )
            venues.append(venue)
        return venues
        
    def generate_interactions(self, users: List[User], venues: List[Venue], count: int = 200) -> List[Interaction]:
        interactions = []
        for i in range(count):
            interaction_id = f"inter_{i:04d}"
            user = random.choice(users)
            venue = random.choice(venues)
            interaction_type = random.choice(list(InteractionType))
            duration = random.randint(10, 600) if interaction_type == InteractionType.VIEW else 0
            
            interaction = Interaction(
                interaction_id=interaction_id,
                user_id=user.user_id,
                venue_id=venue.venue_id,
                interaction_type=interaction_type,
                duration_seconds=duration,
                timestamp=datetime.now() - timedelta(days=random.randint(0, 30))
            )
            interactions.append(interaction)
        return interactions
        
    def generate_posts(self, users: List[User], venues: List[Venue], count: int = 100) -> List[Post]:
        posts = []
        captions = [
            "Just discovered this amazing place! #Luna",
            "Best night out with friends! 🎉",
            "Must try when visiting NYC",
            "Absolutely love this venue!",
            "Worth every penny, highly recommend",
            "Found my new favorite spot",
            "The vibe here is unmatched",
            "Can't wait to come back"
        ]
        
        for i in range(count):
            post_id = f"post_{i:04d}"
            user = random.choice(users)
            venue = random.choice(venues) if random.random() > 0.3 else None
            
            post = Post(
                post_id=post_id,
                user_id=user.user_id,
                venue_id=venue.venue_id if venue else None,
                content=random.choice(captions),
                image_url=venue.image_url if venue else "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=400",
                likes=random.randint(0, 500),
                comments_count=random.randint(0, 100),
                created_at=datetime.now() - timedelta(days=random.randint(0, 30)),
                tagged_users=random.sample([u.user_id for u in users], k=random.randint(0, 3))
            )
            posts.append(post)
        return posts
        
    def generate_comments(self, posts: List[Post], users: List[User], count: int = 150) -> List[Comment]:
        comments = []
        comment_texts = [
            "Looks amazing!",
            "Definitely trying this out!",
            "So jealous! 😍",
            "Already booked my spot!",
            "10/10 would recommend",
            "My favorite too!",
            "Adding to my list!",
            "Just went there yesterday, loved it!"
        ]
        
        for i in range(count):
            comment_id = f"comment_{i:04d}"
            post = random.choice(posts)
            user = random.choice(users)
            
            comment = Comment(
                comment_id=comment_id,
                post_id=post.post_id,
                user_id=user.user_id,
                content=random.choice(comment_texts),
                likes=random.randint(0, 50),
                created_at=datetime.now() - timedelta(days=random.randint(0, 30))
            )
            comments.append(comment)
        return comments

# ========================= RECOMMENDATION ENGINE =========================
class RecommendationEngine:
    def __init__(self, db: LunaDatabase):
        self.db = db
        self.scaler = StandardScaler()
        self.venue_embeddings = {}
        self.user_profiles = {}

    def train_ranker_from_interactions(self, min_samples: int = 50):
        """
        Train a RandomForestRegressor to predict 'preference score' for (user, venue)
        based on real interaction logs:

        - LIKE / SAVE / VISIT  -> positive signal
        - VIEW with long duration        -> positive
        - VIEW with very short duration  -> negative
        """
        X, y = [], []

        user_ids = self.db.get_all_user_ids()
        if not user_ids:
            return

        for uid in user_ids:
            profile = self.build_user_profile(uid)
            interactions = self.db.get_user_interactions(uid)

            # Aggregate strongest signal per (user,venue)
            venue_label = {}  # (venue_id -> best_label_so_far)
            for inter in interactions:
                label = None
                if inter.interaction_type in (
                    InteractionType.LIKE,
                    InteractionType.SAVE,
                    InteractionType.VISIT,
                ):
                    label = 1.0
                elif inter.interaction_type == InteractionType.VIEW:
                    if inter.duration_seconds <= 15:
                        label = 0.0
                    elif inter.duration_seconds >= 60:
                        label = 1.0

                if label is None:
                    continue

                prev = venue_label.get(inter.venue_id, 0.0)
                # Keep the strongest positive signal
                venue_label[inter.venue_id] = max(prev, label)

            for vid, label in venue_label.items():
                venue = self.db.get_venue(vid)
                if not venue:
                    continue
                feats = self.get_venue_features(venue, profile)
                X.append(feats)
                y.append(label)

        if len(X) < min_samples:
            print(f"⚠️ Not enough samples to train ranker (got {len(X)}).")
            return

        X = np.asarray(X)
        y = np.asarray(y)

        # Scale + fit model
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=6,
            random_state=42,
        )
        rf.fit(X_scaled, y)

        self.rank_model = rf
        self.training_samples = len(X)
        print(f"✅ Trained RandomForest ranker on {len(X)} samples.")
        
    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two coordinates"""
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        R = 6371  # Earth's radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
        
    def build_user_profile(self, user_id: str) -> Dict:
        """Build comprehensive user behavior profile"""
        user = self.db.get_user(user_id)
        interactions = self.db.get_user_interactions(user_id)
        
        # Calculate interaction metrics
        category_engagement = defaultdict(lambda: {"views": 0, "likes": 0, "saves": 0, "visits": 0})
        total_view_time = 0
        
        for interaction in interactions:
            venue = self.db.get_venue(interaction.venue_id)
            if venue:
                category = venue.category.value
                if interaction.interaction_type == InteractionType.VIEW:
                    category_engagement[category]["views"] += 1
                    total_view_time += interaction.duration_seconds
                elif interaction.interaction_type == InteractionType.LIKE:
                    category_engagement[category]["likes"] += 1
                elif interaction.interaction_type == InteractionType.SAVE:
                    category_engagement[category]["saves"] += 1
                elif interaction.interaction_type == InteractionType.VISIT:
                    category_engagement[category]["visits"] += 1
        
        # Normalize engagement scores
        category_scores = {}
        for category, metrics in category_engagement.items():
            score = (metrics["views"] * 0.3 + metrics["likes"] * 1.0 + 
                    metrics["saves"] * 1.5 + metrics["visits"] * 2.0)
            category_scores[category] = score
            
        return {
            "user_id": user_id,
            "location": user.location,
            "interests": user.interests,
            "category_scores": category_scores,
            "total_engagement": len(interactions),
            "avg_view_time": total_view_time / max(len([i for i in interactions if i.interaction_type == InteractionType.VIEW]), 1)
        }
        
    def get_venue_features(self, venue: Venue, user_profile: Dict) -> np.ndarray:
        """Extract features for a venue"""
        distance = self.calculate_distance(user_profile["location"], venue.location)
        distance_score = max(0, 1 - distance / 10)  # Normalize by 10km
        
        category_match = 1.0 if venue.category.value in user_profile["category_scores"] else 0.5
        category_engagement = user_profile["category_scores"].get(venue.category.value, 0.1)
        
        features = np.array([
            venue.rating / 5.0,  # Rating (normalized)
            distance_score,  # Distance score
            venue.trending_score,  # Trending score
            category_match * category_engagement,  # Category engagement
            venue.capacity / 1000  # Capacity (normalized)
        ])
        return features
        
    def recommend_venues(self, user_id: str, limit: int = 10, show_reasoning: bool = False) -> Tuple[List[Tuple[Venue, float]], Dict]:
        """Get personalized venue recommendations with reasoning"""
        user_profile = self.build_user_profile(user_id)
        all_venues = self.db.get_all_venues()
        
        scored_venues = []
        reasoning_details = []
        
        for venue in all_venues:
            # Skip if user has already visited (booked) this venue
            if venue.venue_id in [booking.venue_id for booking in self.db.get_user_bookings(user_id)]:
                pass  # Still allow in recs for demo; remove this 'pass' to filter
        
            features = self.get_venue_features(venue, user_profile)
            base_score = np.sum(features * np.array([0.25, 0.25, 0.2, 0.25, 0.05]))
            
            # Boost for matching interests
            interest_boost = 0
            for interest in user_profile["interests"]:
                if interest.lower() in venue.name.lower() or interest.lower() in venue.description.lower():
                    interest_boost += 0.15
                    
            # Boost for category preference
            category_boost = user_profile["category_scores"].get(venue.category.value, 0) * 0.1
            
            final_score = min(1.0, base_score + interest_boost + category_boost)
            scored_venues.append((venue, final_score, interest_boost, category_boost))
            
            reasoning_details.append({
                "venue": venue.name,
                "rating_component": features[0],
                "distance_component": features[1],
                "trending_component": features[2],
                "category_component": features[3],
                "interest_boost": interest_boost,
                "category_boost": category_boost,
                "final_score": final_score
            })
        
        # Sort by score and return top venues
        scored_venues.sort(key=lambda x: x[1], reverse=True)
        recommendations = [(v, s) for v, s, _, _ in scored_venues[:limit]]
        
        reasoning_details.sort(key=lambda x: x["final_score"], reverse=True)
        
        return recommendations, {"algorithm": "Hybrid Recommendation Engine", "details": reasoning_details[:limit]}
        
    def recommend_users(self, user_id: str, limit: int = 5) -> List[Tuple[User, float]]:
        """Recommend compatible users based on interests and behavior"""
        target_user = self.db.get_user(user_id)
        target_profile = self.build_user_profile(user_id)
        
        # Get all other users
        all_users = []
        i = 0
        while True:
            user = self.db.get_user(f"user_{i:03d}")
            if not user:
                break
            if user.user_id != user_id and user.user_id not in target_user.blocked_users:
                all_users.append(user)
            i += 1
        
        scored_users = []
        for other_user in all_users:
            other_profile = self.build_user_profile(other_user.user_id)
            
            # Interest overlap
            interest_overlap = len(set(target_profile["interests"]) & set(other_profile["interests"])) / max(len(set(target_profile["interests"]) | set(other_profile["interests"])), 1)
            
            # Category preference similarity
            target_categories = set(target_profile["category_scores"].keys())
            other_categories = set(other_profile["category_scores"].keys())
            category_overlap = len(target_categories & other_categories) / max(len(target_categories | other_categories), 1)
            
            # Location proximity
            distance = self.calculate_distance(target_profile["location"], other_profile["location"])
            location_score = max(0, 1 - distance / 20)
            
            # Combined compatibility score
            compatibility_score = (interest_overlap * 0.4 + category_overlap * 0.35 + location_score * 0.25)
            scored_users.append((other_user, compatibility_score))
        
        scored_users.sort(key=lambda x: x[1], reverse=True)
        return scored_users[:limit]
        
    def recommend_groups(self, user_id: str, limit: int = 5) -> List[Tuple[Group, float]]:
        """Recommend groups based on user interests"""
        user_profile = self.build_user_profile(user_id)
        
        all_groups = self.db.get_user_groups(user_id)
        scored_groups = []
        
        for group in all_groups:
            if user_id in group.members:
                continue
                
            # Interest overlap with group preferences
            if group.venue_preferences:
                preference_overlap = len(set(user_profile["category_scores"].keys()) & 
                                       {cat.value for cat in group.venue_preferences}) / len(group.venue_preferences)
            else:
                preference_overlap = 0.5
                
            # Group size influence
            size_score = min(1.0, len(group.members) / 10)
            
            score = preference_overlap * 0.6 + size_score * 0.4
            scored_groups.append((group, score))
        
        scored_groups.sort(key=lambda x: x[1], reverse=True)
        return scored_groups[:limit]

# ========================= AI AGENT =========================



# ========================= BOOKING & QR CODE =========================
class BookingManager:
    @staticmethod
    def create_qr_code(booking: Booking, user: User, venue: Venue) -> str:
        """Generate QR code for booking"""
        booking_info = f"LUNA_BOOKING|{booking.booking_id}|{user.user_id}|{venue.venue_id}|{booking.booking_date.isoformat()}|{booking.party_size}"
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(booking_info)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    @staticmethod
    def generate_booking_id() -> str:
        """Generate unique booking ID"""
        return f"BOOK_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"

# ========================= SYSTEM INITIALIZATION =========================
class LunaSystem:
    def __init__(self, gemini_api_key: str = ""):
        self.db = LunaDatabase()
        self.rec_engine = RecommendationEngine(self.db)
        self.ai_agent = LunaAIAgent(gemini_api_key, self.db, self.rec_engine)
        self.booking_manager = BookingManager()
        self.generator = DatasetGenerator()
        self.initialized = False
        
    def initialize_with_demo_data(self):
        """Populate database with demo data"""
        if self.initialized:
            return
            
        print("🚀 Initializing Luna Social with demo data...")
        
        # Generate users
        users = self.generator.generate_users(20)
        for user in users:
            self.db.insert_user(user)
        print(f"✅ Created {len(users)} users")
        
        # Generate venues
        venues = self.generator.generate_venues()
        for venue in venues:
            self.db.insert_venue(venue)
        print(f"✅ Created {len(venues)} venues")
        
        # Generate interactions
        interactions = self.generator.generate_interactions(users, venues, 200)
        for interaction in interactions:
            self.db.insert_interaction(interaction)
        print(f"✅ Created {len(interactions)} interactions")
        
        # Generate posts
        posts = self.generator.generate_posts(users, venues, 100)
        for post in posts:
            self.db.insert_post(post)
        print(f"✅ Created {len(posts)} posts")
        
        # Generate comments
        comments = self.generator.generate_comments(posts, users, 150)
        for comment in comments:
            self.db.insert_comment(comment)
        print(f"✅ Created {len(comments)} comments")
        
        # Create some groups
        group_names = ["NYC Foodies", "Hiking Enthusiasts", "Nightlife Crew", "Art Lovers"]
        for i, group_name in enumerate(group_names):
            group = Group(
                group_id=f"group_{i:02d}",
                name=group_name,
                description=f"A group of {group_name.lower()}",
                members=[users[j].user_id for j in range(random.randint(2, 5))],
                creator_id=users[0].user_id,
                created_at=datetime.now(),
                venue_preferences=random.sample(list(VenueCategory), random.randint(1, 3))
            )
            self.db.insert_group(group)
        print(f"✅ Created {len(group_names)} groups")

        # === Demo enrichment for hero user (user_000) ===
        hero_user = users[0]  # user_000
        hero_venues = venues[:5]  # first few venues as favorites

        # Make hero user member of all groups
        all_groups = [self.db.get_group(f"group_{i:02d}") for i in range(len(group_names))]
        for g in all_groups:
            if g and hero_user.user_id not in g.members:
                g.members.append(hero_user.user_id)
                self.db.insert_group(g)

        # Create a couple of bookings for hero user
        for v in hero_venues[:3]:
            booking = Booking(
                booking_id=BookingManager.generate_booking_id(),
                user_id=hero_user.user_id,
                venue_id=v.venue_id,
                booking_date=datetime.now() + timedelta(days=random.randint(1, 10)),
                party_size=random.randint(2, 5),
                special_notes="Demo auto-generated booking",
                status="confirmed",
                created_at=datetime.now()
            )
            booking.qr_code = BookingManager.create_qr_code(booking, hero_user, v)
            self.db.insert_booking(booking)
            hero_user.bookings.append(booking.booking_id)

        # Save updated hero user
        self.db.insert_user(hero_user)
        
        self.initialized = True
        print("✨ Luna Social initialized successfully!")

# Save this module for use in Streamlit app
if __name__ == "__main__":
    print("\n🚀 Initializing Luna Social Backend...\n")

    # Load Gemini API key if available
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Initialize the full system (DB + Rec Engine + AI Agent)
    system = LunaSystem(gemini_api_key=GEMINI_API_KEY)

    # Seed demo data (users, venues, groups, messages)
    system.initialize_with_demo_data()

    # Train ML ranking model
    try:
        system.rec_engine.train_ranker_from_interactions()
        print("✅ ML Ranker trained successfully.")
    except Exception as e:
        print(f"⚠️ Could not train ML ranker: {e}")

    # Confirm AI agent loaded
    if system.ai_agent and system.ai_agent.model:
        print("🤖 Gemini AI Agent initialized and active.")
    else:
        print("⚠️ AI agent running in fallback mode (no Gemini model).")

    print("\n🎉 Luna Social Backend is fully initialized and ready!\n")
