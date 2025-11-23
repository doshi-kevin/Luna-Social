import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import math
from database import Database

class RecommendationEngine:
    def __init__(self):
        self.db = Database()
        self.scaler = MinMaxScaler()
    
    # ==================== SPATIAL ANALYSIS ====================
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in kilometers"""
        R = 6371  # Earth's radius in km
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    def get_nearby_venues(self, user_id, radius_km=5):
        """Get venues within user's radius with distance scoring"""
        user = self.db.get_user(user_id)
        if not user:
            return []
        
        venues = self.db.get_all_venues()
        nearby = []
        
        for venue in venues:
            dist = self.haversine_distance(user['latitude'], user['longitude'], 
                                          venue['latitude'], venue['longitude'])
            if dist <= radius_km:
                nearby.append({
                    'id': venue['id'],
                    'name': venue['name'],
                    'category': venue['category'],
                    'rating': venue['rating'],
                    'distance': round(dist, 2),
                    'distance_score': max(0, 1 - (dist / radius_km))  # Inverse distance weighting
                })
        
        return sorted(nearby, key=lambda x: x['distance_score'], reverse=True)
    
    # ==================== CONTENT-BASED FILTERING ====================
    def get_user_feature_vector(self, user_id):
        """Create user preference vector from interests and filter history"""
        user = self.db.get_user(user_id)
        if not user:
            return None
        
        interests = json.loads(user['interests']) if user['interests'] else []
        filter_history = self.db.get_user_filter_history(user_id)
        
        # Build feature vector from filters
        feature_dict = {}
        for fh in filter_history:
            key = f"{fh['filter_type']}_{fh['filter_value']}"
            feature_dict[key] = fh['interaction_count']
        
        # Add interests
        for interest in interests:
            feature_dict[f"interest_{interest}"] = 5
        
        return feature_dict, interests
    
    def get_venue_feature_vector(self, venue_id):
        """Create venue feature vector from attributes"""
        venue = self.db.get_venue(venue_id)
        if not venue:
            return None
        
        feature_dict = {
            f"category_{venue['category']}": 10,
            f"rating_{int(venue['rating'])}": venue['rating'],
            'capacity': venue['capacity']
        }
        
        return feature_dict
    
    def calculate_content_similarity(self, user_features, venue_features):
        """Calculate similarity between user preferences and venue"""
        all_keys = set(list(user_features.keys()) + list(venue_features.keys()))
        
        user_vec = [user_features.get(k, 0) for k in all_keys]
        venue_vec = [venue_features.get(k, 0) for k in all_keys]
        
        if sum(user_vec) == 0 or sum(venue_vec) == 0:
            return 0.5
        
        similarity = cosine_similarity([user_vec], [venue_vec])[0][0]
        return float(similarity)
    
    # ==================== COLLABORATIVE FILTERING ====================
    def get_engagement_matrix(self):
        """Build user-venue interaction matrix from posts and views"""
        users = self.db.get_all_users()
        venues = self.db.get_all_venues()
        
        user_idx = {u['id']: i for i, u in enumerate(users)}
        venue_idx = {v['id']: i for i, v in enumerate(venues)}
        
        engagement_matrix = np.zeros((len(users), len(venues)))
        
        # Fill with post view time data
        for user in users:
            posts = self.db.get_user_posts(user['id'])
            for post in posts:
                if post['venue_id']:
                    # Each post view increases engagement
                    engagement_matrix[user_idx[user['id']]][venue_idx[post['venue_id']]] += 0.5
        
        return engagement_matrix, user_idx, venue_idx
    
    def collaborative_recommend(self, user_id, n_recommendations=5):
        """Use collaborative filtering to find similar users and their venues"""
        engagement_matrix, user_idx, venue_idx = self.get_engagement_matrix()
        
        if user_id not in user_idx:
            return []
        
        user_row = user_idx[user_id]
        user_vector = engagement_matrix[user_row].reshape(1, -1)
        
        # Calculate similarity with other users
        similarities = cosine_similarity(user_vector, engagement_matrix)[0]
        similar_users = np.argsort(similarities)[::-1][1:6]  # Top 5 similar users (excluding self)
        
        # Get venues liked by similar users but not by current user
        recommendations = {}
        for sim_user_idx in similar_users:
            sim_score = similarities[sim_user_idx]
            venues_engaged = np.where(engagement_matrix[sim_user_idx] > 0)[0]
            
            for v_idx in venues_engaged:
                # Skip if user already engaged
                if engagement_matrix[user_row][v_idx] > 0:
                    continue
                
                # Find venue id
                venue_id = [v_id for v_id, idx in venue_idx.items() if idx == v_idx][0]
                if venue_id not in recommendations:
                    recommendations[venue_id] = 0
                recommendations[venue_id] += sim_score * engagement_matrix[sim_user_idx][v_idx]
        
        # Return top venues
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        result = []
        for venue_id, score in sorted_recs:
            venue = self.db.get_venue(venue_id)
            result.append({
                'id': venue['id'],
                'name': venue['name'],
                'category': venue['category'],
                'rating': venue['rating'],
                'collab_score': round(float(score), 3)
            })
        
        return result
    
    # ==================== HYBRID RECOMMENDATION ====================
    def recommend_venues(self, user_id, radius_km=5, method='hybrid'):
        """Hybrid recommendation combining spatial, content-based, and collaborative"""
        nearby = self.get_nearby_venues(user_id, radius_km)
        
        if not nearby:
            return []
        
        # Get features
        user_features, interests = self.get_user_feature_vector(user_id)
        
        recommendations = []
        for venue in nearby:
            venue_features = self.get_venue_feature_vector(venue['id'])
            
            # Content similarity
            content_score = self.calculate_content_similarity(user_features, venue_features)
            
            # Combine scores
            hybrid_score = (
                venue['distance_score'] * 0.3 +  # 30% spatial
                content_score * 0.5 +              # 50% content similarity
                (venue['rating'] / 5.0) * 0.2     # 20% venue rating
            )
            
            venue['recommendation_score'] = round(hybrid_score, 3)
            venue['reason'] = self._generate_reason(venue['category'], interests)
            recommendations.append(venue)
        
        return sorted(recommendations, key=lambda x: x['recommendation_score'], reverse=True)
    
    def _generate_reason(self, category, interests):
        """Generate human-readable reason for recommendation"""
        if category.lower() in [i.lower() for i in interests]:
            return f"Matches your interest in {category}"
        return f"Popular {category} venue nearby"
    
    # ==================== SOCIAL COMPATIBILITY SCORING ====================
    def calculate_user_similarity(self, user_id_1, user_id_2):
        """Calculate compatibility between two users"""
        user1 = self.db.get_user(user_id_1)
        user2 = self.db.get_user(user_id_2)
        
        if not user1 or not user2:
            return 0
        
        interests1 = json.loads(user1['interests']) if user1['interests'] else []
        interests2 = json.loads(user2['interests']) if user2['interests'] else []
        
        # Interest overlap
        common = len(set(interests1) & set(interests2))
        total = len(set(interests1) | set(interests2))
        interest_score = common / total if total > 0 else 0.5
        
        # Location proximity
        distance = self.haversine_distance(user1['latitude'], user1['longitude'],
                                          user2['latitude'], user2['longitude'])
        location_score = max(0, 1 - (distance / 10))  # Penalize distance
        
        # Combine scores
        compatibility = (interest_score * 0.6 + location_score * 0.4)
        return round(compatibility, 3)
    
    def recommend_people(self, user_id, top_n=5):
        """Recommend compatible people to connect with"""
        user = self.db.get_user(user_id)
        if not user:
            return []
        
        all_users = self.db.get_all_users()
        compatible = []
        
        for other_user in all_users:
            if other_user['id'] == user_id:
                continue
            
            compatibility = self.calculate_user_similarity(user_id, other_user['id'])
            
            compatible.append({
                'id': other_user['id'],
                'username': other_user['username'],
                'bio': other_user['bio'],
                'profile_pic': other_user['profile_pic'],
                'interests': json.loads(other_user['interests']) if other_user['interests'] else [],
                'compatibility_score': compatibility
            })
        
        # Save to database
        for person in compatible[:top_n]:
            self.db.add_connection(user_id, person['id'], person['compatibility_score'])
        
        return sorted(compatible, key=lambda x: x['compatibility_score'], reverse=True)[:top_n]
    
    # ==================== TIME-BASED ENGAGEMENT ====================
    def track_engagement(self, post_id, user_id, time_spent_seconds):
        """Track how long user spent viewing a post"""
        self.db.track_post_view(post_id, user_id, time_spent_seconds)
        return {'status': 'tracked', 'time': time_spent_seconds}
    
    def get_engagement_stats(self, user_id):
        """Get user's engagement statistics"""
        posts = self.db.get_user_posts(user_id)
        
        if not posts:
            return {'average_engagement': 0, 'total_posts': 0}
        
        total_time = 0
        view_count = 0
        
        for post in posts:
            # In real scenario, would query post_views table
            total_time += 30  # Mock data
            view_count += 1
        
        return {
            'total_posts': len(posts),
            'average_engagement_seconds': round(total_time / len(posts), 2),
            'total_engagement_seconds': total_time
        }