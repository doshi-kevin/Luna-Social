import json
from datetime import datetime, timedelta
import random
from database import Database
from models import RecommendationEngine

class BookingAgent:
    def __init__(self):
        self.db = Database()
        self.engine = RecommendationEngine()
    
    def auto_book_venue(self, user_id, venue_id, preferred_time=None, party_size=2):
        """
        Autonomous agent to automatically book a venue for user
        Agent considers: user preference, companion compatibility, venue availability
        """
        user = self.db.get_user(user_id)
        venue = self.db.get_venue(venue_id)
        
        if not user or not venue:
            return {'status': 'failed', 'reason': 'User or venue not found'}
        
        # Step 1: Find compatible companions
        companions = self._find_companions(user_id, party_size - 1)
        companion_ids = [c['id'] for c in companions] if companions else []
        
        # Step 2: Determine optimal booking time
        booking_time = preferred_time or self._suggest_optimal_time(user_id, venue_id)
        
        # Step 3: Check venue availability
        is_available = self._check_availability(venue_id, booking_time, party_size)
        
        if not is_available:
            return {'status': 'pending', 'reason': 'Venue at capacity, on waitlist'}
        
        # Step 4: Generate booking
        booking_id = self.db.add_booking(user_id, venue_id, booking_time, party_size, companion_ids)
        
        return {
            'status': 'confirmed',
            'booking_id': booking_id,
            'venue': venue['name'],
            'time': booking_time,
            'party_size': party_size,
            'companions': [{'id': c['id'], 'username': c['username']} for c in companions],
            'confirmation': f"Booking confirmed for {venue['name']} on {booking_time}"
        }
    
    def _find_companions(self, user_id, count=1):
        """Agent finds compatible companions from social network"""
        user = self.db.get_user(user_id)
        user_interests = json.loads(user['interests']) if user['interests'] else []
        
        all_users = self.db.get_all_users()
        potential_companions = []
        
        for other_user in all_users:
            if other_user['id'] == user_id:
                continue
            
            other_interests = json.loads(other_user['interests']) if other_user['interests'] else []
            common_interests = len(set(user_interests) & set(other_interests))
            
            if common_interests > 0:
                potential_companions.append({
                    'id': other_user['id'],
                    'username': other_user['username'],
                    'compatibility': common_interests / len(set(user_interests) | set(other_interests))
                })
        
        # Sort by compatibility and return top N
        potential_companions.sort(key=lambda x: x['compatibility'], reverse=True)
        return potential_companions[:count]
    
    def _suggest_optimal_time(self, user_id, venue_id):
        """Agent suggests optimal booking time based on peak hours"""
        # In production, this would analyze historical booking data
        now = datetime.now()
        
        # Weekend evenings are popular for social venues
        if now.weekday() >= 4:  # Friday or Saturday
            suggested = now.replace(hour=19, minute=0)
        else:  # Weekdays
            suggested = now.replace(hour=18, minute=30)
        
        # Ensure future time
        if suggested < now:
            suggested += timedelta(days=1)
        
        return suggested.isoformat()
    
    def _check_availability(self, venue_id, booking_time, party_size):
        """Agent checks venue availability"""
        venue = self.db.get_venue(venue_id)
        # Mock availability check
        return party_size <= venue['capacity']
    
    def auto_invite_companions(self, booking_id, user_id):
        """
        Agent automatically sends invites to recommended companions
        This creates notifications/messages in real system
        """
        booking = self.db.get_user_bookings(user_id)
        
        # Find booking by ID
        booking_record = None
        for b in booking:
            if b['id'] == booking_id:
                booking_record = b
                break
        
        if not booking_record:
            return {'status': 'failed'}
        
        companion_ids = json.loads(booking_record['companion_ids']) if booking_record['companion_ids'] else []
        venue = self.db.get_venue(booking_record['venue_id'])
        
        invites = []
        for companion_id in companion_ids:
            invites.append({
                'to_user': companion_id,
                'type': 'venue_invite',
                'booking_id': booking_id,
                'venue': venue['name'],
                'time': booking_record['booking_date'],
                'message': f"Join me at {venue['name']}!",
                'status': 'sent'
            })
        
        return {
            'status': 'invites_sent',
            'count': len(invites),
            'invites': invites
        }
    
    def process_intelligent_recommendation(self, user_id):
        """
        Master agent that combines recommendations and books
        This is the autonomous recommendation system
        """
        user = self.db.get_user(user_id)
        if not user:
            return {'status': 'failed'}
        
        # Step 1: Get venue recommendations
        venue_recs = self.engine.recommend_venues(user_id)
        
        if not venue_recs:
            return {'status': 'no_recommendations'}
        
        # Step 2: Select top recommendation
        top_venue = venue_recs[0]
        
        # Step 3: Auto-generate booking
        booking_result = self.auto_book_venue(user_id, top_venue['id'], party_size=2)
        
        # Step 4: Get people recommendations
        people_recs = self.engine.recommend_people(user_id, top_n=3)
        
        # Step 5: Create full recommendation package
        recommendation = {
            'venue': top_venue,
            'booking': booking_result,
            'companions': people_recs,
            'insight': f"We recommend {top_venue['name']} - it matches your interests and has compatible friends nearby!"
        }
        
        return recommendation
    
    def get_booking_recommendations_for_group(self, group_id):
        """
        Agent finds optimal venues for group outings
        Considers all group members' preferences
        """
        # Get group members (would query group_members table)
        # Calculate collective preferences
        # Find venue that satisfies most members
        # Auto-suggest group booking
        
        return {
            'status': 'recommended',
            'venue': 'Recommended venue based on group preferences',
            'estimated_party_size': 5,
            'message': 'We found a venue everyone in the group will love!'
        }


class NotificationAgent:
    def __init__(self):
        self.db = Database()
    
    def generate_smart_notifications(self, user_id):
        """
        Agent generates personalized notifications based on activity
        """
        user = self.db.get_user(user_id)
        
        notifications = []
        
        # Check for compatible friends near user
        user_groups = self.db.get_user_groups(user_id)
        if user_groups:
            notifications.append({
                'type': 'group_activity',
                'title': 'Friends in your groups are active',
                'message': f"Join {len(user_groups)} active communities"
            })
        
        # Recommend based on engagement
        bookings = self.db.get_user_bookings(user_id)
        if len(bookings) > 0:
            notifications.append({
                'type': 'booking_reminder',
                'title': 'Upcoming reservations',
                'message': f"You have {len(bookings)} upcoming bookings"
            })
        
        return {
            'user_id': user_id,
            'notification_count': len(notifications),
            'notifications': notifications
        }


class AnalyticsAgent:
    def __init__(self):
        self.db = Database()
        self.engine = RecommendationEngine()
    
    def generate_user_insights(self, user_id):
        """
        Agent analyzes user behavior and generates insights
        """
        user = self.db.get_user(user_id)
        
        # Engagement stats
        engagement = self.engine.get_engagement_stats(user_id)
        
        # Popular venues for user
        bookings = self.db.get_user_bookings(user_id)
        venue_visits = {}
        for booking in bookings:
            venue = self.db.get_venue(booking['venue_id'])
            if venue['category'] not in venue_visits:
                venue_visits[venue['category']] = 0
            venue_visits[venue['category']] += 1
        
        insights = {
            'user': user['username'],
            'engagement_stats': engagement,
            'favorite_categories': sorted(venue_visits.items(), key=lambda x: x[1], reverse=True),
            'network_size': len(self.db.get_user_connections(user_id)),
            'recommendations': f"Based on your activity, you're very engaged with {list(venue_visits.keys())[0] if venue_visits else 'venues'}"
        }
        
        return insights