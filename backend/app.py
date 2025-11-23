from flask import Flask, request, jsonify
from flask_cors import CORS
from database import Database
from models import RecommendationEngine
from agents import BookingAgent, NotificationAgent, AnalyticsAgent
from data_generator import DataGenerator
import json

app = Flask(__name__)
CORS(app)

# Initialize services
db = Database()
engine = RecommendationEngine()
booking_agent = BookingAgent()
notification_agent = NotificationAgent()
analytics_agent = AnalyticsAgent()

# ==================== DATA ENDPOINTS ====================

@app.route('/api/init-data', methods=['POST'])
def init_data():
    """Initialize database with sample data"""
    try:
        # Check if data already exists
        users = db.get_all_users()
        if len(users) > 0:
            return jsonify({
                'status': 'success', 
                'message': 'Sample data already exists!',
                'users_count': len(users)
            })
        
        generator = DataGenerator()
        result = generator.generate_sample_data()
        return jsonify({
            'status': 'success', 
            'data': result, 
            'message': 'Sample data created successfully!'
        })
    except Exception as e:
        print(f"Error initializing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e), 'error': repr(e)}), 400

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    users = db.get_all_users()
    result = []
    for user in users:
        result.append({
            'id': user['id'],
            'username': user['username'],
            'bio': user['bio'],
            'profile_pic': user['profile_pic'],
            'interests': json.loads(user['interests']) if user['interests'] else [],
            'latitude': user['latitude'],
            'longitude': user['longitude']
        })
    return jsonify(result)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user profile"""
    user = db.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user['id'],
        'username': user['username'],
        'bio': user['bio'],
        'profile_pic': user['profile_pic'],
        'interests': json.loads(user['interests']) if user['interests'] else [],
        'latitude': user['latitude'],
        'longitude': user['longitude']
    })

@app.route('/api/venues', methods=['GET'])
def get_venues():
    """Get all venues"""
    venues = db.get_all_venues()
    result = []
    for venue in venues:
        result.append({
            'id': venue['id'],
            'name': venue['name'],
            'category': venue['category'],
            'rating': venue['rating'],
            'latitude': venue['latitude'],
            'longitude': venue['longitude'],
            'description': venue['description'],
            'capacity': venue['capacity'],
            'image_url': venue['image_url']
        })
    return jsonify(result)

# ==================== RECOMMENDATION ENDPOINTS ====================

@app.route('/api/recommend/venues/<int:user_id>', methods=['GET'])
def recommend_venues(user_id):
    """Get personalized venue recommendations"""
    radius = request.args.get('radius', 5, type=float)
    
    recommendations = engine.recommend_venues(user_id, radius)
    
    return jsonify({
        'user_id': user_id,
        'radius_km': radius,
        'recommendations': recommendations
    })

@app.route('/api/recommend/people/<int:user_id>', methods=['GET'])
def recommend_people(user_id):
    """Get compatible people recommendations"""
    top_n = request.args.get('top_n', 5, type=int)
    
    people = engine.recommend_people(user_id, top_n)
    
    return jsonify({
        'user_id': user_id,
        'compatible_people': people
    })

@app.route('/api/nearby-venues/<int:user_id>', methods=['GET'])
def nearby_venues(user_id):
    """Get venues sorted by proximity"""
    radius = request.args.get('radius', 5, type=float)
    
    venues = engine.get_nearby_venues(user_id, radius)
    
    return jsonify({
        'user_id': user_id,
        'nearby_venues': venues
    })

# ==================== ENGAGEMENT TRACKING ====================

@app.route('/api/track-engagement', methods=['POST'])
def track_engagement():
    """Track post engagement (time spent viewing)"""
    data = request.json
    
    result = engine.track_engagement(
        data['post_id'],
        data['user_id'],
        data['time_spent']
    )
    
    return jsonify(result)

@app.route('/api/engagement-stats/<int:user_id>', methods=['GET'])
def get_engagement_stats(user_id):
    """Get user engagement statistics"""
    stats = engine.get_engagement_stats(user_id)
    
    return jsonify(stats)

# ==================== BOOKING & AGENTS ====================

@app.route('/api/book-venue', methods=['POST'])
def book_venue():
    """Manually book a venue"""
    data = request.json
    
    result = booking_agent.auto_book_venue(
        data['user_id'],
        data['venue_id'],
        data.get('preferred_time'),
        data.get('party_size', 2)
    )
    
    return jsonify(result)

@app.route('/api/smart-recommendation/<int:user_id>', methods=['GET'])
def smart_recommendation(user_id):
    """Get complete AI recommendation with auto-booking"""
    recommendation = booking_agent.process_intelligent_recommendation(user_id)
    
    return jsonify(recommendation)

@app.route('/api/bookings/<int:user_id>', methods=['GET'])
def get_bookings(user_id):
    """Get user's bookings"""
    bookings = db.get_user_bookings(user_id)
    
    result = []
    for booking in bookings:
        venue = db.get_venue(booking['venue_id'])
        result.append({
            'id': booking['id'],
            'venue': venue['name'] if venue else 'Unknown',
            'venue_id': booking['venue_id'],
            'booking_date': booking['booking_date'],
            'party_size': booking['party_size'],
            'status': booking['status'],
            'companions': json.loads(booking['companion_ids']) if booking['companion_ids'] else []
        })
    
    return jsonify({
        'user_id': user_id,
        'bookings': result
    })

# ==================== SOCIAL FEATURES ====================

@app.route('/api/groups', methods=['GET'])
def get_groups():
    """Get all groups"""
    groups = db.get_all_groups()
    
    result = []
    for group in groups:
        result.append({
            'id': group['id'],
            'name': group['name'],
            'description': group['description'],
            'interest_tag': group['interest_tag'],
            'member_count': group['member_count']
        })
    
    return jsonify(result)

@app.route('/api/user-groups/<int:user_id>', methods=['GET'])
def get_user_groups(user_id):
    """Get groups user belongs to"""
    groups = db.get_user_groups(user_id)
    
    result = []
    for group in groups:
        result.append({
            'id': group['id'],
            'name': group['name'],
            'interest_tag': group['interest_tag'],
            'member_count': group['member_count']
        })
    
    return jsonify({
        'user_id': user_id,
        'groups': result
    })

@app.route('/api/user-posts/<int:user_id>', methods=['GET'])
def get_user_posts(user_id):
    """Get user's posts"""
    posts = db.get_user_posts(user_id)
    
    result = []
    for post in posts:
        venue = db.get_venue(post['venue_id']) if post['venue_id'] else None
        result.append({
            'id': post['id'],
            'user_id': post['user_id'],
            'venue_name': venue['name'] if venue else 'Check-in',
            'caption': post['caption'],
            'image_url': post['image_url'],
            'created_at': post['created_at']
        })
    
    return jsonify(result)

# ==================== ANALYTICS & INSIGHTS ====================

@app.route('/api/insights/<int:user_id>', methods=['GET'])
def get_insights(user_id):
    """Get analytics and insights for user"""
    insights = analytics_agent.generate_user_insights(user_id)
    
    return jsonify(insights)

@app.route('/api/notifications/<int:user_id>', methods=['GET'])
def get_notifications(user_id):
    """Get personalized notifications"""
    notifications = notification_agent.generate_smart_notifications(user_id)
    
    return jsonify(notifications)

@app.route('/api/compatibility/<int:user_id_1>/<int:user_id_2>', methods=['GET'])
def get_compatibility(user_id_1, user_id_2):
    """Check compatibility between two users"""
    score = engine.calculate_user_similarity(user_id_1, user_id_2)
    
    return jsonify({
        'user_1': user_id_1,
        'user_2': user_id_2,
        'compatibility_score': score
    })

# ==================== FILTER TRACKING ====================

@app.route('/api/track-filter', methods=['POST'])
def track_filter():
    """Track filter interaction for recommendations"""
    data = request.json
    
    db.track_filter_interaction(
        data['user_id'],
        data['filter_type'],
        data['filter_value']
    )
    
    return jsonify({'status': 'tracked'})

@app.route('/api/filter-history/<int:user_id>', methods=['GET'])
def get_filter_history(user_id):
    """Get user's filter interaction history"""
    history = db.get_user_filter_history(user_id)
    
    result = []
    for h in history:
        result.append({
            'filter_type': h['filter_type'],
            'filter_value': h['filter_value'],
            'interaction_count': h['interaction_count']
        })
    
    return jsonify({
        'user_id': user_id,
        'filter_history': result
    })

# ==================== HELPER ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'alive', 'service': 'recommendation-engine'})

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'Social Venue Recommendation Engine',
        'version': '1.0',
        'endpoints': {
            'data': '/api/users, /api/venues',
            'recommendations': '/api/recommend/venues/<user_id>, /api/recommend/people/<user_id>',
            'bookings': '/api/book-venue, /api/bookings/<user_id>',
            'social': '/api/groups, /api/user-groups/<user_id>',
            'analytics': '/api/insights/<user_id>, /api/notifications/<user_id>'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)