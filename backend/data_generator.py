from database import Database
import random
import json

class DataGenerator:
    def __init__(self):
        self.db = Database()
    
    def generate_sample_data(self):
        """Generate complete sample dataset"""
        
        # NYC coordinates bounds (Manhattan area)
        nyc_lat_min, nyc_lat_max = 40.7128 - 0.05, 40.7128 + 0.05
        nyc_lon_min, nyc_lon_max = -74.0060 - 0.05, -74.0060 + 0.05
        
        # === USERS ===
        users_data = [
            ('alice', 40.7580, -73.9855, ['coffee', 'art', 'nightlife'], 'Artist exploring the city'),
            ('bob', 40.7614, -73.9776, ['food', 'fitness', 'nightlife'], 'Foodie and gym enthusiast'),
            ('charlie', 40.7505, -73.9972, ['music', 'art', 'cocktails'], 'Music lover looking for live venues'),
            ('diana', 40.7489, -73.9680, ['yoga', 'wellness', 'vegan'], 'Wellness enthusiast'),
            ('ethan', 40.7549, -73.9840, ['sports', 'beer', 'tech'], 'Tech guy who loves sports bars'),
            ('fiona', 40.7505, -73.9934, ['fashion', 'brunch', 'shopping'], 'Fashion blogger'),
            ('george', 40.7614, -73.9776, ['hiking', 'outdoors', 'photography'], 'Outdoor adventure seeker'),
            ('hannah', 40.7549, -73.9840, ['wine', 'fine dining', 'art'], 'Wine connoisseur'),
        ]
        
        user_ids = {}
        for username, lat, lon, interests, bio in users_data:
            user_id = self.db.add_user(username, lat, lon, interests, bio)
            user_ids[username] = user_id
            print(f"✓ Created user: {username}")
        
        # === VENUES ===
        venues_data = [
            ('Blue Bottle Coffee', 40.7505, -73.9972, 'coffee', 4.7, 'Specialty coffee roastery'),
            ('The Smith', 40.7614, -73.9776, 'restaurant', 4.5, 'Modern American brasserie'),
            ('Mercury Lounge', 40.7214, -73.9900, 'music', 4.6, 'Live music venue'),
            ('Equinox Fitness', 40.7549, -73.9840, 'fitness', 4.4, 'Premium fitness center'),
            ('Balthazar', 40.7200, -73.9976, 'restaurant', 4.8, 'French bistro classic'),
            ('Yoga to the People', 40.7505, -73.9934, 'yoga', 4.3, 'Affordable yoga studio'),
            ('The Back Room', 40.7214, -73.9900, 'bar', 4.4, 'Speakeasy with craft cocktails'),
            ('Whole Foods', 40.7489, -73.9680, 'grocery', 4.2, 'Organic grocery store'),
            ('MoMA PS1', 40.7505, -73.9972, 'art', 4.9, 'Contemporary art museum'),
            ('Bacchanal Wine Bar', 40.7549, -73.9840, 'wine', 4.6, 'Fine wine selection'),
            ('Rockefeller Center', 40.7580, -73.9855, 'landmark', 4.7, 'NYC iconic venue'),
            ('Noise Pop Records', 40.7400, -73.9800, 'music', 4.5, 'Vinyl record store and cafe'),
            ('Gramercy Tavern', 40.7400, -73.9850, 'bar', 4.5, 'Historic tavern'),
            ('SoulCycle', 40.7614, -73.9776, 'fitness', 4.6, 'Indoor cycling studio'),
            ('Sant Ambroeus', 40.7505, -73.9934, 'restaurant', 4.7, 'Italian caffe'),
        ]
        
        venue_ids = {}
        for name, lat, lon, cat, rating, desc in venues_data:
            venue_id = self.db.add_venue(name, lat, lon, cat, rating, desc)
            venue_ids[name] = venue_id
            print(f"✓ Created venue: {name}")
        
        # === GROUPS ===
        groups_data = [
            ('NYC Foodies', 'Share restaurant recommendations', 'food'),
            ('Yoga Enthusiasts', 'Find yoga buddies and studios', 'yoga'),
            ('Music Lovers', 'Discover live music venues', 'music'),
            ('Fitness Freaks', 'Gym buddies and fitness tips', 'fitness'),
            ('Art Enthusiasts', 'Gallery walks and art events', 'art'),
            ('Wine Club', 'Wine tastings and bar recommendations', 'wine'),
        ]
        
        group_ids = {}
        for name, desc, tag in groups_data:
            group_id = self.db.create_group(name, desc, tag)
            group_ids[name] = group_id
            print(f"✓ Created group: {name}")
        
        # === ADD USERS TO GROUPS ===
        group_assignments = {
            'NYC Foodies': ['alice', 'bob', 'fiona', 'hannah'],
            'Yoga Enthusiasts': ['diana', 'fiona', 'hannah'],
            'Music Lovers': ['charlie', 'alice', 'george'],
            'Fitness Freaks': ['bob', 'ethan', 'george'],
            'Art Enthusiasts': ['alice', 'charlie', 'hannah'],
            'Wine Club': ['hannah', 'bob', 'charlie'],
        }
        
        for group_name, members in group_assignments.items():
            for member in members:
                try:
                    self.db.add_group_member(group_ids[group_name], user_ids[member])
                except:
                    pass  # Skip if already exists
        
        print(f"✓ Assigned users to groups")
        
        # === POSTS ===
        posts = [
            (user_ids['alice'], venue_ids['MoMA PS1'], '🎨 Amazing abstract exhibition today!'),
            (user_ids['bob'], venue_ids['The Smith'], '🍔 Best brunch in the city!'),
            (user_ids['charlie'], venue_ids['Mercury Lounge'], '🎸 Live jazz night was fire!'),
            (user_ids['diana'], venue_ids['Yoga to the People'], '🧘‍♀️ Perfect flow session'),
            (user_ids['hannah'], venue_ids['Balthazar'], '🍷 Incredible wine pairing menu'),
            (user_ids['ethan'], venue_ids['Gramercy Tavern'], '🍺 Best beer selection downtown'),
            (user_ids['fiona'], venue_ids['Sant Ambroeus'], '☕ Cappuccino perfection'),
            (user_ids['george'], venue_ids['Rockefeller Center'], '📸 NYC skyline views are insane'),
        ]
        
        for user_id, venue_id, caption in posts:
            post_id = self.db.add_post(user_id, venue_id, caption)
            # Track some views
            random_viewers = random.sample(list(user_ids.values()), k=random.randint(2, 5))
            for viewer_id in random_viewers:
                if viewer_id != user_id:
                    time_spent = random.randint(10, 120)
                    self.db.track_post_view(post_id, viewer_id, time_spent)
        
        print(f"✓ Created posts with engagement tracking")
        
        # === FILTER INTERACTIONS ===
        for username, user_id in list(user_ids.items())[:5]:
            user = self.db.get_user(user_id)
            interests = json.loads(user['interests'])
            
            for interest in interests:
                # Each user interacts with their interests multiple times
                for _ in range(random.randint(2, 5)):
                    try:
                        self.db.track_filter_interaction(user_id, 'category', interest)
                    except:
                        pass  # Skip if error
        
        print(f"✓ Created filter interaction history")
        
        # === BOOKINGS ===
        bookings = [
            (user_ids['bob'], venue_ids['The Smith'], '2024-12-20 19:00', 4, [user_ids['ethan'], user_ids['charlie']]),
            (user_ids['hannah'], venue_ids['Balthazar'], '2024-12-21 20:00', 3, [user_ids['alice']]),
            (user_ids['diana'], venue_ids['Yoga to the People'], '2024-12-22 09:00', 1, None),
            (user_ids['alice'], venue_ids['Mercury Lounge'], '2024-12-23 21:00', 2, [user_ids['charlie']]),
        ]
        
        for user_id, venue_id, date, party_size, companions in bookings:
            try:
                booking_id = self.db.add_booking(user_id, venue_id, date, party_size, companions)
            except:
                pass  # Skip if already exists
        
        print(f"✓ Created bookings")
        
        print("\n" + "="*60)
        print("SAMPLE DATA GENERATION COMPLETE")
        print("="*60)
        print(f"Users created: {len(user_ids)}")
        print(f"Venues created: {len(venue_ids)}")
        print(f"Groups created: {len(group_ids)}")
        print(f"Total posts with engagement: {len(posts)}")
        print(f"Total bookings: {len(bookings)}")
        print("="*60 + "\n")
        
        return {
            'users': user_ids,
            'venues': venue_ids,
            'groups': group_ids
        }

if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_sample_data()