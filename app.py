# app.py — Luna Social (Hybrid, Clean, Explainable Version)

import os
from datetime import datetime, timedelta
from collections import defaultdict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

from dotenv import load_dotenv

from backend import (
    LunaSystem, User, Venue, VenueCategory, Post, Comment,
    Group, Message, Booking, Interaction, InteractionType, BookingManager
)

# =========================================================
# 1. ENV & GLOBAL CONFIG
# =========================================================

load_dotenv()  # Load .env file if present

APP_NAME = os.getenv("APP_NAME", "Luna Social")
APP_DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "AI-powered social discovery and venue recommendation platform"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATABASE_PATH = os.getenv("DATABASE_PATH", "luna_social.db")

# Flags (not strictly necessary but nice for visibility)
ENABLE_AI_AGENT = os.getenv("ENABLE_AI_AGENT", "True") == "True"
ENABLE_BOOKING = os.getenv("ENABLE_BOOKING", "True") == "True"
ENABLE_GROUPS = os.getenv("ENABLE_GROUPS", "True") == "True"
ENABLE_MAPS = os.getenv("ENABLE_MAPS", "True") == "True"
ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "True") == "True"

# =========================================================
# 2. STREAMLIT CONFIG & STYLES
# =========================================================

st.set_page_config(
    page_title=f"🌙 {APP_NAME}",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e, #0f172a 50%, #020617 100%);
        color: #ffffff;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.6);
        margin-bottom: 12px;
    }
    .metric-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.4);
        font-size: 11px;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #ec4899);
        color: white;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
    }
    h1, h2, h3, h4 {
        color: #e5e7eb !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 3. SESSION STATE & SYSTEM INITIALIZATION
# =========================================================

if "luna_system" not in st.session_state:
    # NOTE: backend.LunaSystem internally creates LunaDatabase("luna_social.db")
    # If you want to force DATABASE_PATH, you’d adjust backend; here we just show it.
    system = LunaSystem(GEMINI_API_KEY if ENABLE_AI_AGENT else "")
    system.initialize_with_demo_data()
    st.session_state.luna_system = system
    st.session_state.current_user_id = "user_000"
    st.session_state.chat_history = []          # For Luna AI chat
    st.session_state.booking_venue = None       # Preselected booking venue
    st.session_state.selected_dm_user = None    # For DM chat
    st.session_state.selected_group = None      # For group chat
else:
    system = st.session_state.luna_system

# Convenience handles
db = system.db
rec_engine = system.rec_engine
ai_agent = system.ai_agent
booking_manager = system.booking_manager

current_user: User = db.get_user(st.session_state.current_user_id)

# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def record_interaction(venue_id: str, interaction_type: InteractionType, duration: int = 0):
    """Record behavioral signals used by recommendation engine.
    
    Signals used:
    - VIEW: time spent, feed scroll
    - LIKE / SAVE / CLICK / VISIT: stronger preference signals
    """
    interaction = Interaction(
        interaction_id=f"inter_{datetime.now().timestamp()}_{interaction_type.value}",
        user_id=st.session_state.current_user_id,
        venue_id=venue_id,
        interaction_type=interaction_type,
        duration_seconds=duration,
        timestamp=datetime.now(),
    )
    db.insert_interaction(interaction)


def create_booking(venue_id: str, booking_date: datetime, party_size: int, notes: str = "") -> Booking:
    """Create booking + QR using backend BookingManager (agent-like automation)."""
    venue = db.get_venue(venue_id)
    user = db.get_user(st.session_state.current_user_id)

    booking = Booking(
        booking_id=BookingManager.generate_booking_id(),
        user_id=user.user_id,
        venue_id=venue_id,
        booking_date=booking_date,
        party_size=party_size,
        special_notes=notes,
        status="confirmed",
        created_at=datetime.now(),
    )
    booking.qr_code = BookingManager.create_qr_code(booking, user, venue)
    db.insert_booking(booking)
    user.bookings.append(booking.booking_id)
    db.insert_user(user)
    return booking


def toggle_post_like(post: Post):
    """Increment likes on a post and record a LIKE interaction for its venue."""
    query = "SELECT * FROM posts WHERE post_id = ?"
    result = db.execute_query(query, (post.post_id,), fetch=True)
    if result:
        row = result[0]
        new_likes = row[5] + 1
        update_query = "UPDATE posts SET likes = ? WHERE post_id = ?"
        db.execute_query(update_query, (new_likes, post.post_id))
        if post.venue_id:
            record_interaction(post.venue_id, InteractionType.LIKE)
        st.success("❤️ Liked!")


def add_comment(post: Post, content: str):
    """Add comment on a post (comment counts as engagement & weak signal)."""
    if not content.strip():
        return
    comment = Comment(
        comment_id=f"comment_{datetime.now().timestamp()}",
        post_id=post.post_id,
        user_id=current_user.user_id,
        content=content,
        created_at=datetime.now(),
    )
    db.insert_comment(comment)
    st.success("💬 Comment added!")


def send_message(recipient_id: str = None, group_id: str = None, content: str = ""):
    """Send DM or group message."""
    if not content.strip():
        return
    msg = Message(
        message_id=f"msg_{datetime.now().timestamp()}",
        sender_id=current_user.user_id,
        receiver_id=recipient_id,
        group_id=group_id,
        content=content,
        timestamp=datetime.now(),
        read=False,
    )
    db.insert_message(msg)


def get_all_demo_users():
    """Utility to list demo users user_000, user_001, ..."""
    users = []
    i = 0
    while True:
        u = db.get_user(f"user_{i:03d}")
        if not u:
            break
        users.append(u)
        i += 1
    return users

# =========================================================
# 5. SIDEBAR — NAVIGATION & SYSTEM STATUS
# =========================================================

with st.sidebar:
    st.markdown(f"## 🌙 {APP_NAME}")
    st.caption(APP_DESCRIPTION)

    st.markdown("---")
    st.markdown("### 👤 Active User")

    if current_user:
        st.markdown(f"**@{current_user.username}**")
        st.caption(f"📧 {current_user.email}")
        st.caption(f"📍 Location: {current_user.location[0]:.3f}, {current_user.location[1]:.3f}")
        st.caption(f"✅ Verified: {current_user.verified}")
    else:
        st.error("Active user not found in DB.")

    st.markdown("---")
    st.markdown("### 🔄 Switch Demo User")

    users = get_all_demo_users()
    user_ids = [u.user_id for u in users]
    selected_user_for_switch = st.selectbox(
        "Select user",
        options=user_ids,
        index=user_ids.index(st.session_state.current_user_id),
        format_func=lambda uid: next(u.username for u in users if u.user_id == uid),
    )
    if selected_user_for_switch != st.session_state.current_user_id:
        st.session_state.current_user_id = selected_user_for_switch
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 System Status")

    ai_status = "🟢 Gemini Connected" if (ENABLE_AI_AGENT and GEMINI_API_KEY) else "⚪ LLM Fallback Mode"
    st.caption(ai_status)
    st.caption(f"📂 DB Path: `{DATABASE_PATH}` (configured)")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "🎯 Discover",
            "👥 People",
            "🏘️ Groups",
            "💬 Messages",
            "🎫 Bookings",
            "👤 Profile",
            "📊 Analytics",
            "🤖 Luna AI",
        ],
        label_visibility="collapsed",
    )

# =========================================================
# MAIN TITLE (top of page)
# =========================================================

st.title("🌙 Luna Social")
st.caption(
    "End-to-end AI-powered recommendation system • Spatial analysis • Social compatibility • Agentic bookings"
)
# =========================================================
# 6. PAGE: HOME — FEED & TRENDING
# =========================================================

if page == "🏠 Home":
    col_feed, col_side = st.columns([2.4, 1])

    with col_feed:
        st.subheader("📱 Community Feed")

        posts = db.get_posts(limit=20)
        if not posts:
            st.info("No posts yet; demo data will populate them shortly.")
        else:
            for post in posts:
                author = db.get_user(post.user_id)
                venue = db.get_venue(post.venue_id) if post.venue_id else None

                with st.container():
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    # Header
                    st.markdown(
                        f"**@{author.username if author else 'unknown'}** "
                        f"{'· 📍 ' + venue.name if venue else ''}"
                    )
                    st.caption(post.created_at.strftime("%b %d, %Y %H:%M"))

                    # Content
                    st.write(post.content)

                    # Image
                    if post.image_url:
                        try:
                            st.image(post.image_url, use_container_width=True)
                        except Exception:
                            st.caption("Image unavailable")

                    # Actions
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button(f"❤️ {post.likes}", key=f"like_{post.post_id}"):
                            toggle_post_like(post)
                            st.rerun()
                    with c2:
                        st.caption(f"💬 {post.comments_count}")
                    with c3:
                        if venue:
                            if st.button("🔍 View Venue", key=f"view_ven_{post.post_id}"):
                                record_interaction(venue.venue_id, InteractionType.CLICK)
                                st.session_state.booking_venue = venue.venue_id
                                st.success(f"Selected {venue.name} for booking.")
                    # Comments
                    with st.expander("💬 Comments"):
                        comments = db.get_comments(post.post_id)
                        for c in comments[:5]:
                            cu = db.get_user(c.user_id)
                            st.markdown(f"**@{cu.username if cu else 'user'}**: {c.content}")
                        new_comment = st.text_input(
                            "Add a comment", key=f"comment_input_{post.post_id}"
                        )
                        if new_comment:
                            add_comment(post, new_comment)
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        st.subheader("🔥 Trending Venues")
        venues = db.get_all_venues()
        if not venues:
            st.info("No venues in DB.")
        else:
            for v in sorted(venues, key=lambda vv: vv.trending_score, reverse=True)[:6]:
                with st.container():
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f"**{v.name}**")
                    st.caption(f"{v.category.value} · ⭐ {v.rating}/5")
                    st.caption(f"🔥 Trending Score: {v.trending_score:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 7. PAGE: DISCOVER — RECOMMENDER + AI AGENT (ONE-SHOT)
# =========================================================

elif page == "🎯 Discover":
    st.subheader("🎯 Discover Venues For You")

    # ------------------------ AI Agent one-shot ------------------------
    st.markdown("### 🤖 Ask Luna for a One-Shot Recommendation or Plan")

    col_q, col_btn = st.columns([3, 1])
    with col_q:
        user_query = st.text_input(
            "Describe what you want (romantic dinner, rooftop bar with EDM, quiet study café, etc.)",
            key="discover_query",
        )
    with col_btn:
        ask_clicked = st.button("🚀 Ask Luna")

    if ask_clicked and user_query:
        with st.spinner("Luna is thinking (AI agent + rec engine)..."):
            reply_text, reply_data = ai_agent.chat(current_user.user_id, user_query)

        col_resp, col_meta = st.columns([2.2, 1])
        with col_resp:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌟 Luna’s Response")
            st.write(reply_text)
            st.markdown("</div>", unsafe_allow_html=True)

            # Show itinerary if present (this is where automated planning happens)
            if reply_data.get("itinerary"):
                st.markdown("#### 📅 Planned Itinerary (Agentic Planning)")
                for item in reply_data["itinerary"]["itinerary"]:
                    st.markdown(
                        f"- **{item['time']}** — **{item['venue']}** ({item['category']})  \n"
                        f"  📍 {item['address']}"
                    )

                # Demonstrate "agent → bookings" for the track requirement
                if ENABLE_BOOKING:
                    if st.button("✨ Auto-create bookings for this itinerary"):
                        for item in reply_data["itinerary"]["itinerary"]:
                            v = db.get_venue(item["venue_id"])
                            if v:
                                start_hour = int(item["time"].split(":")[0])
                                dt = datetime.now().replace(
                                    hour=start_hour, minute=0, second=0, microsecond=0
                                ) + timedelta(days=1)
                                create_booking(v.venue_id, dt, party_size=2, notes="AI itinerary booking")
                        st.success("Created demo bookings for all itinerary venues!")

        with col_meta:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🧠 Parsed Intent")
            st.write(f"**Action:** {reply_data.get('intent')}")
            st.write(f"**Parameters:** {reply_data.get('parameters')}")
            st.caption(
                "Intent + parameters are derived from your message and used to pick between "
                "recommendation, planning, and booking flows."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Recommendation Engine — Personalized Ranking")

    # ------------------------ Core Recommendation Engine ------------------------
    with st.spinner("Computing personalized recommendations with spatial + behavioral signals..."):
        recommendations, reasoning = rec_engine.recommend_venues(
            current_user.user_id,
            limit=10,
            show_reasoning=True,
        )

    left, right = st.columns([2.2, 1.1])
    with left:
        for i, (venue, score) in enumerate(recommendations, start=1):
            detail = next((d for d in reasoning["details"] if d["venue"] == venue.name), None)
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"#### {i}. {venue.name}  ")
                st.caption(
                    f"{venue.category.value} · ⭐ {venue.rating}/5 · "
                    f"Score: <span class='score-badge'>{score*100:.1f}% match</span>",
                    unsafe_allow_html=True,
                )
                st.caption(venue.address)

                if venue.image_url:
                    try:
                        st.image(venue.image_url, use_container_width=True)
                    except Exception:
                        st.caption("Image unavailable")

                # Quick actions
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("❤️ Save", key=f"save_{venue.venue_id}"):
                        if venue.venue_id not in current_user.saved_venues:
                            current_user.saved_venues.append(venue.venue_id)
                            db.insert_user(current_user)
                        record_interaction(venue.venue_id, InteractionType.SAVE)
                        st.success("Saved to profile!")
                with c2:
                    if st.button("📅 Book", key=f"book_{venue.venue_id}"):
                        st.session_state.booking_venue = venue.venue_id
                        st.success("Go to the Bookings tab to confirm.")
                with c3:
                    if st.button("👣 I visited", key=f"visit_{venue.venue_id}"):
                        record_interaction(venue.venue_id, InteractionType.VISIT)
                        st.success("Marked as visited (strong signal).")

                # Explanation: show how the score was formed
                if detail:
                    with st.expander("🔍 Why was this recommended? (Score Breakdown)"):
                        st.write("The hybrid model combines several components:")
                        st.markdown(
                            f"""
- ⭐ **Rating Component:** `{detail['rating_component']:.3f}`  
- 📍 **Distance (Spatial) Component:** `{detail['distance_component']:.3f}`  
- 🔥 **Trending Component:** `{detail['trending_component']:.3f}`  
- 🎯 **Category Engagement:** `{detail['category_component']:.3f}`  
- 💬 **Interest (Text/Tag) Boost:** `{detail['interest_boost']:.3f}`  
- 🧩 **Category Preference Boost:** `{detail['category_boost']:.3f}`  
- 🧮 **Final Score:** `{detail['final_score']:.3f}`
                            """
                        )
                        st.caption(
                            "These components come from the ML-driven RecommendationEngine in backend.py, "
                            "which uses behavior (views/likes/saves/visits), spatial distance, and trend signals."
                        )

                st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Model Overview")

            algo = reasoning.get("algorithm", "Hybrid Engine")
            training_samples = reasoning.get("training_samples", 0)

            st.markdown(
                f"""
    **Engine:** `{algo}`  

    - ⭐ Venue rating (feature 1)  
    - 📍 Distance / spatial proximity (feature 2)  
    - 🔥 Trending score (feature 3)  
    - 🎯 Category engagement from behavior (feature 4)  
    - 🪑 Capacity / crowd factor (feature 5)  

    Behavioral signals used:  
    `VIEW` duration, `LIKE`, `SAVE`, `VISIT` interactions.

    {"**ML ranker:** RandomForest trained on "
    f"{training_samples} interaction-derived samples." if training_samples else "_ML ranker not trained yet (cold start)._"}
    """
            )
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 8. PAGE: PEOPLE — SOCIAL COMPATIBILITY
# =========================================================

elif page == "👥 People":
    st.subheader("👥 People You Might Enjoy Going Out With")

    with st.spinner("Computing compatibility scores (interests + categories + distance)..."):
        rec_users = rec_engine.recommend_users(current_user.user_id, limit=10)

    if not rec_users:
        st.info("Not enough data yet to recommend compatible people.")
    else:
        for u, comp_score in rec_users:
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    st.markdown(f"#### @{u.username}")
                    st.caption(f"Compatibility: **{comp_score*100:.1f}%**")
                    st.caption(f"Interests: {', '.join(u.interests[:4])}")
                    st.caption(f"Location: {u.location[0]:.3f}, {u.location[1]:.3f}")
                    st.caption(u.bio)
                with cols[1]:
                    try:
                        if u.avatar_url:
                            st.image(u.avatar_url, use_container_width=True)
                    except Exception:
                        st.write("👤")

                # Actions
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💬 Message", key=f"msg_{u.user_id}"):
                        st.session_state.selected_dm_user = u.user_id
                        # Jump to Messages page
                        st.experimental_set_query_params(page="messages")
                        st.rerun()
                with c2:
                    st.button("👍 Wave", key=f"wave_{u.user_id}")

                with st.expander("🔍 How compatibility is computed"):
                    st.markdown(
                        """
The **social compatibility** score comes from `RecommendationEngine.recommend_users` and includes:

- 🎵 **Interest Overlap** (40%) — shared topics like Italian, hiking, nightlife  
- 🎯 **Category Preference Similarity** (35%) — categories you both engage with (cafés, hiking, nightlife)  
- 📍 **Location Proximity** (25%) — closer users are easier to coordinate with  

These components are combined into a single compatibility score.
                        """
                    )
                st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 9. PAGE: GROUPS — SOCIAL PLANNING & SIMPLE RSVP
# =========================================================

elif page == "🏘️ Groups":
    st.subheader("🏘️ Social Groups & Outings")

    tab_my, tab_new = st.tabs(["👀 My & Suggested Groups", "🆕 Create Group"])

    with tab_my:
        groups = db.get_user_groups(current_user.user_id)
        if not groups:
            st.info("You are not part of any groups yet.")
        else:
            for g in groups:
                with st.container():
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f"#### {g.name}")
                    st.caption(g.description)
                    st.caption(f"Members: {len(g.members)}")

                    if g.venue_preferences:
                        st.caption(
                            "Venue Preferences: " + ", ".join([cat.value for cat in g.venue_preferences])
                        )

                    if g.planned_venues:
                        venue_id, when_str = g.planned_venues[-1]
                        v = db.get_venue(venue_id)
                        when_dt = datetime.fromisoformat(when_str)
                        st.markdown(
                            f"📅 **Next Plan:** {v.name if v else venue_id} — "
                            f"{when_dt.strftime('%b %d, %I:%M %p')}"
                        )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("💬 Open Group Chat", key=f"chat_{g.group_id}"):
                            st.session_state.selected_group = g.group_id
                            st.rerun()
                    with c2:
                        if st.button("📅 Plan New Outing", key=f"plan_{g.group_id}"):
                            # Simple demo: pick top recommended venue and schedule tomorrow at 8PM
                            recs, _ = rec_engine.recommend_venues(current_user.user_id, limit=1)
                            if recs:
                                v, s = recs[0]
                                dt = datetime.now().replace(
                                    hour=20, minute=0, second=0, microsecond=0
                                ) + timedelta(days=1)
                                g.planned_venues.append((v.venue_id, dt.isoformat()))
                                db.insert_group(g)
                                st.success(
                                    f"Planned {v.name} for {dt.strftime('%b %d, %I:%M %p')} (simple RSVP demo)."
                                )
                    with c3:
                        st.caption("RSVP logic can be extended with accept/decline states.")

                    st.markdown("</div>", unsafe_allow_html=True)

    with tab_new:
        st.markdown("### 🆕 Create a New Group")
        name = st.text_input("Group name")
        desc = st.text_area("Description")
        cats = st.multiselect(
            "Preferred venue categories",
            [c.value for c in VenueCategory],
        )
        if st.button("✨ Create Group"):
            if not name.strip():
                st.error("Please give the group a name.")
            else:
                group = Group(
                    group_id=f"group_{datetime.now().timestamp()}",
                    name=name,
                    description=desc,
                    members=[current_user.user_id],
                    creator_id=current_user.user_id,
                    created_at=datetime.now(),
                    venue_preferences=[VenueCategory(c) for c in cats],
                )
                db.insert_group(group)
                st.success("Group created!")
                st.rerun()
# =========================================================
# 10. PAGE: MESSAGES — DM & GROUP CHAT
# =========================================================

elif page == "💬 Messages":
    st.subheader("💬 Messages & Conversations")

    col_list, col_chat = st.columns([1.2, 2])

    # --------------------------
    # LEFT: DM + Group list
    # --------------------------
    with col_list:
        st.markdown("### 👥 Direct Messages")
        all_users = [u for u in get_all_demo_users() if u.user_id != current_user.user_id]
        for u in all_users[:15]:
            if st.button(f"@{u.username}", key=f"dm_btn_{u.user_id}", use_container_width=True):
                st.session_state.selected_dm_user = u.user_id
                st.session_state.selected_group = None
                st.rerun()

        st.markdown("---")
        st.markdown("### 🏘️ Groups")
        user_groups = db.get_user_groups(current_user.user_id)
        if not user_groups:
            st.caption("You are not in any groups yet.")
        else:
            for g in user_groups:
                label = f"{g.name} ({len(g.members)} members)"
                if st.button(label, key=f"group_btn_{g.group_id}", use_container_width=True):
                    st.session_state.selected_group = g.group_id
                    st.session_state.selected_dm_user = None
                    st.rerun()

    # --------------------------
    # RIGHT: Chat window
    # --------------------------
    with col_chat:
        # DM chat
        if st.session_state.selected_dm_user:
            other = db.get_user(st.session_state.selected_dm_user)
            st.markdown(f"### 💬 Chat with @{other.username}")

            msgs = db.get_messages(
                current_user.user_id,
                other_user_id=other.user_id,
                group_id=None,
                limit=50,
            )

            for msg in reversed(msgs):
                sender = db.get_user(msg.sender_id)
                with st.chat_message(
                    "user" if msg.sender_id == current_user.user_id else "assistant"
                ):
                    st.write(f"@{sender.username if sender else 'user'}: {msg.content}")
                    st.caption(msg.timestamp.strftime("%b %d, %H:%M"))

            new_msg = st.chat_input("Send a message...")
            if new_msg:
                send_message(recipient_id=other.user_id, content=new_msg)
                st.rerun()

        # Group chat
        elif st.session_state.selected_group:
            g = db.get_group(st.session_state.selected_group)
            st.markdown(f"### 🏘️ Group: {g.name}")

            msgs = db.get_messages(
                current_user.user_id,
                other_user_id=None,
                group_id=g.group_id,
                limit=50,
            )

            # Show messages
            for msg in reversed(msgs):
                sender = db.get_user(msg.sender_id)
                with st.chat_message(
                    "user" if msg.sender_id == current_user.user_id else "assistant"
                ):
                    st.write(f"@{sender.username if sender else 'user'}: {msg.content}")
                    st.caption(msg.timestamp.strftime("%b %d, %H:%M"))

            # Send new group message
            new_msg = st.chat_input("Send a message to the group...")
            if new_msg:
                send_message(group_id=g.group_id, content=new_msg)
                st.rerun()

            # --------- AI ANALYSIS BUTTONS ---------

            if msgs:
                if st.button("🤖 Analyze this group's chat & suggest places", key="analyze_real_group"):
                    history_texts = [m.content for m in msgs]
                    result = system.ai_agent.analyze_group_chat(history_texts, current_user.user_id)
                    st.markdown("### 🤖 AI Suggestion (from real chat)")
                    st.markdown(result["formatted"])

            # Extra: hardcoded demo foodie chat (button-triggered)
            if st.button("🍝 Use built-in foodie chat & analyze", key="analyze_demo_group"):
                demo_chat = [
                    "Guys I'm craving proper Italian tonight, not fast food.",
                    "Yeah, let's do pasta and maybe some wood-fired pizza.",
                    "Also not too loud, I want to actually talk.",
                    "Outdoor seating would be amazing if the weather is nice.",
                    "Budget friendly would be good, but I'm okay paying more if it's really good.",
                ]
                result = system.ai_agent.analyze_group_chat(demo_chat, current_user.user_id)
                st.markdown("### 🤖 AI Suggestion (demo foodie chat)")
                st.markdown(result["formatted"])

        else:
            st.info("Select a DM or group from the left to start chatting.")

# =========================================================
# 11. PAGE: BOOKINGS — AUTOMATED RESERVATIONS (AGENT ACTION)
# =========================================================

elif page == "🎫 Bookings":
    st.subheader("🎫 Your Bookings")

    bookings = db.get_user_bookings(current_user.user_id)
    if bookings:
        for b in bookings:
            v = db.get_venue(b.venue_id)
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"#### {v.name if v else b.venue_id}")
                st.caption(
                    f"📅 {b.booking_date.strftime('%b %d, %Y %I:%M %p')} · "
                    f"👥 {b.party_size} · Status: **{b.status.upper()}**"
                )
                if b.special_notes:
                    st.caption(f"📝 {b.special_notes}")
                if b.qr_code:
                    st.markdown("**🎟️ QR Code (Demo):**", unsafe_allow_html=True)
                    st.markdown(f"<img src='{b.qr_code}' width='140'>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No bookings yet. Use Discover or AI to pick a place, then create a booking here.")

    st.markdown("---")
    st.markdown("### 🆕 New Booking")

    venues = db.get_all_venues()
    venue_ids = [v.venue_id for v in venues]

    # Preselect from Discover's 'Book' button
    default_idx = 0
    if st.session_state.booking_venue and st.session_state.booking_venue in venue_ids:
        default_idx = venue_ids.index(st.session_state.booking_venue)

    col1, col2 = st.columns(2)
    with col1:
        selected_vid = st.selectbox(
            "Select venue",
            venue_ids,
            index=default_idx,
            format_func=lambda vid: next(v.name for v in venues if v.venue_id == vid),
        )
    with col2:
        date = st.date_input("Date", value=datetime.now().date() + timedelta(days=1))
        time = st.time_input("Time", value=datetime.now().time().replace(second=0, microsecond=0))

    c1, c2 = st.columns(2)
    with c1:
        party_size = st.number_input("Party size", min_value=1, max_value=20, value=2)
    with c2:
        notes = st.text_input("Special requests", value="")

    if st.button("✨ Confirm Booking"):
        dt = datetime.combine(date, time)
        b = create_booking(selected_vid, dt, party_size, notes)
        st.success(f"Booking created! ID: {b.booking_id}")
        st.session_state.booking_venue = selected_vid
        st.rerun()

# =========================================================
# 12. PAGE: PROFILE — USER SNAPSHOT
# =========================================================

elif page == "👤 Profile":
    st.subheader(f"👤 Profile — @{current_user.username}")

    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            if current_user.avatar_url:
                st.image(current_user.avatar_url, use_container_width=True)
        except Exception:
            st.write("👤")
    with col2:
        st.markdown(f"**Email:** {current_user.email}")
        st.markdown(f"**Verified:** {'✅ Yes' if current_user.verified else '❌ No'}")
        st.markdown(f"**Joined:** {current_user.created_at.strftime('%b %d, %Y')}")
        st.markdown(f"**Bio:** {current_user.bio}")
        st.markdown(f"**Interests:** {', '.join(current_user.interests)}")

    st.markdown("---")
    st.markdown("### 📌 Saved Venues")

    if current_user.saved_venues:
        for vid in current_user.saved_venues:
            v = db.get_venue(vid)
            if v:
                st.markdown(f"- **{v.name}** — {v.category.value} · ⭐ {v.rating}/5")
    else:
        st.caption("No saved venues yet.")

# =========================================================
# 13. PAGE: ANALYTICS — BEHAVIOR & MODEL INSIGHTS
# =========================================================

elif page == "📊 Analytics":
    st.subheader("📊 Behavioral Analytics & Model Insights")
    st.markdown("This dashboard shows how the recommendation engine understands your behavior, how the ML model ranks venues, and how the AI agent interprets group chats.")

    # -----------------------------------------
    # USER PROFILE
    # -----------------------------------------
    profile = rec_engine.build_user_profile(current_user.user_id)
    interactions = db.get_user_interactions(current_user.user_id)

    st.markdown("### 1️⃣ User Interaction Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Interactions", len(interactions))
    with c2:
        st.metric("Likes", len([i for i in interactions if i.interaction_type == InteractionType.LIKE]))
    with c3:
        st.metric("Views", len([i for i in interactions if i.interaction_type == InteractionType.VIEW]))
    with c4:
        st.metric("Saves", len([i for i in interactions if i.interaction_type == InteractionType.SAVE]))

    st.markdown("---")

    # -----------------------------------------
    # CATEGORY ENGAGEMENT + TIME SPENT
    # -----------------------------------------
    col_cat, col_time = st.columns(2)

    with col_cat:
        st.markdown("### 🎯 Category Engagement (Model Input)")
        cat_scores = profile.get("category_scores", {})
        if cat_scores:
            df_cat = pd.DataFrame(
                {"Category": list(cat_scores.keys()), "Score": list(cat_scores.values())}
            ).sort_values("Score", ascending=False)

            fig = px.bar(df_cat, x="Category", y="Score", title="Category Engagement Score")
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category engagement data yet.")

    with col_time:
        st.markdown("### ⏱️ Time Spent per Category")
        time_data = defaultdict(int)
        for inter in interactions:
            if inter.interaction_type == InteractionType.VIEW:
                v = db.get_venue(inter.venue_id)
                if v:
                    time_data[v.category.value] += inter.duration_seconds

        if time_data:
            df_time = pd.DataFrame({"Category": list(time_data.keys()), "Seconds": list(time_data.values())})
            fig = px.pie(df_time, values="Seconds", names="Category", title="View Time Distribution")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No view-time data yet.")

    st.markdown("---")

    # -----------------------------------------
    # FEATURE IMPORTANCE (ML RANKER)
    # -----------------------------------------
    st.markdown("### 2️⃣ ML Ranker — Feature Importance")
    if hasattr(rec_engine, "rank_model") and rec_engine.rank_model:
        importances = rec_engine.rank_model.feature_importances_
        labels = ["Rating", "Distance", "Trending", "Category Score", "Capacity"]

        df_fi = pd.DataFrame({"Feature": labels, "Importance": importances}).sort_values("Importance", ascending=False)
        fig_fi = px.bar(df_fi, x="Feature", y="Importance", title="RandomForest Feature Importance")
        fig_fi.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("ML Ranker not trained yet — not enough data.")

    st.markdown("---")

    # -----------------------------------------
    # RECOMMENDATION SCORE BREAKDOWN
    # -----------------------------------------
    st.markdown("### 3️⃣ Recommendation Score Breakdown (Top Venues)")

    recs, reasoning = rec_engine.recommend_venues(current_user.user_id, limit=5, show_reasoning=True)

    if recs:
        rows = []
        for d in reasoning["details"]:
            rows.append(
                {
                    "Venue": d["venue"],
                    "Rating": d["rating_component"],
                    "Distance": d["distance_component"],
                    "Trending": d["trending_component"],
                    "Category": d["category_component"],
                    "Final Score": d["final_score"],
                    "ML Score": d.get("ml_score"),
                }
            )
        df = pd.DataFrame(rows)

        fig = go.Figure()
        for comp in ["Rating", "Distance", "Trending", "Category"]:
            fig.add_trace(go.Bar(name=comp, x=df["Venue"], y=df[comp]))

        fig.update_layout(
            barmode="stack",
            title="Stacked Recommendation Components",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recommendations yet.")

    st.markdown("---")

    # -----------------------------------------
    # GROUP CHAT ANALYSIS DEBUG PANEL
    # -----------------------------------------
    st.markdown("### 4️⃣ LLM Group Chat Analysis (Debug View)")

    if "last_group_analysis" in st.session_state:
        a = st.session_state.last_group_analysis
        st.json({
            "Detected Vibe": a["vibe"],
            "Cuisine Preference": a["cuisine"],
            "Budget Preference": a["budget"],
            "Mood Score": a["mood"],
            "Chosen Venues": [v.name for v, _ in a["recommendations"]],
        })
    else:
        st.info("Analyze any group chat in the Messages tab to populate this.")

    st.markdown("---")

    # -----------------------------------------
    # PIPELINE VISUALIZATION
    # -----------------------------------------
    st.markdown("### 5️⃣ Recommendation Engine Architecture Overview")

    st.markdown("""
#### 🧠 End-to-End Pipeline

**1. Behavioral Signal Processing**
- Track view duration → attention score  
- Likes / Saves / Visits → preference label  
- Category engagement computed dynamically  

**2. Venue Feature Vector**
`[rating, distance, trending, category_score, capacity]` normalized per venue  

**3. Hybrid Ranking Model**
- Heuristic weighted score  
- RandomForest ML Ranker  
- Final score = **0.5 ML + 0.5 heuristic**  

**4. LLM Agent Understanding**
- Gemini extracts:  
  - Vibe (romantic, calm, adventure…)  
  - Cuisine preference (Italian, Japanese…)  
  - Budget  
  - Mood score (0–100)  
- These constraints further filter ML suggestions  

**5. Fallback & Robustness**
- If vibe/cuisine filtering empties the list → return top ML-ranked venues  
- Always guarantees minimum 3 results  

**6. Itinerary Generation**
- Auto 6 PM → 8 PM → 10 PM route  
- Based on filtered best venues  

""")

    st.markdown("---")

    st.markdown("### ℹ️ How This Satisfies the Official Problem Statement")
    st.caption(
        """
**Track 2 Backend Requirements Mapping:**  

- ✅ **Recommendation Engine**  
   - Machine Learning hybrid scoring  
   - Uses behavioral interaction signals  
   - Spatial distance + trending + rating + preference vectors  
   - ML Ranker trained on user interaction logs  

- ✅ **Spatial Analysis**  
   - Haversine distance → used in scoring  

- ✅ **Social Compatibility**  
   - Group chat LLM analysis  
   - Group vibe / cuisine / budget factor into final recs  

- ✅ **AI Agents**  
   - LLM Intent Parsing (Gemini 2.5 Flash)  
   - Automated multi-step itinerary generation  
   - Booking simulation + QR codes  

- ✅ **Full System Insight**  
   - Analytics dashboard displays:   
      * User engagement  
      * Feature vectors  
      * Model weights  
      * LLM extraction outputs  
      * Pipeline visualization  
"""
    )




# =========================================================
# 14. PAGE: LUNA AI — FULL CHAT EXPERIENCE
# =========================================================

elif page == "🤖 Luna AI":
    st.subheader("🤖 Chat with Luna (AI Assistant)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # New message
    user_msg = st.chat_input("Ask Luna about venues, plans, people, or bookings...")
    if user_msg:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        # Call AI agent (Gemini if key available, fallback otherwise)
        with st.chat_message("assistant"):
            with st.spinner("Luna is thinking..."):
                reply_text, reply_data = ai_agent.chat(current_user.user_id, user_msg)
                st.write(reply_text)

                # OPTIONAL: show debug info on intent
                with st.expander("🧠 Debug: Parsed Intent & Data"):
                    st.write(reply_data)

        st.session_state.chat_history.append({"role": "assistant", "content": reply_text})

# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:12px; color:#9ca3af;'>"
    "🌙 Luna Social &mdash; AI Agents · Spatial Recs · Social Matching · Bookings"
    "</div>",
    unsafe_allow_html=True,
)


# ============================
# HARD-CODED MAP OF SAVED VENUES
# ============================
import plotly.express as px

st.subheader("🗺️ Your Liked Venues — Map View")

saved_venue_ids = current_user.saved_venues

if not saved_venue_ids:
    st.info("Here's Your latest RSVP distance")
else:
    venues = []
    for vid in saved_venue_ids:
        v = db.get_venue(vid)
        if v:
            venues.append({
                "name": v.name,
                "lat": v.location[0],
                "lon": v.location[1],
                "rating": v.rating,
                "category": v.category.value,
                "address": v.address
            })

    df = pd.DataFrame(venues)

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        hover_name="name",
        hover_data=["category", "rating", "address"],
        zoom=11,
        height=450
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

google_map_iframe = """
<iframe 
  src="https://www.google.com/maps/embed?pb=!1m28!1m12!1m3!1d193571.4383982468!2d-74.11976332364884!3d40.69767006412767!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!4m13!3e0!4m3!3m2!1d40.7580!2d-73.9855!4m3!3m2!1d40.785091!2d-73.968285!4m3!3m2!1d40.7061!2d-73.9969!5e0!3m2!1sen!2sus!4v1700000000000!5m2!1sen!2sus" 
  width="100%" 
  height="450" 
  style="border:0;" 
  allowfullscreen="" 
  loading="lazy" 
  referrerpolicy="no-referrer-when-downgrade">
</iframe>
"""

st.components.v1.html(google_map_iframe, height=500)