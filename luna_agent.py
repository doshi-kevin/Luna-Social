# luna_agent.py

import os
import json
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


class LunaAIAgent:
    """
    AI Agent powered by Gemini + hybrid recommendation logic.
    Handles: intent parsing, itinerary creation, semantic search, group-chat analysis, and chat responses.
    """

    def __init__(self, api_key, db, rec_engine):
        self.db = db
        self.rec_engine = rec_engine

        self.model = None
        self.embed_model = None

        if api_key:
            try:
                genai.configure(api_key=api_key)

                # Chat model (Gemini Flash / 2.5 Flash configurable via env)
                self.model = genai.GenerativeModel(
                    os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
                )

                # Embedding model for semantic search
                self.embed_model = os.getenv(
                    "GEMINI_EMBED_MODEL", "models/text-embedding-004"
                )

                print("✅ Gemini AI Agent initialized.")
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
                self.model = None
        else:
            print("⚪ No Gemini API key provided. Running in fallback mode.")

    # ---------------------------------------------------------------------
    # SEMANTIC SEARCH USING EMBEDDINGS (optional extra)
    # ---------------------------------------------------------------------
    def semantic_venue_search(self, query: str, top_k: int = 5):
        """Semantic search using Gemini embeddings."""
        if not self.embed_model:
            return []

        try:
            q_emb = genai.embed_content(
                model=self.embed_model,
                content=query
            )["embedding"]

            venues = self.rec_engine.db.get_all_venues()
            scores = []

            for v in venues:
                text = f"{v.name}. {v.description}. {v.category.value}"
                v_emb = genai.embed_content(
                    model=self.embed_model,
                    content=text,
                )["embedding"]

                sim = cosine_similarity(
                    np.array(q_emb).reshape(1, -1),
                    np.array(v_emb).reshape(1, -1),
                )[0][0]

                scores.append((v, float(sim)))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        except Exception as e:
            print(f"⚠️ semantic_venue_search failed: {e}")
            return []

    # ---------------------------------------------------------------------
    # INTENT PARSING
    # ---------------------------------------------------------------------
    def parse_user_intent(self, user_message: str) -> Dict:
        """
        Parse user intent using Gemini LLM (JSON output) with rule fallback.
        """
        # --- TRY GEMINI FIRST ---
        if self.model:
            try:
                system_prompt = (
                    "You are an intent parser for a social/venue recommendation app.\n"
                    "Actions: recommend, plan, book, query.\n"
                    "Return ONLY JSON like:\n"
                    "{ \"action\": \"plan\", \"parameters\": { \"vibe\": \"romantic\" } }"
                )

                result = self.model.generate_content(
                    [system_prompt, f"User message: {user_message}"]
                )

                text = result.text.strip()
                json_text = text[text.find("{"): text.rfind("}") + 1]
                parsed = json.loads(json_text)

                if "action" in parsed:
                    return parsed

            except Exception as e:
                print(f"⚠️ LLM intent parsing failed: {e}")

        # --- FALLBACK RULE-BASED SYSTEM ---
        message = user_message.lower()
        intent = {"action": "search", "parameters": {}}

        if any(w in message for w in ["plan", "organize", "create", "schedule"]):
            intent["action"] = "plan"
        elif any(w in message for w in ["recommend", "suggest", "find", "discover"]):
            intent["action"] = "recommend"
        elif any(w in message for w in ["book", "reserve"]):
            intent["action"] = "book"
        else:
            intent["action"] = "query"

        # Vibe keywords
        if "romantic" in message or "date" in message:
            intent["parameters"]["vibe"] = "romantic"
        if "adventure" in message:
            intent["parameters"]["vibe"] = "adventure"
        if "quiet" in message or "calm" in message:
            intent["parameters"]["vibe"] = "calm"

        return intent

    # ---------------------------------------------------------------------
    # ITINERARY GENERATION
    # ---------------------------------------------------------------------
    def generate_itinerary(self, user_id: str, num_venues: int = 3, vibe: str = "mixed") -> Dict:
        """Generate full-night itinerary with vibe filtering."""
        recommendations, reasoning = self.rec_engine.recommend_venues(user_id, limit=num_venues * 2)

        # Filter by vibe
        if vibe == "romantic":
            filtered = [v for v, s in recommendations if any(k in v.name.lower() for k in ["cafe", "rooftop", "intimate"])]
        elif vibe == "adventure":
            filtered = [v for v, s in recommendations if any(k in v.category.value.lower() for k in ["hiking", "beach", "nightlife"])]
        elif vibe == "calm":
            filtered = [v for v, s in recommendations if any(k in v.category.value.lower() for k in ["cafe", "park", "art"])]
        else:
            filtered = [v for v, s in recommendations]

        selected = filtered[:num_venues]

        start_time = 18
        itinerary = []

        for i, venue in enumerate(selected):
            itinerary.append({
                "time": f"{(start_time + i*2)%24:02d}:00",
                "venue": venue.name,
                "venue_id": venue.venue_id,
                "category": venue.category.value,
                "address": venue.address,
                "duration_hours": 2
            })

        return {
            "itinerary": itinerary,
            "total_duration_hours": num_venues * 2,
            "recommendation_reasoning": reasoning
        }

    # ---------------------------------------------------------------------
    # GROUP CHAT ANALYSIS
    # ---------------------------------------------------------------------
    def analyze_group_chat(self, group_messages: List[str], user_id: str) -> Dict:
        """
        Analyze group chat history using LLM + ML recommender + rule signals.
        Always returns valid recommendations.
        Detects: vibe, cuisine preference (e.g., Italian), budget, mood.
        """

        if not group_messages:
            return {"message": "No group messages found.", "recommendations": []}

        chat_text = "\n".join([f"- {msg}" for msg in group_messages])

        # -----------------------------------------
        # 1. DEFAULTS
        # -----------------------------------------
        group_vibe = "mixed"
        cuisine_pref = None
        budget_pref = "medium"
        mood_score = 60    # baseline value

        # -----------------------------------------
        # 2. LLM-based extraction (vibe + cuisine + budget + mood)
        # -----------------------------------------
        if self.model:
            try:
                prompt = (
                    "You are an AI analyzing a GROUP CHAT for planning a night out.\n"
                    "Extract the following ONLY as JSON:\n"
                    "{\n"
                    "  'vibe': 'romantic/adventure/calm/party/mixed',\n"
                    "  'cuisine': 'italian/chinese/japanese/indian/american/any',\n"
                    "  'budget': 'low/medium/high',\n"
                    "  'mood': 0-100 (overall positivity/energy)\n"
                    "}\n\n"
                    "Group Chat:\n" + chat_text + "\n\n"
                    "Respond ONLY in JSON. No commentary."
                )
                result = self.model.generate_content(prompt)
                text = result.text.strip()
                json_text = text[text.find("{"): text.rfind("}")+1]
                data = json.loads(json_text)

                group_vibe = data.get("vibe", "mixed").lower()
                cuisine_pref = data.get("cuisine", "any").lower()
                budget_pref = data.get("budget", "medium").lower()
                mood_score = int(data.get("mood", 60))

                if group_vibe not in ["romantic", "adventure", "calm", "party", "mixed"]:
                    group_vibe = "mixed"

            except Exception as e:
                print(f"⚠️ LLM group extraction failed: {e}")

        # -----------------------------------------
        # 3. RULE-BASED CUISINE detection (backup)
        # -----------------------------------------
        text_lower = chat_text.lower()
        if "italian" in text_lower or "pasta" in text_lower or "pizza" in text_lower:
            cuisine_pref = "italian"
        if "sushi" in text_lower:
            cuisine_pref = "japanese"
        if "spicy" in text_lower and not cuisine_pref:
            cuisine_pref = "indian"

        if not cuisine_pref:
            cuisine_pref = "any"

        # -----------------------------------------
        # 4. RULE-BASED Budget detection
        # -----------------------------------------
        if "cheap" in text_lower or "budget" in text_lower:
            budget_pref = "low"
        if "expensive" in text_lower or "premium" in text_lower:
            budget_pref = "high"

        # -----------------------------------------
        # 5. Get base ML recommendations
        # -----------------------------------------
        recs, reasoning = self.rec_engine.recommend_venues(user_id, limit=20)

        # -----------------------------------------
        # 6. Apply cuisine filtering FIRST (strongest signal)
        # -----------------------------------------
        cuisine_filtered = recs
        if cuisine_pref == "italian":
            cuisine_filtered = [(v, s) for v, s in recs if "ital" in v.name.lower() or "ital" in v.description.lower()]
        elif cuisine_pref == "japanese":
            cuisine_filtered = [(v, s) for v, s in recs if "japan" in v.description.lower() or "sushi" in v.name.lower()]
        elif cuisine_pref == "indian":
            cuisine_filtered = [(v, s) for v, s in recs if "indian" in v.description.lower()]

        # -----------------------------------------
        # 7. Apply vibe filtering
        # -----------------------------------------
        vibe_filtered = cuisine_filtered
        if group_vibe == "romantic":
            vibe_filtered = [(v, s) for v, s in vibe_filtered if "cafe" in v.name.lower() or "rooftop" in v.name.lower()]
        elif group_vibe == "adventure":
            vibe_filtered = [(v, s) for v, s in vibe_filtered if v.category.value.lower() in ["hiking", "nightlife"]]
        elif group_vibe == "calm":
            vibe_filtered = [(v, s) for v, s in vibe_filtered if v.category.value.lower() in ["park", "art", "cafe"]]
        elif group_vibe == "party":
            vibe_filtered = [(v, s) for v, s in vibe_filtered if v.category.value.lower() in ["nightlife"]]

        # -----------------------------------------
        # 8. FAILSAFE: If filtering produces empty list → use top recs
        # -----------------------------------------
        if not vibe_filtered:
            vibe_filtered = cuisine_filtered
        if not vibe_filtered:
            vibe_filtered = recs

        final_recs = vibe_filtered[:3]

        # -----------------------------------------
        # 9. Auto-Itinerary (Extra Feature)
        # -----------------------------------------
        itinerary = []
        start_time = 18
        for i, (v, s) in enumerate(final_recs):
            itinerary.append(f"🕒 {start_time + i*2}:00 → {v.name}")

        # -----------------------------------------
        # 10. Format Final Response
        # -----------------------------------------
        formatted = (
            f"🎉 **Group vibe detected:** {group_vibe.capitalize()}\n"
            f"🍽️ **Cuisine preference:** {cuisine_pref.capitalize()}\n"
            f"💰 **Budget:** {budget_pref.capitalize()}\n"
            f"😊 **Mood score:** {mood_score}/100\n\n"
            f"### 🏆 Top Recommendations:\n\n" +
            "\n".join(
                f"• **{v.name}** ({v.category.value}) — ⭐ {v.rating}/5\n  📍 {v.address}"
                for v, _ in final_recs
            ) +
            "\n\n### 🗺️ Suggested Itinerary:\n" +
            "\n".join(itinerary)
        )

        return {
            "vibe": group_vibe,
            "cuisine": cuisine_pref,
            "budget": budget_pref,
            "mood": mood_score,
            "recommendations": final_recs,
            "formatted": formatted,
        }

    # ---------------------------------------------------------------------
    # MAIN CHAT CONTROLLER
    # ---------------------------------------------------------------------
    def chat(self, user_id: str, message: str) -> Tuple[str, Dict]:
        """Main LLM-powered chat handler."""
        intent = self.parse_user_intent(message)

        response = {
            "intent": intent["action"],
            "parameters": intent["parameters"],
            "recommendations": None,
            "itinerary": None,
            "message": ""
        }

        try:
            if intent["action"] == "recommend":
                recs, reasoning = self.rec_engine.recommend_venues(user_id, limit=5)
                response["recommendations"] = recs
                response["message"] = "Here are top places for you:\n\n" + "\n".join(
                    f"• {v.name} ({v.category.value}) — ⭐ {v.rating}/5"
                    for v, _ in recs
                )

            elif intent["action"] == "plan":
                vibe = intent["parameters"].get("vibe", "mixed")
                itinerary = self.generate_itinerary(user_id, vibe=vibe)
                response["itinerary"] = itinerary
                response["message"] = self._format_itinerary(itinerary)

            elif intent["action"] == "book":
                response["message"] = (
                    "Sure! I can help book a venue 🎫\n"
                    f"Parameters detected: {intent['parameters']}\n\n"
                    "Tell me the venue name and preferred time!"
                )

            else:
                if self.model:
                    result = self.model.generate_content(
                        f"Answer briefly about nightlife/venues: {message}"
                    )
                    response["message"] = result.text
                else:
                    response["message"] = "Hey! How can I assist you?"

        except Exception as e:
            response["message"] = f"⚠️ Error: {e}"

        return response["message"], response

    # ---------------------------------------------------------------------
    # FORMAT ITINERARY
    # ---------------------------------------------------------------------
    def _format_itinerary(self, itinerary: Dict) -> str:
        text = "📅 Your Night Out Plan:\n\n"
        for item in itinerary["itinerary"]:
            text += (
                f"🕐 {item['time']} — {item['venue']} ({item['category']})\n"
                f"   📍 {item['address']}\n"
                f"   ⏱  {item['duration_hours']} hours\n\n"
            )
        return text
