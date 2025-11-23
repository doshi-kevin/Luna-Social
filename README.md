# 🚀 **Luna Social — AI-Powered Venue & Group Recommendation Engine**

## 🌟 Overview

**Luna Social** is an **AI-powered recommendation system** designed to plan nights out, recommend compatible people to go with, and analyze group chats to automatically suggest the best venues.
The system combines:

* **Machine Learning (Hybrid ML/Heuristic ranking)**
* **Spatial Analysis (Haversine distance)**
* **Behavioral Modeling**
* **LLM (Gemini 2.5 Flash) Agents**
* **Group Chat Reasoning**
* **Streamlit Frontend Analytics Dashboard**
* **Booking + QR Code Simulation**

This project fully satisfies **Track 2 Backend requirements**, showing complex ML logic, LLM reasoning, agentic actions, and end-to-end backend functionality.

---

## 🧠 Core Features

### ⭐ **1. Hybrid Recommendation Engine**

Uses both ML + heuristic scoring.

**Venue Feature Vector**
Each venue becomes a 5-dimensional feature vector:

```
[rating, distance_score, trending_score, category_preference, capacity_score]
```

**Hybrid Score:**

```
final_score = 0.5 * ML_ranker_prediction  +  0.5 * heuristic_score
```

### ⭐ **2. Behavioral Modeling**

We track:

* View duration (attention)
* Likes / Saves / Visits
* Category engagement
* Recency boosts
* User profile vector

This enables personalized recommendations per user.

---

## 🌍 **3. Spatial Analysis**

Distance between user & venue is computed using the **Haversine formula**.

Distance becomes a negative weighted component of recommendation score, ensuring closer venues are preferred unless overridden by user interest.

---

## 👥 **4. Social Compatibility Engine**

We compute cross-user compatibility with:

* Category overlap
* Interest similarity
* Chat similarity
* Geographical proximity

Used to recommend **people to go with**.

---

## 🤖 **5. AI Agent (Gemini 2.5 Flash)**

### The Luna AI Agent performs:

#### **a. Intent parsing**

Understands:

* “Find romantic places”
* “Plan my night”
* “Book a table”
* “Recommend Italian spots”

#### **b. Group chat analysis**

Agent extracts:

* **Vibe** (romantic, calm, adventure, party, mixed)
* **Cuisine preference** (Italian, Japanese, Indian…)
* **Budget level** (low/medium/high)
* **Mood score** (0–100)

#### **c. Venue filtering through LLM-derived signals**

ML recommendations are further filtered using:

* Vibe
* Cuisine
* Budget
* Mood

And fallback logic ensures **never empty recommendations**.

#### **d. Auto-Itinerary Generator**

Creates 3-stop evening plan:

```
6 PM  → Venue 1  
8 PM  → Venue 2  
10 PM → Venue 3  
```

#### **e. Booking Agent**

Simulates:

* Reservation creation
* Auto-booking triggers
* QR Code for confirmation

---

## 💬 **6. Group Chat Intelligence**

Users can chat inside groups.
Clicking **Analyze Group Chat** triggers:

* LLM sentiment & cuisine extraction
* Hybrid ML re-ranking
* Best 3 venues for the group
* Full itinerary
* Mood score
* Budget-friendly filters

Special feature:
**Built-in demo "NYC Italian Foodies Chat"** shows curated Italian restaurants.

---

## 📊 **7. Analytics Dashboard** *(Streamlit)*

We provide a full ML+LLM analytics console:

### 🧩 Behavioral Analytics

* Total interactions
* Likes/Saves/Views breakdown
* Category engagement bar chart
* Time spent per category (pie chart)

### 🧠 ML Model Internals

* RandomForest feature importance
* Stacked bar decomposition of recommendation score
* User profile vector breakdown

### 🤖 LLM Insight Panel

Shows:

* Extracted vibe
* Cuisine
* Budget
* Mood score
* Recommended venues

### 🔁 Pipeline Visualization

Clear step-by-step explanation of:

1. Behavioral signal processing
2. Feature extraction
3. ML ranking
4. LLM reasoning
5. Final filtering
6. Itinerary generation

---

## 🗂️ **Project Structure**

```
📦 luna-social-recommender/
│
├── app.py                     # Streamlit frontend (UI + analytics)
├── backend.py                 # Database, Models, Recommender, Booking logic
├── luna_agent.py              # AI Agent (LLM logic, chat analysis, itinerary)
├── luna_social.db             # SQLite database
├── .env                       # Gemini API key + model names
└── README.md                  # You're reading it :)
```

---

## 🧪 **Datasets**

We use synthetic + procedurally generated demo data:

### Users

* 50 demo users
* geolocation
* interests

### Venues

* Bars
* Restaurants
* Cafes
* Parks
* Nightlife spots
* Each with rating, trending, capacity, description, category

### Interactions

* Created for ML ranker training
* Includes VIEW + LIKE + SAVE + VISIT events

### Groups & Chats

* 5 demo groups
* “NYC Foodies” includes **hardcoded Italian-focused messages** for LLM reasoning
* All groups used for testing LLM group analysis

---

## ⚙️ **How to Run**

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Inside `.env`:

```
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL_NAME=gemini-2.5-flash
GEMINI_EMBED_MODEL=models/text-embedding-004
```

### 3. Run the app

```
streamlit run app.py
```

---

## 🏁 **How This Meets Track 2 Requirements**

### ✔ **Recommendation Engine**

* Hybrid ML + Heuristic scoring
* Feature vectors for every venue
* User behavior integrated

### ✔ **Spatial Analysis**

* Real Haversine distance scoring

### ✔ **Social Compatibility**

* Group analysis
* Compatibility scoring via overlap + proximity

### ✔ **AI Agents**

* Gemini powers:

  * Group chat extraction
  * Intent understanding
  * Itinerary building
  * Booking

### ✔ **Automation**

* Booking Agent auto-generates QR codes
* Agent-based flow for planning

---

## 🧩 **Why This System Stands Out**

* Combines **ML + LLM + analytics** in a single coherent pipeline
* Exhibits **agentic behavior** (not just recommendation)
* Provides explainability for every step
* Allows real user interaction & group discussions
* Demonstrates “backend heavy” engineering skills required for Track 2


📘 **Luna Social AI Companion — Presentation PDF**  
[Click here to open the PDF](socials/Luna-Social-AI-Companion.pdf)
<iframe
    src="https://docs.google.com/gview?url=https://raw.githubusercontent.com/USERNAME/REPO/main/socials/Luna-Social-AI-Companion.pdf&embedded=true"
    style="width:100%; height:600px;"
    frameborder="0">
</iframe>

## 🧠 NotebookLM Mind Map
<img src="socials/NotebookLM Mind Map.png" width="700">

## 🖼️ System Diagram
<img src="socials/unnamed.png" width="700">




