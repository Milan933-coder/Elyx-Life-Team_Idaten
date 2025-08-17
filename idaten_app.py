import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import re
import os
from collections import defaultdict
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import json
import hashlib

# Load environment variables
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state with enhanced structure
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
if 'episodes' not in st.session_state:
    st.session_state.episodes = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'text_corpus' not in st.session_state:
    st.session_state.text_corpus = []
if 'selected_episode' not in st.session_state:
    st.session_state.selected_episode = None
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""
if 'active_query' not in st.session_state:
    st.session_state.active_query = ""
if 'reranker' not in st.session_state:
    try:
        st.session_state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except:
        st.session_state.reranker = None
if 'topic_flow' not in st.session_state:
    st.session_state.topic_flow = []
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = []
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Timeline"

# Set page config
st.set_page_config(page_title="Elyx Member Journey", layout="wide")

# Custom CSS for better styling
# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding: 2rem;}
    .sidebar .sidebar-content {padding: 1rem;}
    .episode-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1890ff;
        margin-bottom: 1.5rem;
        background-color: #2c2c2c; /* dark gray */
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #3a3f47; /* slate gray */
        color: white;
        margin-bottom: 1rem;
    }
    .persona-state {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .positive-state { background-color: #145214; border-left: 4px solid #52c41a; }
    .negative-state { background-color: #661a1a; border-left: 4px solid #f5222d; }
    .neutral-state { background-color: #5a4a14; border-left: 4px solid #faad14; }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        color: white;
    }
    .user-message {
        background-color: #34495e; /* dark slate */
        margin-left: 20%;
        border-left: 4px solid #1890ff;
        color: white;
    }
    .bot-message {
        background-color: #1f3a1f; /* dark green */
        margin-right: 20%;
        border-left: 4px solid #52c41a;
        color: white;
    }
    .selected-episode {
        background-color: #003a8c; /* darker blue */
        border: 2px solid #1890ff;
        color: white;
    }
    .sankey-link { opacity: 0.6; transition: opacity 0.3s; }
    .sankey-link:hover { opacity: 1; stroke-width: 3px; }
    .interactive-chart { cursor: pointer; }
    .source-badge {
        background-color: #2c2c2c;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.8em;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 5px;
        color: white;
    }
    .query-expander {
        border-left: 3px solid #1890ff;
        padding-left: 10px;
        margin-bottom: 10px;
        font-style: italic;
        color: #ccc; /* lighter gray for contrast */
    }
    .stButton>button {
        background-color: #1890ff;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #40a9ff;
    }
</style>
""", unsafe_allow_html=True)


# OpenAI Model configuration
SUMMARIZATION_MODEL = "gpt-4o-mini"
REASONING_MODEL = "gpt-4o-mini"

# Consultation time estimates (in minutes)
CONSULTATION_ESTIMATES = {
    "Dr. Warren": {
        "test_review": 15,    # Reviewing test results
        "diagnosis": 20,      # Making a diagnosis
        "medication": 10,     # Prescribing medication
        "consult": 30         # General consultation
    },
    "Rachel": {
        "exercise_plan": 30,  # Creating exercise plan
        "form_review": 15,    # Reviewing exercise form
        "rehab": 20,          # Rehabilitation planning
        "checkin": 10         # Quick check-in
    },
    "Carla": {
        "diet_plan": 30,      # Creating diet plan
        "supplement": 15,     # Supplement recommendation
        "food_review": 10,    # Reviewing food log
        "checkin": 10         # Quick check-in
    },
    "Advik": {
        "data_analysis": 30,  # Data analysis session
        "experiment": 20,     # Designing an experiment
        "report": 15,         # Report review
        "checkin": 10         # Quick check-in
    },
    "Ruby": {
        "logistics": 10,      # Handling logistics
        "scheduling": 5,      # Scheduling appointments
        "travel": 15,         # Travel planning
        "coordination": 10    # General coordination
    },
    "Neel": {
        "strategy": 30,       # Strategic planning
        "escalation": 20,     # Handling escalations
        "review": 15,         # Quarterly review
        "feedback": 10        # Feedback session
    }
}

# Function to parse conversation data
def parse_conversation_data(uploaded_file):
    conversations = []
    current_conversation = None
    
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.split('\n')
    
    for line in lines:
        if re.match(r'^\[\d+/\d+/\d+, \d+:\d+ [AP]M\]', line):
            if current_conversation:
                conversations.append(current_conversation)
            parts = line.split('] ')
            timestamp = parts[0][1:]
            sender_content = parts[1].split(': ')
            if len(sender_content) > 1:
                sender = sender_content[0]
                content = ': '.join(sender_content[1:])
            else:
                sender = "System"
                content = sender_content[0]
            
            current_conversation = {
                "timestamp": datetime.strptime(timestamp, '%m/%d/%y, %I:%M %p'),
                "sender": sender,
                "content": content
            }
        elif current_conversation:
            current_conversation["content"] += "\n" + line
    
    if current_conversation:
        conversations.append(current_conversation)
    
    return sorted(conversations, key=lambda x: x["timestamp"])

# Function to create strict bi-weekly episodes
def create_biweekly_episodes(conversations):
    if not conversations:
        return []
    
    start_date = conversations[0]["timestamp"]
    end_date = conversations[-1]["timestamp"]
    
    episodes = []
    current_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_start < end_date:
        current_end = current_start + timedelta(days=14)
        
        episode_convs = [
            conv for conv in conversations
            if current_start <= conv["timestamp"] < current_end
        ]
        
        if episode_convs:
            episode = analyze_episode(episode_convs, current_start, current_end)
            episodes.append(episode)
        else:
            episodes.append({
                "start_date": current_start,
                "end_date": current_end,
                "conversations": [], 
                "participants": [],
                "topics": [],
                "metrics": {
                    "message_count": 0, 
                    "response_times": [],
                    "member_messages": 0,
                    "team_messages": 0,
                    "consultation_time": defaultdict(int),
                    "avg_response_time": 0, 
                    "max_response_time": 0
                },
                "persona_states": [],
                "summary": "No conversations during this period"
            })
        
        current_start = current_end
    
    return episodes

# Function to estimate consultation time
def estimate_consultation_time(sender, content):
    content = content.lower()
    consultation_time = 0
    role = "Unknown"
    
    # Map sender to role
    if "Dr. Warren" in sender:
        role = "Dr. Warren"
        if "test result" in content or "blood panel" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["test_review"]
        elif "diagnosis" in content or "condition" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["diagnosis"]
        elif "prescribe" in content or "medication" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["medication"]
        else:
            consultation_time = CONSULTATION_ESTIMATES[role]["consult"]
    
    elif "Rachel" in sender:
        role = "Rachel"
        if "workout plan" in content or "exercise program" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["exercise_plan"]
        elif "form" in content or "technique" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["form_review"]
        elif "rehab" in content or "injury" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["rehab"]
        else:
            consultation_time = CONSULTATION_ESTIMATES[role]["checkin"]
    
    elif "Carla" in sender:
        role = "Carla"
        if "diet" in content or "meal plan" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["diet_plan"]
        elif "supplement" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["supplement"]
        elif "food log" in content or "nutrition" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["food_review"]
        else:
            consultation_time = CONSULTATION_ESTIMATES[role]["checkin"]
    
    elif "Advik" in sender:
        role = "Advik"
        if "data" in content or "analysis" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["data_analysis"]
        elif "experiment" in content or "trial" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["experiment"]
        elif "report" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["report"]
        else:
            consultation_time = CONSULTATION_ESTIMATES[role]["checkin"]
    
    elif "Ruby" in sender:
        role = "Ruby"
        if "travel" in content or "itinerary" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["travel"]
        elif "schedule" in content or "appointment" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["scheduling"]
        elif "coordinate" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["coordination"]
        else:
            consultation_time = CONSULTATION_ESTIMATES[role]["logistics"]
    
    elif "Neel" in sender:
        role = "Neel"
        if "strategy" in content or "plan" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["strategy"]
        elif "escalate" in content or "concern" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["escalation"]
        elif "review" in content or "QBR" in content:
            consultation_time = CONSULTATION_ESTIMATES[role]["review"]
        else:
            consultation_time = CONSULTATION_ESTIMATES[role]["feedback"]
    
    return role, consultation_time

# Function to analyze an episode
def analyze_episode(conversations, start_date, end_date):
    participants = set(conv["sender"] for conv in conversations)
    topics = detect_topics(conversations)
    metrics = calculate_metrics(conversations)
    persona_states = analyze_persona_states(conversations)
    
    consultation_time = defaultdict(int)
    for conv in conversations:
        if any(role in conv["sender"] for role in CONSULTATION_ESTIMATES.keys()):
            role, time_estimate = estimate_consultation_time(conv["sender"], conv["content"])
            if role != "Unknown":
                consultation_time[role] += time_estimate
    
    metrics["consultation_time"] = dict(consultation_time)
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "conversations": conversations,
        "participants": list(participants),
        "topics": topics,
        "metrics": metrics,
        "persona_states": persona_states,
        "summary": generate_episode_summary(conversations)
    }

# Function to detect topics
def detect_topics(conversations):
    topic_keywords = {
        "onboarding": ["onboard", "initial", "first", "sign up"],
        "diagnostics": ["test", "result", "blood", "panel", "scan"],
        "nutrition": ["food", "diet", "supplement", "eat", "meal"],
        "exercise": ["workout", "exercise", "gym", "run", "train"],
        "travel": ["travel", "flight", "trip", "hotel"],
        "frustration": ["frustrat", "angry", "disappoint", "unhappy"],
        "medication": ["med", "pill", "prescription", "dose", "take"],
        "therapy": ["therapy", "session", "treatment", "intervention"],
        "progress": ["improve", "better", "progress", "gain", "achieve"]
    }
    
    topics = defaultdict(int)
    for conv in conversations:
        content = conv["content"].lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in content for keyword in keywords):
                topics[topic] += 1
    
    return sorted(topics.items(), key=lambda x: x[1], reverse=True)

# Function to calculate metrics
def calculate_metrics(conversations):
    metrics = {
        "message_count": len(conversations),
        "response_times": [],
        "member_messages": 0,
        "team_messages": 0
    }
    
    member_name = "Member"
    if conversations:
        for conv in conversations:
            if "(" not in conv["sender"] and ")" not in conv["sender"] and conv["sender"] != "System":
                member_name = conv["sender"].split(":")[0]
                break
    
    team_members = ["Dr.", "Advik", "Carla", "Rachel", "Neel", "Ruby", "Concierge", "Elyx"]
    
    for i in range(len(conversations) - 1):
        if member_name in conversations[i]["sender"]:
            metrics["member_messages"] += 1
            # Ensure next message is from the team to calculate response time
            if any(member in conversations[i+1]["sender"] for member in team_members):
                response_time = (conversations[i+1]["timestamp"] - conversations[i]["timestamp"])
                metrics["response_times"].append(response_time.total_seconds() / 60)
        elif any(member in conversations[i]["sender"] for member in team_members):
            metrics["team_messages"] += 1
    
    if metrics["response_times"]:
        metrics["avg_response_time"] = np.mean(metrics["response_times"])
        metrics["max_response_time"] = np.max(metrics["response_times"])
    else:
        metrics["avg_response_time"] = 0
        metrics["max_response_time"] = 0
    
    return metrics

# Function to analyze persona states
def analyze_persona_states(conversations):
    states = []
    for conv in conversations:
        content = conv["content"].lower()
        if any(k in content for k in ["frustrat", "angry", "disappoint", "unhappy"]):
            state = "negative"
        elif any(k in content for k in ["happy", "good", "improve", "better"]):
            state = "positive"
        else:
            state = "neutral"
        
        states.append({
            "state": state,
            "reason": conv["content"],
            "sender": conv["sender"], 
            "timestamp": conv["timestamp"]
        })
    
    return states

# Function to generate episode summary using OpenAI
def generate_episode_summary(conversations):
    if not openai.api_key:
        return "OpenAI API not configured - cannot generate summary"
    
    context = "\n".join(
        f"{conv['timestamp'].strftime('%m/%d %H:%M')} - {conv['sender']}: {conv['content'][:200]}"
        for conv in conversations[:20]
    )
    
    try:
        response = openai.chat.completions.create(
            model=SUMMARIZATION_MODEL,
            messages=[
                {"role": "system", "content": "You are a healthcare journey analyst. Create a concise summary of this episode with these sections: 1) Primary Goal/Trigger, 2) Key Outcomes, 3) Member State Analysis. Use bullet points."},
                {"role": "user", "content": f"Summarize this healthcare episode:\n\n{context}"}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

# Function to initialize TF-IDF vectorizer
def initialize_vectorizer(conversations):
    text_corpus = [f"{conv['sender']}: {conv['content']}" for conv in conversations]
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    return vectorizer, tfidf_matrix, text_corpus

# Enhanced RAG system with two-stage retrieval
def retrieve_relevant_context(question, top_k=5, rerank_top_k=20):
    """Enhanced retrieval with query transformation and reranking"""
    if not st.session_state.conversations:
        return "No conversation data available"
    
    # Step 1: Query transformation
    expanded_query = transform_query(question)
    
    # Step 2: Initial retrieval (semantic + metadata filtering)
    question_vec = st.session_state.vectorizer.transform([expanded_query])
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(question_vec, st.session_state.tfidf_matrix)[0]
    top_indices = np.argsort(similarities)[-rerank_top_k:][::-1]
    candidate_contexts = [st.session_state.text_corpus[idx] for idx in top_indices]
    
    # Step 3: Reranking with cross-encoder
    if st.session_state.reranker:
        pairs = [(expanded_query, context) for context in candidate_contexts]
        rerank_scores = st.session_state.reranker.predict(pairs)
        top_reranked_indices = np.argsort(rerank_scores)[-top_k:][::-1]
    else:
        top_reranked_indices = range(min(top_k, len(candidate_contexts)))
    
    # Step 4: Format context with sources
    context = ""
    for idx in top_reranked_indices:
        conv_idx = top_indices[idx]
        conv = st.session_state.conversations[conv_idx]
        source = f"{conv['timestamp'].strftime('%m/%d %H:%M')} - {conv['sender']}"
        context += f"<div class='source-badge'>{source}</div> {candidate_contexts[idx]}\n\n"
    
    return context

def transform_query(question):
    """Use LLM to expand and contextualize the query"""
    if not openai.api_key:
        return question
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a query transformation expert. Expand the user's question to include relevant context for searching healthcare conversations. Include implied timeframes, speaker roles, and health concepts."},
                {"role": "user", "content": f"Original question: {question}\n\nExpanded query:"}
            ],
            max_tokens=100,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except:
        return question

# Enhanced chatbot response with RAG improvements
def get_chatbot_response(question):
    if not openai.api_key:
        return "OpenAI API not configured - chatbot unavailable"
    
    if not st.session_state.conversations:
        return "No conversation data available"
    
    context = retrieve_relevant_context(question)
    
    try:
        # Store active query for UI display
        st.session_state.active_query = question
        
        response = openai.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an Elyx healthcare assistant. Answer the user's question about a member's journey "
                        "using ONLY the provided context. For clinical decisions (medications, tests, therapies):\n"
                        "1. State the recommendation\n"
                        "2. Cite the timestamped source conversation\n"
                        "3. Explain the medical rationale\n"
                        "4. Mention any relevant member context\n"
                        "Format: [Source: Timestamp - Sender] Explanation"
                    )
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            max_tokens=400,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting chatbot response: {str(e)}"

# New visualization: Topic Flow Sankey Diagram
def create_topic_sankey(episodes):
    if not episodes or len(episodes) < 2:
        return None
    
    # Extract topics over time
    topic_sequence = []
    for i, ep in enumerate(episodes):
        if ep["topics"]:
            primary_topic = ep["topics"][0][0]
            topic_sequence.append((f"Ep {i+1}", primary_topic))
    
    # Create node list
    nodes = list(set([item for sublist in topic_sequence for item in sublist]))
    node_dict = {node: idx for idx, node in enumerate(nodes)}
    
    # Create links between episodes
    links = defaultdict(int)
    for i in range(len(topic_sequence)-1):
        source = node_dict[topic_sequence[i][0]]
        target = node_dict[topic_sequence[i+1][1]]
        links[(source, target)] += 1
    
    # Prepare Sankey data
    source, target, value = [], [], []
    for (s, t), v in links.items():
        source.append(s)
        target.append(t)
        value.append(v)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=["#1890ff" if "Ep" in node else "#52c41a" for node in nodes]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(180, 180, 180, 0.6)"
        )
    )])
    
    fig.update_layout(
        title_text="Topic Evolution Flow Between Episodes",
        font_size=12,
        height=600
    )
    return fig

# New visualization: Sentiment Timeline
#-------------------------------------------------------------#
# --- Assume OpenAI client is already initialized in your main code ---
# from openai import OpenAI
# client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
# --------------------------------------------------------------------

def get_sentiment_from_openai(text, client):
    """
    Analyzes the sentiment of a given text using the OpenAI API.

    Args:
        text (str): The text content to analyze.
        client (openai.OpenAI): The initialized OpenAI client.

    Returns:
        int: A sentiment score (-1 for negative, 0 for neutral, 1 for positive).
    """
    # We limit the text length to avoid excessive token usage for very long messages
    max_text_length = 500 
    truncated_text = text[:max_text_length]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or another model like gpt-4o
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert. Classify the user's text as 'positive', 'negative', or 'neutral'. Respond with only a single word: positive, negative, or neutral."
                },
                {
                    "role": "user",
                    "content": truncated_text
                }
            ],
            temperature=0,
            max_tokens=5
        )
        
        sentiment = response.choices[0].message.content.lower().strip()

        # Map the text response to a numerical score
        if "positive" in sentiment:
            return 1
        elif "negative" in sentiment:
            return -1
        else:
            return 0 # Default to neutral for "neutral" or any unexpected response
            
    except Exception as e:
        print(f"An error occurred while calling OpenAI API: {e}")
        return 0 # Return neutral if the API call fails
    

def create_sentiment_timeline(conversations, client):
    """
    Creates a sentiment timeline chart from conversation data using OpenAI for sentiment analysis.
    """
    if not conversations:
        return None
    
    data = []
    for conv in conversations:
        # We only analyze messages from the member/customer side
        if "Member" in conv["sender"] or "Rohan" in conv["sender"]:
            
            # --- MODIFIED PART ---
            # Replace the hardcoded logic with a call to the OpenAI API
            score = get_sentiment_from_openai(conv["content"], client)
            # -------------------

            data.append({
                "timestamp": conv["timestamp"],
                "sentiment": score,
                "content": conv["content"][:100] + ("..." if len(conv["content"]) > 100 else "")
            })
    
    if not data:
        # This can happen if no messages from "Member" or "Rohan" are found
        return None
    
    # The rest of the function remains the same
    df = pd.DataFrame(data)
    # Use a rolling average to smooth out the sentiment trend
    df['smoothed'] = df['sentiment'].rolling(window=5, min_periods=1).mean()
    
    fig = px.line(df, x="timestamp", y="smoothed", 
                  title="Member Sentiment Trend (Analyzed by AI)",
                  hover_data=["content"])
    
    fig.update_layout(
        yaxis_title="Sentiment Score (Smoothed)",
        xaxis_title="Date",
        hovermode="x unified",
        height=400
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig

#-------------------------------------------------------------#

# def create_sentiment_timeline(conversations):
#     if not conversations:
#         return None
    
#     # Calculate sentiment scores
#     data = []
#     for conv in conversations:
#         if "Member" in conv["sender"] or "Rohan" in conv["sender"]:
#             content = conv["content"].lower()
#             if any(k in content for k in ["frustrat", "angry", "disappoint", "unhappy"]):
#                 score = -1
#             elif any(k in content for k in ["happy", "good", "improve", "better"]):
#                 score = 1
#             else:
#                 score = 0
#             data.append({
#                 "timestamp": conv["timestamp"],
#                 "sentiment": score,
#                 "content": conv["content"][:100] + ("..." if len(conv["content"]) > 100 else "")
#             })
    
#     if not data:
#         return None
    
#     df = pd.DataFrame(data)
#     df['smoothed'] = df['sentiment'].rolling(window=5, min_periods=1).mean()
    
#     fig = px.line(df, x="timestamp", y="smoothed", 
#                  title="Member Sentiment Trend",
#                  hover_data=["content"])
    
#     fig.update_layout(
#         yaxis_title="Sentiment Score",
#         xaxis_title="Date",
#         hovermode="x unified",
#         height=400
#     )
#     fig.add_hline(y=0, line_dash="dash", line_color="gray")
#     return fig

# Generate unique key for buttons
def generate_unique_key(episode, prefix):
    """Generate unique key using episode dates"""
    key_str = f"{prefix}_{episode['start_date']}_{episode['end_date']}"
    return hashlib.md5(key_str.encode()).hexdigest()

# Function to display episode card
def display_episode(episode, index, tab_name):
    # Add custom class if this episode is selected
    extra_class = "selected-episode" if st.session_state.selected_episode == index else ""
    
    with st.container():
        st.markdown(f'<div class="episode-card {extra_class}">', unsafe_allow_html=True)
        
        st.subheader(f"Episode {index + 1}: {episode['start_date'].strftime('%b %d')} to {episode['end_date'].strftime('%b %d, %Y')}")
        
        # Summary
        st.markdown("**Summary:**")
        st.write(episode["summary"])
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", episode["metrics"]["message_count"])
        col2.metric("Avg Response Time", f"{episode['metrics']['avg_response_time']:.1f} min")
        col3.metric("Member/Team Ratio", f"{episode['metrics']['member_messages']}/{episode['metrics']['team_messages']}")
        
        # Consultation time
        if episode["metrics"]["consultation_time"]:
            st.markdown("**Consultation Time (minutes):**")
            for role, minutes in episode["metrics"]["consultation_time"].items():
                st.write(f"- {role}: {minutes} min")
        
        # Topics
        if episode["topics"]:
            st.markdown("**Key Topics:**")
            st.write(", ".join([f"{topic} ({count})" for topic, count in episode["topics"]]))
        
        # Persona analysis
        st.markdown("**Member State Analysis:**")
        if episode["persona_states"]:
            state_counts = {
                "positive": len([s for s in episode["persona_states"] if s["state"] == "positive"]),
                "negative": len([s for s in episode["persona_states"] if s["state"] == "negative"]),
                "neutral": len([s for s in episode["persona_states"] if s["state"] == "neutral"])
            }
            
            # Simple visualization
            st.write(f"😊 Positive: {state_counts['positive']} | 😐 Neutral: {state_counts['neutral']} | 😠 Negative: {state_counts['negative']}")
            
            # Show last state
            last_state = episode["persona_states"][-1]
            state_class = f"{last_state['state']}-state"
            state_emoji = "😊" if last_state["state"] == "positive" else "😠" if last_state["state"] == "negative" else "😐"
            st.markdown(f'<div class="persona-state {state_class}">'
                       f'{state_emoji} Final State: {last_state["state"].title()}<br>'
                       f'<small>"{last_state["reason"][:100]}{"..." if len(last_state["reason"]) > 100 else ""}"</small></div>', 
                       unsafe_allow_html=True)
        
        # Add contextual action button with unique key
        unique_key = generate_unique_key(episode, f"episode_btn_{tab_name}")
        if st.button(f"Ask about Episode {index+1}", key=unique_key):
            st.session_state.chat_input = (
                f"What were the key decisions made in Episode {index+1} and what was their rationale? "
                f"Timeframe: {episode['start_date'].strftime('%b %d')} to {episode['end_date'].strftime('%b %d')}"
            )
            st.session_state.selected_tab = "Chat Assistant"
            st.rerun()
        
        # Show full conversations if expanded
        with st.expander("View detailed conversations"):
            for conv in episode["conversations"]:
                st.markdown(f"**{conv['timestamp'].strftime('%m/%d %H:%M')} - {conv['sender']}**")
                st.write(conv["content"])
                st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Function to create topic evolution plot with interactivity
def create_interactive_topic_plot(episodes):
    plot_data = []
    for i, ep in enumerate(episodes):
        for topic, count in ep["topics"]:
            plot_data.append({
                "Episode": f"Ep {i+1}",
                "Start Date": ep["start_date"].strftime('%Y-%m-%d'),
                "Topic": topic,
                "Count": count,
                "Index": i
            })
    
    if not plot_data:
        return None
    
    df = pd.DataFrame(plot_data)
    # Create interactive plot
    fig = px.bar(df, x="Episode", y="Count", color="Topic",
                 hover_data=["Start Date"], title="Topic Evolution Over Time",
                 barmode="stack")
    
    fig.update_layout(xaxis_title="Bi-Weekly Episode", yaxis_title="Topic Count",
                      legend_title="Topics", hovermode="x unified")
    
    return fig

# Enhanced chat UI with source citations
def display_chat_ui():
    st.header("Elyx Journey Assistant")
    st.markdown("Ask questions about the member's journey.")
    
    # Display active query context
    if st.session_state.active_query:
        st.markdown(f"<div class='query-expander'>Active context: {st.session_state.active_query}</div>", 
                   unsafe_allow_html=True)
    
    # Capability explanation
    st.info("""
    I can answer questions about Rohan's health journey. Try:
    - "Why was Magnesium Threonate recommended?"
    - "What triggered the change in exercise plan in Episode 3?"
    - "Show decisions related to cardiovascular health"
    """)
    
    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Why was the magnesium supplement prescribed?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing journey..."):
                response = get_chatbot_response(prompt)
                st.markdown(response, unsafe_allow_html=True)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Main App
def main():
    st.markdown(
        "<p style='text-align:center; color:grey; font-size:20px;'>"
        "TEAM IDATEN | © Created by <b>Dhirendra & Lokaranjan</b>"
        "</p>",
        unsafe_allow_html=True
    )
    st.title("Elyx Member Journey Analytics")
    
    with st.sidebar:
        st.header("Upload Conversation Data")
        uploaded_file = st.file_uploader("Choose a conversation file", type=["txt"])
        
        if uploaded_file is not None:
            st.session_state.conversations = parse_conversation_data(uploaded_file)
            st.session_state.episodes = create_biweekly_episodes(st.session_state.conversations)
            
            # Initialize the vectorizer for RAG
            if st.session_state.conversations:
                st.session_state.vectorizer, st.session_state.tfidf_matrix, st.session_state.text_corpus = initialize_vectorizer(st.session_state.conversations)
            st.success("Data uploaded and processed successfully!")
        
        st.markdown("### Sample Data Format")
        st.code("[1/15/25, 2:15 PM] Rohan: Ruby...\n[1/15/25, 2:38 PM] Ruby (Elyx): Hi...", language="text")
        
        if openai.api_key:
            st.success("OpenAI API connected")
        else:
            st.warning("OpenAI API not configured")
            
        st.markdown("### Models Used")
        st.markdown(f"- **Summarization**: `{SUMMARIZATION_MODEL}`\n- **Reasoning**: `{REASONING_MODEL}`")
        st.markdown("### RAG System")
        st.markdown("- Query transformation\n- Two-stage retrieval\n- Cross-encoder reranking")
    
    if not st.session_state.episodes:
        st.warning("Please upload conversation data to begin")
        return
    
    # Initialize tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Metrics", "Analytics", "Chat Assistant"])
    
    with tab1:
        st.header("Bi-Weekly Journey Timeline")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", 
                                     value=st.session_state.episodes[0]["start_date"].date(),
                                     min_value=st.session_state.episodes[0]["start_date"].date(),
                                     max_value=st.session_state.episodes[-1]["end_date"].date())
        with col2:
            end_date = st.date_input("End date", 
                                   value=st.session_state.episodes[-1]["end_date"].date(),
                                   min_value=st.session_state.episodes[0]["start_date"].date(),
                                   max_value=st.session_state.episodes[-1]["end_date"].date())
        
        # Filter episodes
        filtered_episodes = [
            ep for ep in st.session_state.episodes
            if start_date <= ep["start_date"].date() <= end_date
            or start_date <= ep["end_date"].date() <= end_date
        ]
        
        # Add interactive selection
        if filtered_episodes:
            selected_index = st.selectbox("Jump to episode:", 
                                         range(len(filtered_episodes)),
                                         format_func=lambda i: f"Episode {i+1}",
                                         key="episode_selector")
            display_episode(filtered_episodes[selected_index], selected_index, "timeline")
        else:
            st.warning("No episodes found in the selected date range")
    
    with tab2:
        st.header("Journey Metrics")
        
        # Overall statistics
        st.subheader("Overall Statistics")
        total_messages = sum(ep["metrics"]["message_count"] for ep in st.session_state.episodes)
        avg_response = sum(ep["metrics"]["avg_response_time"] for ep in st.session_state.episodes) / len(st.session_state.episodes)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("Average Response Time", f"{avg_response:.1f} minutes")
        
        # Consultation time statistics
        st.subheader("Consultation Time Summary")
        consultation_totals = defaultdict(int)
        for ep in st.session_state.episodes:
            for role, minutes in ep["metrics"].get("consultation_time", {}).items():
                consultation_totals[role] += minutes
        
        if consultation_totals:
            # Convert to hours
            consultation_hours = {role: minutes/60 for role, minutes in consultation_totals.items()}
            df_consult = pd.DataFrame({
                "Role": list(consultation_hours.keys()),
                "Hours": list(consultation_hours.values())
            })
            fig = px.bar(df_consult, x="Role", y="Hours", 
                         title="Total Consultation Hours by Role",
                         color="Role", text="Hours")
            fig.update_traces(texttemplate='%{y:.1f}h', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No consultation data available")
        
        # Response time trend
        st.subheader("Response Time Trend")
        response_data = []
        for ep in st.session_state.episodes:
            response_data.append({
                "Date": ep["start_date"],
                "Avg Response Time (min)": ep["metrics"]["avg_response_time"],
                "Max Response Time (min)": ep["metrics"]["max_response_time"]
            })
        
        if response_data:
            response_df = pd.DataFrame(response_data)
            response_df.set_index("Date", inplace=True)
            st.line_chart(response_df)
        else:
            st.warning("No response time data available")
        
        # Interactive topic evolution
        st.subheader("Topic Evolution")
        topic_plot = create_interactive_topic_plot(st.session_state.episodes)
        if topic_plot:
            st.plotly_chart(topic_plot, use_container_width=True)
            
            # Add cross-filtering
            if topic_plot:
                selected_topic = st.selectbox("Filter by topic:", 
                                             [t for t,_ in st.session_state.episodes[0]["topics"]],
                                             key="topic_filter")
                filtered_eps = [i for i, ep in enumerate(st.session_state.episodes) 
                               if any(t[0] == selected_topic for t in ep["topics"])]
                
                if filtered_eps:
                    st.subheader(f"Episodes with '{selected_topic}' topic")
                    for i in filtered_eps:
                        display_episode(st.session_state.episodes[i], i, "metrics")
        else:
            st.warning("No topic data available")
            
    with tab3:
        st.header("Advanced Journey Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Topic Flow Analysis")
            sankey = create_topic_sankey(st.session_state.episodes)
            if sankey:
                st.plotly_chart(sankey, use_container_width=True)
            else:
                st.warning("Not enough data for topic flow analysis")
        
        with col2:
            st.subheader("Member Sentiment Timeline")
            sentiment_plot = create_sentiment_timeline(st.session_state.conversations,openai)
            if sentiment_plot:
                st.plotly_chart(sentiment_plot, use_container_width=True)
            else:
                st.warning("No member sentiment data available")
    
    with tab4:
        display_chat_ui()

if __name__ == "__main__":

    main()

