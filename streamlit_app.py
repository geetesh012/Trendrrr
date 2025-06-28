import streamlit as st
import pandas as pd
import requests
import joblib
import shap
import streamlit.components.v1 as components
import os
import uuid
import matplotlib.pyplot as plt
import plotly.express as px
import http.client
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from datetime import datetime
from faker import Faker
from transformers import pipeline
import nltk
from urllib.parse import quote

fake = Faker()
nltk.download("vader_lexicon")

st.set_page_config(layout="wide", page_title="Trendrrr")
# st.markdown('<a name="top"></a>', unsafe_allow_html=True)
st.title("X (Twitter) Trend Analyzer")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(
        """
    <div style='display: flex; align-items: center; height: 100%;'>
    <div style='font-size:48px; color:#ffffff; line-height:1.3; padding-top: 50px; color: #1A93DE; font-weight: bold;'>
        Innovate Insights Shaping Social Media Marketing. What's Trending???
        <div style='margin-top: 30px; font-size: 22px; font-weight: bold; color: #ffffff; padding-top: 5%; padding-bottom: 5%'>Want to know more?</div>
    </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("Get Started"):
        st.success("Let's go! Scroll down to explore the features.")

with col2:
    st.markdown(
        "<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>",
        unsafe_allow_html=True,
    )
    st.image("assets/image.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Custom CSS for some spacing and styling
st.markdown(
    """
<style>
    body {
        background-color: #040C18;
        color: #ffffff;
    }
    .stApp {
        background-color: #040C18;
        color: #ffffff;
    }
    .section-header {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #AE67FA;
    }
    .subsection-header {
        font-size: 18px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #1A93DE;
        margin-top: 40px;
        margin-bottom: 10px;
    }
        h1 {
        background: linear-gradient(89.97deg, #AE67FA 1.84%, #F49867 102.67%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }
    /* Input fields */
    .stTextInput > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div,
    .stRadio > div {
        color: #ffffff !important;
        background-color: #0B0F1A !important;
    }
    label, .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #cccccc !important;
        font-weight: 500;
    }
    /* Buttons */
    .stButton>button {
        background-color: #1A93DE;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #167ABC;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)
# st.markdown("<br><hr>", unsafe_allow_html=True)
# footer = """
# <div style="text-align: center; padding: 10px;">
#     <a href="#top" style="margin: 0 15px; text-decoration: none; color: #4F8BF9;">Home</a> |
#     <a href="https://yourwebsite.com/about" target="_blank" style="margin: 0 15px; text-decoration: none; color: #4F8BF9;">About</a> |
#     <a href="mailto:contact@yourwebsite.com" style="margin: 0 15px; text-decoration: none; color: #4F8BF9;">Contact</a>
# </div>
# """

@st.cache_data
# def load_data(filepath):
#     try:
#         data = pd.read_csv(filepath, engine="python").dropna(subset=["tweet"])
#         return data
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return pd.DataFrame()


# def filter_tweets_by_person(df, person):
#     if person:
#         return df[df["tweet"].str.contains(person, case=False, na=False)]
#     return df


# def filter_tweets_by_sentiment(df, sentiment):
#     if sentiment.lower() != "all" and "Sentiment" in df.columns:
#         return df[df["Sentiment"].str.lower() == sentiment.lower()]
#     return df


# def plot_sentiment_distribution(df):
#     sentiment_counts = df["Sentiment"].value_counts()
#     st.markdown(
#         '<div class="subsection-header">Sentiment Distribution</div>',
#         unsafe_allow_html=True,
#     )

#     col1, col2 = st.columns(2)

#     with col1:
#         with st.expander("Bar Chart"):
#             fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
#             sentiment_counts.plot(
#                 kind="bar", color=["green", "orange", "red"], ax=ax_bar
#             )
#             ax_bar.set_xlabel("Sentiment")
#             ax_bar.set_ylabel("Count")
#             ax_bar.grid(axis="y", linestyle="--", alpha=0.7)
#             st.pyplot(fig_bar)

#     with col2:
#         with st.expander("Pie Chart"):
#             fig_pie, ax_pie = plt.subplots(figsize=(5, 3))
#             ax_pie.pie(
#                 sentiment_counts,
#                 labels=sentiment_counts.index,
#                 autopct="%1.1f%%",
#                 startangle=90,
#                 colors=["#2ca02c", "#ff7f0e", "#d62728"],
#             )
#             ax_pie.axis("equal")
#             st.pyplot(fig_pie)


# def plot_trend_over_time(df, person):
#     if 'date' not in df.columns:
#         st.info("No 'date' column found. Cannot show trend over time.")
#         return

#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df = df.dropna(subset=['date'])

#     if df.empty:
#         st.info("No valid date data available for trend analysis.")
#         return

#     st.markdown('<div class="subsection-header">Tweet Trend Over Time</div>', unsafe_allow_html=True)

#     with st.expander("Tweet Frequency Over Time"):
#         trend_data = df.groupby(df['date'].dt.date).size().reset_index(name='Tweet Count')
#         fig_trend = px.line(trend_data, x='date', y='Tweet Count',
#                             title=f"Tweet Frequency Over Time for '{person}'",
#                             width=700, height=350)
#         st.plotly_chart(fig_trend, use_container_width=True)

#     if 'Sentiment' in df.columns:
#         with st.expander("Daily Tweet Sentiment Distribution"):
#             trend_sentiment = df.groupby([df['date'].dt.date, 'Sentiment']).size().reset_index(name='Count')
#             fig_stack = px.area(trend_sentiment, x='date', y='Count', color='Sentiment',
#                                 title="Daily Tweet Sentiment Distribution",
#                                 line_group='Sentiment',
#                                 width=700, height=350)
#             st.plotly_chart(fig_stack, use_container_width=True)


# def analyze_sentiment_with_distilbert(tweet):
#     try:
#         with st.spinner("Analyzing sentiment with DistilBERT..."):
#             response = requests.post(
#                 "http://localhost:5000/predict", json={"text": tweet}
#             )
#             if response.status_code == 200:
#                 result = response.json()
#                 st.success(
#                     f"Predicted Sentiment (DistilBERT): **{result.get('sentiment', 'Unknown')}**"
#                 )
#             else:
#                 st.error(f"Prediction API returned status code {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error connecting to prediction API: {e}")


# def get_trend_prediction(tweet):
#     """
#     Call the backend to get trend prediction, confidence,
#     reason why it will trend or not, and estimated trend duration in days.
#     """
#     try:
#         with st.spinner("Predicting trend likelihood..."):
#             res = requests.post(
#                 "http://localhost:5000/predict_trend", json={"text": tweet}
#             )
#             if res.status_code == 200:
#                 result = res.json()
#                 # Expected keys: will_trend (bool), confidence (float), explanation (str), trend_duration_days (int)
#                 will_trend = result.get("will_trend", False)
#                 confidence = result.get("confidence", 0)
#                 explanation = result.get(
#                     "explanation", "No detailed explanation provided."
#                 )
#                 duration = result.get("trend_duration_days", 0)
#                 return will_trend, confidence, explanation, duration
#             else:
#                 st.error(f"Trend API error: {res.status_code}")
#                 return None, 0, "API error occurred", 0
#     except Exception as e:
#         st.error(f"Connection error: {e}")
#         return None, 0, "Connection error", 0


# def analyze_sentiment_with_distilbert(tweet):
#     try:
#         with st.spinner("Analyzing sentiment..."):
#             response = requests.post(
#                 "http://localhost:5000/predict_sentiment", json={"text": tweet}
#             )
#             if response.status_code == 200:
#                 result = response.json()
#                 st.success(
#                     f"Predicted Sentiment: *{result.get('sentiment', 'Unknown')}*"
#                 )
#             else:
#                 st.error(f"API error: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error: {e}")


# ---------------------- Visualizations ------------------------


# def plot_sentiment_distribution(df):
#     sentiment_counts = df["Sentiment"].value_counts()
#     st.markdown(
#         '<div class="subsection-header">Sentiment Distribution</div>',
#         unsafe_allow_html=True,
#     )
#     col1, col2 = st.columns(2)
#     with col1:
#         with st.expander("Bar Chart"):
#             fig, ax = plt.subplots(figsize=(5, 3))
#             sentiment_counts.plot(kind="bar", color=["green", "orange", "red"], ax=ax)
#             ax.set_xlabel("Sentiment")
#             ax.set_ylabel("Count")
#             ax.grid(axis="y", linestyle="--", alpha=0.7)
#             st.pyplot(fig)
#     with col2:
#         with st.expander("Pie Chart"):
#             fig, ax = plt.subplots(figsize=(5, 3))
#             ax.pie(
#                 sentiment_counts,
#                 labels=sentiment_counts.index,
#                 autopct="%1.1f%%",
#                 startangle=90,
#                 colors=["#2ca02c", "#ff7f0e", "#d62728"],
#             )
#             ax.axis("equal")
#             st.pyplot(fig)


# def plot_trend_over_time(df):
#     if "date" not in df.columns:
#         st.info("No 'date' column found.")
#         return

#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
#     df = df.dropna(subset=["date"])

#     if df.empty:
#         st.info("No valid date data available.")
#         return

    # st.markdown('<div class="subsection-header">Tweet Trend Over Time</div>', unsafe_allow_html=True)
    # with st.expander("Tweet Frequency Over Time"):
    #     trend_data = df.groupby(df['date'].dt.date).size().reset_index(name='Tweet Count')
    #     fig_trend = px.line(trend_data, x='date', y='Tweet Count', title=f"Tweet Frequency Over Time for '{person}'")
    #     st.plotly_chart(fig_trend, use_container_width=True)

    # if 'Sentiment' in df.columns:
    #     with st.expander("Daily Tweet Sentiment Distribution"):
    #         trend_sentiment = df.groupby([df['date'].dt.date, 'Sentiment']).size().reset_index(name='Count')
    #         fig_stack = px.area(trend_sentiment, x='date', y='Count', color='Sentiment')
    #         st.plotly_chart(fig_stack, use_container_width=True)

    # -------- Divider --------
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.markdown("<hr>", unsafe_allow_html=True)
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")

def trend_duration_prediction():
    st.markdown(
        '<div class="section-header">Trend Duration Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div style='font-size:16px; margin-bottom:20px;'>
    Predict how long a trend will last based on its characteristics and current engagement.
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Trend Duration Predictor", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            trend_text = st.text_area(
                "Describe the trend:",
                placeholder="e.g., 'New AI tool for content creation gaining popularity among marketers'",
                height=100,
            )

            col1a, col1b, col1c = st.columns(3)
            with col1a:
                category = st.selectbox(
                    "Category",
                    [
                        "Technology",
                        "Fashion",
                        "Business",
                        "Social Media",
                        "Health",
                        "Sports",
                        "Politics",
                        "Other",
                    ],
                )
            with col1b:
                current_engagement = st.selectbox(
                    "Current Engagement", ["Low", "Medium", "High", "Viral"]
                )
            with col1c:
                confidence_threshold = st.slider("Confidence Level", 70, 95, 80)

        with col2:
            st.markdown("<div style='margin-top:35px;'></div>", unsafe_allow_html=True)
            if st.button("Predict Duration", key="predict_duration"):
                if trend_text:
                    with st.spinner("Analyzing trend..."):
                        # Mock prediction - in a real app you'd use a proper model
                        base_duration = {
                            "Technology": 45,
                            "Fashion": 30,
                            "Business": 60,
                            "Social Media": 20,
                            "Health": 50,
                            "Sports": 25,
                            "Politics": 40,
                            "Other": 35,
                        }.get(category, 30)

                        engagement_multiplier = {
                            "Low": 0.7,
                            "Medium": 1.0,
                            "High": 1.5,
                            "Viral": 2.0,
                        }.get(current_engagement, 1.0)

                        # Simple calculation based on text length
                        text_length_factor = min(1.0, len(trend_text) / 200)

                        predicted_duration = (
                            base_duration
                            * engagement_multiplier
                            * (1 + text_length_factor)
                        )
                        confidence = confidence_threshold / 100

                        # Display results
                        st.success("Prediction complete!")

                        col_result1, col_result2 = st.columns(2)
                        with col_result1:
                            st.metric(
                                "Predicted Duration", f"{int(predicted_duration)} days"
                            )
                        with col_result2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")

                        # Timeline visualization
                        st.markdown("**Trend Timeline Projection**")
                        timeline_df = pd.DataFrame(
                            {
                                "Phase": [
                                    "Emerging",
                                    "Growing",
                                    "Peak",
                                    "Declining",
                                    "Fading",
                                ],
                                "Days": [
                                    0,
                                    predicted_duration * 0.3,
                                    predicted_duration * 0.5,
                                    predicted_duration * 0.8,
                                    predicted_duration,
                                ],
                                "Interest": [10, 60, 100, 40, 5],
                            }
                        )

                        fig = px.line(
                            timeline_df,
                            x="Days",
                            y="Interest",
                            title="Projected Trend Lifecycle",
                            markers=True,
                            line_shape="spline",
                        )
                        fig.update_traces(line=dict(width=4))
                        st.plotly_chart(fig, use_container_width=True)

                        # Key factors
                        st.markdown("**Key Influencing Factors**")
                        factors = {
                            "Category Impact": f"{category} (Base: {base_duration} days)",
                            "Current Engagement": f"{current_engagement} (x{engagement_multiplier})",
                            "Content Richness": f"{len(trend_text)} chars ({text_length_factor*100:.0f}% impact)",
                            "Sentiment": (
                                "Positive"
                                if "good" in trend_text.lower()
                                or "great" in trend_text.lower()
                                else "Neutral"
                            ),
                        }

                        for factor, value in factors.items():
                            st.write(f"🔹 **{factor}**: {value}")
                else:
                    st.error("Please describe the trend to get a prediction")


def load_text_generator():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, gpt2_model

tokenizer, gpt2_model = load_text_generator()

def generate_overview(trend_name, max_length=100):
    prompt = f"Provide a detailed explanation about the trending topic '{trend_name}'.\nExplanation:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.replace(prompt, "").strip()


@st.cache_resource
def get_simulator():
    return DeepfakeTrendSimulator()


class DeepfakeTrendSimulator:
    def __init__(self):
        self.text_generator = pipeline("text-generation", model="gpt2")
        self.influencers = self._generate_influencers(50)

    def _generate_influencers(self, count):
        types = ["Celebrity", "Journalist", "Politician", "Expert", "Meme Page"]
        return [
            {
                "handle": "@" + fake.user_name(),
                "type": random.choice(types),
                "followers": int(np.random.lognormal(10, 2)),
                "credulity": random.uniform(0, 1),
                "reach_multiplier": random.uniform(0.5, 3),
            }
            for _ in range(count)
        ]

    def get_twitter_style(self, handle, count=5):
        """Fetch real tweets for style analysis"""
        conn = http.client.HTTPSConnection("twitter241.p.rapidapi.com")
        try:
            conn.request(
                "GET",
                f"/search-v2?type=Top&count={count}&query={handle.replace('@', '')}",
                headers={
                    "x-rapidapi-key": "9a94713416mshe8ac12056097737p1c5799jsn17513c35a462",
                    "x-rapidapi-host": "twitter241.p.rapidapi.com",
                },
            )
            res = conn.getresponse()
            if res.status == 200:
                data = json.loads(res.read().decode("utf-8"))
                return [t["text"] for t in data.get("data", {}).get("tweets", [])]
            return None
        finally:
            conn.close()

    def generate_tweet(self, seed_text, mimic_handle=None):
        """Generate tweet with optional style imitation"""
        if mimic_handle:
            real_tweets = self.get_twitter_style(mimic_handle)
            if real_tweets:
                prompt = (
                    f"Generate a tweet in this style:\n"
                    + "\n".join(real_tweets)
                    + f"\n\nAbout: {seed_text}"
                )
            else:
                prompt = f"Tweet in the style of {mimic_handle}: {seed_text}"
        else:
            prompt = f"Viral tweet about: {seed_text}"

        generated = self.text_generator(
            prompt, max_length=280, num_return_sequences=1, do_sample=True
        )
        return generated[0]["generated_text"]

    def simulate_spread(self, fake_tweet, hours=24):
        """Simulate trend propagation"""
        timeline = []
        current_reach = 1

        for hour in range(hours):
            hour_events = []

            # Influencer engagement
            for influencer in self.influencers:
                if random.random() < (influencer["credulity"] * 0.01):
                    new_reach = int(
                        influencer["followers"] * influencer["reach_multiplier"] * 0.001
                    )
                    current_reach += new_reach
                    hour_events.append(
                        {
                            "hour": hour,
                            "account": influencer["handle"],
                            "type": influencer["type"],
                            "action": random.choice(["retweet", "quote tweet"]),
                            "reach_added": new_reach,
                        }
                    )

            # Media pickup
            if current_reach > 10000 and random.random() < 0.2:
                outlet = random.choice(["CNN", "Fox News", "BuzzFeed"])
                current_reach *= 2
                hour_events.append(
                    {
                        "hour": hour,
                        "account": outlet,
                        "type": "Media",
                        "action": "reported",
                        "reach_added": current_reach,
                    }
                )

            # Fact-checking
            if hour > 4 and random.random() < (current_reach / 1000000):
                debunker = random.choice(["@CommunityNotes", "@Snopes"])
                hour_events.append(
                    {
                        "hour": hour,
                        "account": debunker,
                        "type": "Fact-checker",
                        "action": "debunked",
                        "reach_added": -current_reach * 0.5,
                    }
                )
                current_reach *= 0.5

            timeline.append(
                {"hour": hour, "total_reach": current_reach, "events": hour_events}
            )

        return timeline


def extract_tweets_from_response(data):
    tweets = []
    try:
        instructions = data["result"]["timeline"]["instructions"]
        for instruction in instructions:
            for entry in instruction.get("entries", []):
                content = entry.get("content", {})
                if content.get("entryType") == "TimelineTimelineItem":
                    tweet_result = (
                        content.get("itemContent", {})
                        .get("tweet_results", {})
                        .get("result", {})
                    )
                    legacy = tweet_result.get("legacy", {})
                    user_info = (
                        tweet_result.get("core", {})
                        .get("user_results", {})
                        .get("result", {})
                        .get("legacy", {})
                    )

                    tweet_text = legacy.get("full_text", "No text available")
                    username = user_info.get("screen_name", "unknown")

                    tweets.append((username, tweet_text))
    except Exception as e:
        print(f"[Parser Error] {e}")
    return tweets


def main():

    model_xgboost = joblib.load("xgboost_trending.pkl")

    # --- UI Header ---
    st.markdown(
        '<div class="section-header">Trend Prediction</div>', unsafe_allow_html=True
    )

    # --- User Input ---
    username = st.text_input("Enter Twitter username (without @):", value="elonmusk")
    tweet_limit = 5

    # --- Clear Tweets Button ---
    col1, col2 = st.columns([3, 1])

    with col1:
        fetch_clicked = st.button("Fetch Latest Tweets")

    with col2:
        clear_clicked = st.button("Clear Tweets")

    if fetch_clicked:
        st.session_state.pop("fetched_tweets", None)
        try:
            # Get user ID
            conn_user = http.client.HTTPSConnection("twitterr.p.rapidapi.com")
            headers_user = {
                "x-rapidapi-key": "55f56f17bemsh27eb6cc653eab4dp1395fdjsn2540786151af",
                "x-rapidapi-host": "twitterr.p.rapidapi.com",
            }
            conn_user.request(
                "GET", f"/twitter/user/info?userName={username}", headers=headers_user
            )
            user_data = json.loads(conn_user.getresponse().read().decode("utf-8"))
            user_id = user_data["data"]["id"]

            # Get tweets
            conn_tweet = http.client.HTTPSConnection("twitter241.p.rapidapi.com")
            headers_tweet = {
                "x-rapidapi-key": "9a94713416mshe8ac12056097737p1c5799jsn17513c35a462",
                "x-rapidapi-host": "twitter241.p.rapidapi.com",
            }
            conn_tweet.request(
                "GET",
                f"/user-tweets?user={user_id}&count={tweet_limit}",
                headers=headers_tweet,
            )
            tweet_data = json.loads(conn_tweet.getresponse().read().decode("utf-8"))

            instructions = (
                tweet_data.get("result", {}).get("timeline", {}).get("instructions", [])
            )

            tweets = []
            for instruction in instructions:
                if instruction.get("type") in [
                    "TimelineAddEntries",
                    "TimelinePinEntry",
                ]:
                    entries = (
                        instruction.get("entries", [])
                        if instruction.get("type") == "TimelineAddEntries"
                        else [instruction.get("entry", {})]
                    )
                    for entry in entries:
                        content = entry.get("content", {})
                        item_content = content.get("itemContent", {})
                        tweet_results = item_content.get("tweet_results", {}).get(
                            "result", {}
                        )
                        legacy = tweet_results.get("legacy", {})
                        views = tweet_results.get("views", {}).get("count", 1000)

                        if legacy.get("full_text"):
                            tweets.append(
                                {
                                    "text": legacy["full_text"],
                                    "likes": legacy.get("favorite_count", 0),
                                    "retweets": legacy.get("retweet_count", 0),
                                    "replies": legacy.get("reply_count", 0),
                                    "views": views,
                                    "created_at": legacy.get("created_at"),
                                }
                            )

            if tweets:
                st.session_state["fetched_tweets"] = tweets
                st.success(f"{len(tweets)} Tweets fetched successfully!")

        except Exception as e:
            st.error(f"Error fetching tweet: {e}")

    # --- Dropdown + Auto-fill ---
    auto_filled = {}

    if "fetched_tweets" in st.session_state:
        tweets = st.session_state["fetched_tweets"]

        tweet_options = {f"{i+1}: {t['text'][:50]}...": t for i, t in enumerate(tweets)}
        selected_option = st.selectbox(
            "Select a tweet to analyze", list(tweet_options.keys())
        )
        selected_tweet = tweet_options[selected_option]

        st.markdown(f"**Selected Tweet:** {selected_tweet['text']}")

        created_at = pd.to_datetime(selected_tweet["created_at"])
        has_hashtag = 1 if "#" in selected_tweet["text"] else 0

        auto_filled = {
            "platform": "Twitter",
            "content_type": "Post/Tweet",
            "region": "India",
            "views": (
                int(selected_tweet["views"])
                if str(selected_tweet["views"]).isdigit()
                else 0
            ),
            "likes": (
                int(selected_tweet["likes"])
                if str(selected_tweet["likes"]).isdigit()
                else 0
            ),
            "shares": (
                int(selected_tweet["retweets"])
                if str(selected_tweet["retweets"]).isdigit()
                else 0
            ),
            "comments": (
                int(selected_tweet["replies"])
                if str(selected_tweet["replies"]).isdigit()
                else 0
            ),
            "has_hashtag": bool(has_hashtag),
            "post_date": created_at.date(),
        }

    # --- Manual Entry ---
    st.markdown("---")
    st.subheader("Manually enter post details")

    platform = st.selectbox(
        "Platform",
        ["Instagram", "Twitter", "TikTok", "YouTube"],
        index=["Instagram", "Twitter", "TikTok", "YouTube"].index(
            auto_filled.get("platform", "Twitter")
        ),
    )

    content_type = st.selectbox(
        "Content Type",
        ["Post/Tweet", "Video", "Shorts", "Story"],
        index=["Post/Tweet", "Video", "Shorts", "Story"].index(
            auto_filled.get("content_type", "Post/Tweet")
        ),
    )

    region = st.selectbox(
        "Region",
        ["India", "USA", "UK", "Brazil", "Australia", "Other"],
        index=["India", "USA", "UK", "Brazil", "Australia", "Other"].index(
            auto_filled.get("region", "India")
        ),
    )

    views = st.number_input(
        "Views", min_value=0, step=1000, value=auto_filled.get("views", 0)
    )
    likes = st.number_input(
        "Likes", min_value=0, step=100, value=auto_filled.get("likes", 0)
    )
    shares = st.number_input(
        "Shares", min_value=0, step=100, value=auto_filled.get("shares", 0)
    )
    comments = st.number_input(
        "Comments", min_value=0, step=10, value=auto_filled.get("comments", 0)
    )
    has_hashtag = st.checkbox(
        "Has Hashtag?", value=auto_filled.get("has_hashtag", True)
    )
    post_date = st.date_input(
        "Post Date", value=auto_filled.get("post_date", pd.to_datetime("today").date())
    )

    # --- Predict Button ---
    if st.button("Predict"):
        platform_map = {"Instagram": 0, "Twitter": 1, "TikTok": 2, "YouTube": 3}
        content_map = {"Post/Tweet": 0, "Video": 1, "Shorts": 2, "Story": 3}
        region_map = {
            "India": 0,
            "USA": 1,
            "UK": 2,
            "Brazil": 3,
            "Australia": 4,
            "Other": 5,
        }

        df_input = pd.DataFrame(
            [
                {
                    "Platform": platform_map[platform],
                    "Content_Type": content_map[content_type],
                    "Region": region_map[region],
                    "Views": views,
                    "Likes": likes,
                    "Shares": shares,
                    "Comments": comments,
                    "Has_Hashtag": int(has_hashtag),
                    "Engagement_Rate": (likes + shares + comments) / (views + 1),
                    "Post_Year": post_date.year,
                    "Post_Month": post_date.month,
                    "Post_DayOfWeek": post_date.weekday(),
                    "Likes_per_View": likes / (views + 1),
                    "Shares_per_View": shares / (views + 1),
                    "Comments_per_View": comments / (views + 1),
                    "Platform_Region": platform_map[platform] * 10 + region_map[region],
                }
            ]
        )

        prediction = model_xgboost.predict(df_input)[0]
        confidence = model_xgboost.predict_proba(df_input)[0][prediction]

        st.subheader("Prediction Result")
        st.markdown(f"*Trending:* {'✅ Yes' if prediction else '❌ No'}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        if prediction:
            st.success(
                "Your post is likely to trend due to high engagement and good platform-region match."
            )
        else:
            st.warning("This post may not trend. Consider improving these areas:")

        # --- Suggestions ---
        suggestions = []
        if likes / (views + 1) < 0.05:
            suggestions.append(
                "Increase likes through better content appeal or visual design."
            )
        if shares / (views + 1) < 0.01:
            suggestions.append(
                "Boost shareability with emotional hooks or surprising facts."
            )
        if has_hashtag:
            suggestions.append("Limit or avoid hashtags — they may reduce visibility.")

        if post_date.weekday() in [5, 6]:  # Saturday=5, Sunday=6
            suggestions.append("Try posting on weekdays for better algorithmic reach.")

        if suggestions:
            st.markdown("**Suggestions to Improve**")
            for s in suggestions:
                st.write(f"- {s}")

        # --- SHAP Explainability ---
        st.markdown("**Why this prediction? (SHAP Explanation)**")
        explainer = shap.Explainer(model_xgboost)
        shap_values = explainer(df_input)
        shap_id = uuid.uuid4().hex
        shap_html_file = f"shap_{shap_id}.html"
        shap.save_html(
            shap_html_file, shap.plots.force(shap_values[0], matplotlib=False)
        )

        with open(shap_html_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=400, scrolling=True)
        os.remove(shap_html_file)

        # -------- Divider --------
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # Load Data
    # df = load_data("data/India_with_vader.csv")
    # if df.empty:
    #     st.stop()

        # st.markdown('<div class="section-header">Tweet Trend Prediction</div>', unsafe_allow_html=True)
        # trend_tweet = st.text_area("Enter a Tweet to Predict Trend", placeholder="e.g., Huge Black Friday deals on Amazon!")

        # if st.button("Predict Trend"):
        #     if trend_tweet.strip():
        #         will_trend, confidence, explanation, duration = get_trend_prediction(trend_tweet)
        #         if will_trend is None:
        #             st.error("Could not get trend prediction.")
        #         else:
        #             trend_str = "YES! This tweet is likely to TREND" if will_trend else "NO, this tweet is unlikely to trend."
        #             st.success(trend_str)
        #             st.info(f"Confidence: *{round(confidence * 100, 2)}%*")
        #             st.markdown(f'<div class="explanation-box"><strong>Why:</strong> {explanation}</div>', unsafe_allow_html=True)
        #             if will_trend:
        #                 st.markdown(f"Estimated Trending Duration: *{duration} days*")

        #     else:
        #         st.warning("Please enter a tweet first.")

        # # -------- Divider --------
        # st.write("")
        # st.write("")
        # st.write("")
        # st.write("")
        # st.markdown("<hr>", unsafe_allow_html=True)
        # st.write("")
        # st.write("")
        # st.write("")
        # st.write("")

        # # -------- Sentiment Analyzer Section --------
        # st.markdown('<div class="section-header">Sentiment Analyzer</div>', unsafe_allow_html=True)

        # df = load_data("data/India_with_vader.csv")
        # if df.empty:
        #     st.stop()

        # # User input topic/person name
        # topic_input = st.text_input("Enter a Topic or Name to Analyze Sentiment", placeholder="e.g., Elon Musk")

        # sentiment_filter = st.radio("Filter Sentiment", ['All', 'Positive', 'Neutral', 'Negative'], horizontal=True)

        # if topic_input:
        #     filtered_df = df[df['tweet'].str.contains(topic_input, case=False, na=False)]
        #     if sentiment_filter != 'All':
        #         filtered_df = filtered_df[filtered_df['Sentiment'].str.lower() == sentiment_filter.lower()]

        #     if filtered_df.empty:
        #         st.warning("No matching tweets found.")
        #         return

        #     st.markdown(f"### Found {len(filtered_df)} tweets about *{topic_input}*")
        #     plot_sentiment_distribution(filtered_df)

        # Select or enter tweet for sentiment
    #     st.markdown(
    #         '<div class="section-header">DistilBERT Sentiment Classifier</div>',
    #         unsafe_allow_html=True,
    #     )
    #     analyze_mode = st.radio(
    #         "Choose Tweet Input",
    #         ["Select from Dataset", "Enter Your Own"],
    #         horizontal=True,
    #     )

    #     if analyze_mode == "Select from Dataset":
    #         selected_tweet = st.selectbox("Tweet", filtered_df["tweet"].tolist())
    #     else:
    #         selected_tweet = st.text_area("Enter Tweet for Sentiment")

    #     if st.button("Analyze Sentiment"):
    #         if selected_tweet.strip():
    #             analyze_sentiment_with_distilbert(selected_tweet)
    #         else:
    #             st.warning("Please enter or select a tweet.")

    #     st.markdown(
    #         '<div class="section-header">Filtered Tweets</div>', unsafe_allow_html=True
    #     )
    #     st.dataframe(
    #         filtered_df[["username", "tweet", "Sentiment"]].reset_index(drop=True),
    #         height=300,
    #     )

    #     plot_trend_over_time(filtered_df)

    # else:
    #     ()

    # st.markdown(
    #     '<div class="section-header">Filter Tweets</div>', unsafe_allow_html=True
    # )

    # # Sidebar for filters (or use columns)
    # common_names = [
    #     "Kohli",
    #     "Virat Kohli",
    #     "Rohit Sharma",
    #     "Yuzvendra Chahal",
    #     "Jasprit Bumrah",
    #     "Babar Azam",
    #     "Modi",
    #     "Putin Sweden",
    #     "Covid",
    #     "Eminem",
    #     "Alice",
    #     "Harsh V Pant",
    #     "Hansa Mehta",
    #     "Ladakh",
    #     "BestBid ExchangeCoindcx",
    #     "BestBid ExchangeZebpay",
    #     "Millie Book",
    #     "Twitter",
    #     "USDT",
    # ]

    # # Use sidebar or main layout - here main for website feel
    # col_filter1, col_filter2 = st.columns([2, 3])

    # with col_filter1:
    #     person = st.selectbox("Select a person/topic mentioned in tweets", common_names)

    # with col_filter2:
    #     filter_option = st.radio(
    #         "Filter by Sentiment",
    #         ["All", "Positive", "Neutral", "Negative"],
    #         horizontal=True,
    #     )

    # # Filter Data
    # filtered_df = filter_tweets_by_person(df, person)
    # if filtered_df.empty:
    #     st.warning(f"No tweets found mentioning **{person}**.")
    #     st.stop()

    # filtered_df = filter_tweets_by_sentiment(filtered_df, filter_option)
    # if filtered_df.empty:
    #     st.warning(f"No tweets matching the sentiment filter '{filter_option}'.")
    #     st.stop()

    # st.markdown(f"### {len(filtered_df)} Tweets mentioning **{person}**")

    # # Sentiment Distribution Plots
    # plot_sentiment_distribution(filtered_df)

    # # -------- Divider --------
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.markdown("<hr>", unsafe_allow_html=True)
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")

    # st.markdown(
    #     '<div class="section-header">DistilBERT Sentiment Analysis</div>',
    #     unsafe_allow_html=True,
    # )

    # analyze_mode = st.radio(
    #     "Choose Tweet Input Mode",
    #     ["Select from Dataset", "Enter Your Own Tweet"],
    #     horizontal=True,
    # )

    # selected_tweet = ""
    # if analyze_mode == "Select from Dataset":
    #     selected_tweet = st.selectbox(
    #         "Select a Tweet to Analyze", filtered_df["tweet"].tolist()
    #     )
    # else:
    #     selected_tweet = st.text_area("Enter your own tweet for sentiment analysis")

    # if selected_tweet and st.button("Analyze with DistilBERT"):
    #     analyze_sentiment_with_distilbert(selected_tweet)

    # st.markdown(
    #     '<div class="section-header">Filtered Tweets Data</div>', unsafe_allow_html=True
    # )
    # st.dataframe(
    #     filtered_df[["username", "tweet", "Sentiment"]].reset_index(drop=True),
    #     height=300,
    # )

    # plot_trend_over_time(filtered_df)

    #             # -------- Divider --------
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.markdown("<hr>", unsafe_allow_html=True)
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")

    st.markdown(
        '<div class="section-header">Real or Fake User</div>', unsafe_allow_html=True
    )
    st.markdown("""
    ### About this Feature
    This tool analyzes Twitter usernames to estimate the likelihood of being a **bot** using Botometer API. Each user is scored between `0.0` and `1.0`, where:

    | **Bot Score** | **Interpretation**         |
    |---------------|----------------------------|
    | 0.0 - 0.3     | 🟢 Likely Human            |
    | 0.3 - 0.6     | 🟡 Suspicious / Uncertain  |
    | 0.6 - 1.0     | 🔴 Likely Bot              |
        """)
    user_input = st.text_area(
        "Enter Username",
        placeholder="Example: narendramodi , elonmusk , FabrizioRomano",
    )

    if st.button("Check Bot Scores") and user_input:
        # Clean and prepare the usernames
        usernames = [
            uname.strip().lstrip("@")
            for uname in user_input.split(",")
            if uname.strip()
        ]

        payload = json.dumps({"usernames": usernames})

        headers = {
            "x-rapidapi-key": "9a94713416mshe8ac12056097737p1c5799jsn17513c35a462",
            "x-rapidapi-host": "botometer-pro.p.rapidapi.com",
            "Content-Type": "application/json",
        }

        try:
            with st.spinner("Fetching bot scores..."):
                conn = http.client.HTTPSConnection("botometer-pro.p.rapidapi.com")
                conn.request(
                    "POST", "/botometer-x/get_botscores_in_batch", payload, headers
                )

                res = conn.getresponse()
                data = res.read()
                result = json.loads(data.decode("utf-8"))

                parsed = [
                    {
                        "User ID": item.get("user_id", "N/A"),
                        "Username": item.get("username", "N/A"),
                        "Bot Score": round(item.get("bot_score", 0), 3),
                    }
                    for item in result
                ]

            df = pd.DataFrame(parsed)
            st.success("Result:")
            st.dataframe(df)

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

    # | **Bot Score** | **Interpretation**     |
    # | ------------- | ---------------------- |
    # | 0.0 - 0.3     | Likely human           |
    # | 0.3 - 0.6     | Suspicious / uncertain |
    # | 0.6 - 1.0     | Likely bot             |

    # -------- Divider --------
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.markdown(
        '<div class="section-header">Twitter Trend By Location</div>',
        unsafe_allow_html=True,
    )

    locations = {
        "India": 23424848,
        "United States": 23424977,
        "United Kingdom": 23424975,
        "Canada": 23424775,
        "Australia": 23424748,
        "Japan": 23424856,
        "Germany": 23424829,
        "Brazil": 23424768,
    }

    # Dropdown to select a location
    selected_location = st.selectbox("Select a Location", list(locations.keys()))
    selected_woeid = locations[selected_location]

    if st.button("Get Trends"):
        with st.spinner("Fetching Twitter trends..."):
            try:
                conn = http.client.HTTPSConnection("twitter241.p.rapidapi.com")
                headers = {
                    "x-rapidapi-key": "9a94713416mshe8ac12056097737p1c5799jsn17513c35a462",
                    "x-rapidapi-host": "twitter241.p.rapidapi.com",
                }

                endpoint = f"/trends-by-location?woeid={selected_woeid}"
                conn.request("GET", endpoint, headers=headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data.decode("utf-8"))

                trends = json_data["result"][0]["trends"][:5]

                st.subheader(f"Top Trends in {selected_location}")
                for trend in trends:
                    name = trend.get("name", "N/A")
                    volume = trend.get("tweet_volume", "N/A")

                    st.markdown(f"### 🔹 {name}")
                    st.markdown(
                        f"**Tweet Volume**: {volume if volume else 'Not Available'}"
                    )

                    with st.spinner(f"Fetching tweets for '{name}'..."):
                        try:
                            search_conn = http.client.HTTPSConnection(
                                "twitter241.p.rapidapi.com"
                            )
                            query = name.replace("#", "%23").replace(" ", "%20")
                            search_endpoint = (
                                f"/search-v2?type=Top&count=10&query={query}"
                            )
                            search_conn.request("GET", search_endpoint, headers=headers)
                            search_res = search_conn.getresponse()
                            search_data = search_res.read()
                            search_json = json.loads(search_data.decode("utf-8"))

                            tweets = extract_tweets_from_response(search_json)

                            if tweets:
                                for username, text in tweets[:3]:
                                    st.markdown(f"**@{username}**: {text}")
                                    st.markdown("---")
                            else:
                                st.info("No tweets found.")

                        except Exception as tweet_error:
                            st.error(f"Error fetching tweets: {tweet_error}")

            except Exception as e:
                st.error(f"Error: {e}")

    NEWS_API_HOST = "news-api14.p.rapidapi.com"
    NEWS_API_KEY = "9a94713416mshe8ac12056097737p1c5799jsn17513c35a462"

    st.markdown(
        '<div class="section-header">Tweet Deep Analysis: Meaning, Context & Impact</div>',
        unsafe_allow_html=True,
    )

    # Tweet input area (formerly in sidebar)
    tweet_text = st.text_area("Paste tweet text below for analysis:", height=150)

    ANALYSIS_TEMPLATE = """
        ###  Analysis Report

        1. Meaning 
        {meaning}

        2. Context  
        {context}

        3. Why It Matters 
        {contribution}

        4. What It Leads To  
        {impact}
        """

    # Get related news from API
    def get_related_news(query):
        try:
            clean_query = quote(query[:100])
            conn = http.client.HTTPSConnection(NEWS_API_HOST)
            headers = {"x-rapidapi-key": NEWS_API_KEY, "x-rapidapi-host": NEWS_API_HOST}
            path = f"/v2/search?q={clean_query}&language=en&sortBy=relevance"
            conn.request("GET", path, headers=headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            return data.get("articles", [])[:3]
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return []

    # Use GPT-2 to generate analysis
    def generate_analysis(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = gpt2_model.generate(
            inputs.input_ids,
            max_length=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(
            prompt, ""
        )

    # Generate 4-part analysis
    def generate_comprehensive_analysis(text, news_context=""):
        meaning_prompt = (
            f"Explain the literal and implied meaning of this tweet: '{text}'."
        )
        context_prompt = f"Provide historical, cultural or social context for this tweet: '{text}'. {news_context}"
        contribution_prompt = (
            f"Analyze how this tweet contributes to public discourse: '{text}'."
        )
        impact_prompt = (
            f"Predict the potential real-world impact of this tweet: '{text}'."
        )

        return {
            "meaning": generate_analysis(meaning_prompt),
            "context": generate_analysis(context_prompt),
            "contribution": generate_analysis(contribution_prompt),
            "impact": generate_analysis(impact_prompt),
        }

    # Run full analysis
    def run_analysis():
        if not tweet_text:
            st.warning("Please enter tweet text.")
            return

        with st.spinner("Analyzing tweet..."):
            news_articles = get_related_news(tweet_text)
            news_context = (
                "\n".join([f"- {art['title']}" for art in news_articles])
                if news_articles
                else "No recent news context found"
            )
            analysis = generate_comprehensive_analysis(tweet_text, news_context)

            st.subheader("Original Tweet")
            st.write(tweet_text)

            st.subheader("AI-Based Analysis")
            st.markdown(
                ANALYSIS_TEMPLATE.format(
                    meaning=analysis["meaning"],
                    context=analysis["context"],
                    contribution=analysis["contribution"],
                    impact=analysis["impact"],
                )
            )

            if news_articles:
                st.subheader("Related News")
                for article in news_articles:
                    with st.expander(article["title"]):
                        st.write(article.get("description", "No description available"))
                        if article.get("url"):
                            st.markdown(f"[Read more]({article['url']})")

    # Analyze button triggers analysis
    if st.button("Analyze Tweet"):
        run_analysis()
    else:
        ()

    # -------- Divider --------
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.markdown(
        '<div class="section-header">Deepfake Trend</div>', unsafe_allow_html=True
    )
    simulator = get_simulator()

    st.header("Simulation Controls")
    col1, col2 = st.columns([2, 1])

    with col1:
        seed_text = st.text_area(
            "Trend seed:",
            "Breaking: Study finds chocolate helps weight loss",
            help="Base content for the fake trend",
        )
        mimic_handle = st.text_input(
            "Mimic account style (optional):", "", help="e.g. @elonmusk"
        )

    with col2:
        simulation_hours = st.slider("Simulation duration (hours):", 6, 72, 24)
        run_simulation = st.button("Launch Simulation")

    if run_simulation:
        with st.spinner("Generating scenario..."):
            # Generate fake content
            fake_tweet = simulator.generate_tweet(
                seed_text, mimic_handle if mimic_handle else None
            )

            # Simulate spread
            timeline = simulator.simulate_spread(fake_tweet, simulation_hours)
            events = [e for t in timeline for e in t["events"]]

            # Calculate metrics
            final_reach = timeline[-1]["total_reach"]
            peak_reach = max(t["total_reach"] for t in timeline)
            debunk_time = next(
                (e["hour"] for e in events if e["type"] == "Fact-checker"), None
            )

            # Top amplifiers
            amplifier_impact = {}
            for e in events:
                if e["type"] in ["Celebrity", "Journalist", "Politician"]:
                    amplifier_impact[e["account"]] = (
                        amplifier_impact.get(e["account"], 0) + e["reach_added"]
                    )
            top_amplifiers = sorted(
                amplifier_impact.items(), key=lambda x: x[1], reverse=True
            )[:3]

        # Results display
        st.subheader("Generated Content")
        with st.expander("View synthetic tweet"):
            st.code(fake_tweet, language="text")
            st.caption("This is AI-generated content - not real")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak Reach", f"{peak_reach:,}")
        col2.metric("Final Reach", f"{final_reach:,}")
        col3.metric(
            "Debunk Time", f"{debunk_time} hours" if debunk_time else "Not debunked"
        )

        # Visualization
        st.subheader("Spread Analysis")
        fig, ax = plt.subplots(figsize=(10, 4))
        hours = [t["hour"] for t in timeline]
        reach = [t["total_reach"] for t in timeline]
        ax.plot(hours, reach, marker="o", color="#1DA1F2")

        if debunk_time:
            ax.axvline(x=debunk_time, color="red", linestyle="--", label="Debunked")
            ax.text(
                debunk_time, max(reach) * 0.8, "Fact-checked", rotation=90, color="red"
            )

        ax.set_title("Estimated Reach Over Time")
        ax.set_xlabel("Hours After Posting")
        ax.set_ylabel("Potential Reach")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        # Event log
        st.subheader("Key Events Timeline")
        event_df = pd.DataFrame(
            [
                {
                    "Hour": e["hour"],
                    "Account": e["account"],
                    "Type": e["type"],
                    "Action": e["action"],
                    "Reach Impact": f"{e['reach_added']:,}",
                }
                for e in events
            ]
        )
        st.dataframe(
            event_df.style.applymap(
                lambda x: (
                    "color: red"
                    if x == "Fact-checker"
                    else ("color: green" if x == "Media" else "")
                ),
                subset=["Type"],
            ),
            hide_index=True,
            use_container_width=True,
        )

        # Amplifier analysis
        st.subheader("Top Amplifiers")
        if top_amplifiers:
            for account, impact in top_amplifiers:
                st.progress(
                    min(impact / peak_reach, 1.0), text=f"{account} (+{impact:,} reach)"
                )
        else:
            st.info("No significant amplifiers detected")

        # Media coverage
        st.subheader("Media Coverage")
        if media_events := [e for e in events if e["type"] == "Media"]:
            for e in media_events:
                st.write(
                    f"**{e['account']}** reported at hour {e['hour']} (+{e['reach_added']:,} reach)"
                )
        else:
            st.info("No media coverage generated")

    #         # -------- Divider --------
    st.write("")
    # st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    # st.write("")
    st.write("")
    # Load best model (e.g., XGBoost for SHAP explainability)

    st.markdown('<div class="footer">Geetesh Kankonkar | Rishikesh Naik | Shubham Kapolkar</div>', unsafe_allow_html=True)    
    # st.markdown(footer, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
