import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px

# Page config for wide layout
st.set_page_config(layout="wide", page_title="Trendrrr")

st.title("Twitter Trend Analyzer")
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("""
    <div style='display: flex; align-items: center; height: 100%;'>
    <div style='font-size:48px; color:#ffffff; line-height:1.3; padding-top: 50px; color: #1A93DE; font-weight: bold;'>
        Innovate Insights Shaping Social Media Marketing. What's Trending???
        <div style='margin-top: 30px; font-size: 22px; font-weight: bold; color: #ffffff; padding-top: 5%; padding-bottom: 5%'>Want to know more?</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Get Started"):
        st.success("Let's go! Scroll down to explore the features. 🚀")

with col2:
    st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>", unsafe_allow_html=True)
    st.image("assets/image.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Custom CSS for some spacing and styling
st.markdown("""
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
        font-size: 28px;
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
""", unsafe_allow_html=True)

@st.cache_data
def load_data(filepath):
    try:
        data = pd.read_csv(filepath, engine='python').dropna(subset=['tweet'])
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def filter_tweets_by_person(df, person):
    if person:
        return df[df['tweet'].str.contains(person, case=False, na=False)]
    return df

def filter_tweets_by_sentiment(df, sentiment):
    if sentiment.lower() != 'all' and 'Sentiment' in df.columns:
        return df[df['Sentiment'].str.lower() == sentiment.lower()]
    return df

def plot_sentiment_distribution(df):
    sentiment_counts = df['Sentiment'].value_counts()
    st.markdown('<div class="subsection-header">Sentiment Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Bar Chart"):
            fig_bar, ax_bar = plt.subplots(figsize=(5,3))
            sentiment_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax_bar)
            ax_bar.set_xlabel("Sentiment")
            ax_bar.set_ylabel("Count")
            ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig_bar)

    with col2:
        with st.expander("Pie Chart"):
            fig_pie, ax_pie = plt.subplots(figsize=(5,3))
            ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#2ca02c','#ff7f0e','#d62728'])
            ax_pie.axis('equal')
            st.pyplot(fig_pie)

def plot_trend_over_time(df, person):
    if 'date' not in df.columns:
        st.info("No 'date' column found. Cannot show trend over time.")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if df.empty:
        st.info("No valid date data available for trend analysis.")
        return

    st.markdown('<div class="subsection-header">Tweet Trend Over Time</div>', unsafe_allow_html=True)

    with st.expander("Tweet Frequency Over Time"):
        trend_data = df.groupby(df['date'].dt.date).size().reset_index(name='Tweet Count')
        fig_trend = px.line(trend_data, x='date', y='Tweet Count',
                            title=f"Tweet Frequency Over Time for '{person}'",
                            width=700, height=350)
        st.plotly_chart(fig_trend, use_container_width=True)

    if 'Sentiment' in df.columns:
        with st.expander("Daily Tweet Sentiment Distribution"):
            trend_sentiment = df.groupby([df['date'].dt.date, 'Sentiment']).size().reset_index(name='Count')
            fig_stack = px.area(trend_sentiment, x='date', y='Count', color='Sentiment',
                                title="Daily Tweet Sentiment Distribution",
                                line_group='Sentiment',
                                width=700, height=350)
            st.plotly_chart(fig_stack, use_container_width=True)

def analyze_sentiment_with_distilbert(tweet):
    try:
        with st.spinner('Analyzing sentiment with DistilBERT...'):
            response = requests.post("http://localhost:5000/predict", json={"text": tweet})
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Sentiment (DistilBERT): **{result.get('sentiment', 'Unknown')}**")
            else:
                st.error(f"Prediction API returned status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to prediction API: {e}")

def trend_duration_prediction():
    st.markdown("---")
    st.markdown('<div class="section-header">Trend Duration Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:16px; margin-bottom:20px;'>
    Predict how long a trend will last based on its characteristics and current engagement.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Trend Duration Predictor", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            trend_text = st.text_area("Describe the trend:", 
                                    placeholder="e.g., 'New AI tool for content creation gaining popularity among marketers'",
                                    height=100)
            
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                category = st.selectbox("Category", 
                                      ["Technology", "Fashion", "Business", "Social Media", "Health", "Sports", "Politics", "Other"])
            with col1b:
                current_engagement = st.selectbox("Current Engagement", 
                                                ["Low", "Medium", "High", "Viral"])
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
                            "Other": 35
                        }.get(category, 30)
                        
                        engagement_multiplier = {
                            "Low": 0.7,
                            "Medium": 1.0,
                            "High": 1.5,
                            "Viral": 2.0
                        }.get(current_engagement, 1.0)
                        
                        # Simple calculation based on text length
                        text_length_factor = min(1.0, len(trend_text) / 200)
                        
                        predicted_duration = base_duration * engagement_multiplier * (1 + text_length_factor)
                        confidence = confidence_threshold / 100
                        
                        # Display results
                        st.success("Prediction complete!")
                        
                        col_result1, col_result2 = st.columns(2)
                        with col_result1:
                            st.metric("Predicted Duration", f"{int(predicted_duration)} days")
                        with col_result2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Timeline visualization
                        st.markdown("**Trend Timeline Projection**")
                        timeline_df = pd.DataFrame({
                            'Phase': ['Emerging', 'Growing', 'Peak', 'Declining', 'Fading'],
                            'Days': [0, predicted_duration*0.3, predicted_duration*0.5, predicted_duration*0.8, predicted_duration],
                            'Interest': [10, 60, 100, 40, 5]
                        })
                        
                        fig = px.line(timeline_df, x='Days', y='Interest', 
                                     title="Projected Trend Lifecycle",
                                     markers=True,
                                     line_shape='spline')
                        fig.update_traces(line=dict(width=4))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Key factors
                        st.markdown("**Key Influencing Factors**")
                        factors = {
                            "Category Impact": f"{category} (Base: {base_duration} days)",
                            "Current Engagement": f"{current_engagement} (x{engagement_multiplier})",
                            "Content Richness": f"{len(trend_text)} chars ({text_length_factor*100:.0f}% impact)",
                            "Sentiment": "Positive" if "good" in trend_text.lower() or "great" in trend_text.lower() else "Neutral"
                        }
                        
                        for factor, value in factors.items():
                            st.write(f"🔹 **{factor}**: {value}")
                else:
                    st.error("Please describe the trend to get a prediction")

def main():
    # Load Data
    df = load_data("data/India_with_vader.csv")
    if df.empty:
        st.stop()

    st.markdown('<div class="section-header">Filter Tweets</div>', unsafe_allow_html=True)

    # Sidebar for filters (or use columns)
    common_names = [
        'Kohli', 'Virat Kohli', 'Rohit Sharma', 'Yuzvendra Chahal', 'Jasprit Bumrah', 'Babar Azam',
        'Modi', 'Putin Sweden', 'Covid', 'Eminem', 'Alice', 'Harsh V Pant', 'Hansa Mehta', 'Ladakh',
        'BestBid ExchangeCoindcx', 'BestBid ExchangeZebpay', 'Millie Book', 'Twitter', 'USDT'
    ]

    # Use sidebar or main layout - here main for website feel
    col_filter1, col_filter2 = st.columns([2, 3])

    with col_filter1:
        person = st.selectbox("Select a person/topic mentioned in tweets", common_names)

    with col_filter2:
        filter_option = st.radio("Filter by Sentiment", ['All', 'Positive', 'Neutral', 'Negative'], horizontal=True)

    # Filter Data
    filtered_df = filter_tweets_by_person(df, person)
    if filtered_df.empty:
        st.warning(f"No tweets found mentioning **{person}**.")
        st.stop()

    filtered_df = filter_tweets_by_sentiment(filtered_df, filter_option)
    if filtered_df.empty:
        st.warning(f"No tweets matching the sentiment filter '{filter_option}'.")
        st.stop()

    st.markdown(f"### {len(filtered_df)} Tweets mentioning **{person}**")

    # Sentiment Distribution Plots
    plot_sentiment_distribution(filtered_df)

    st.markdown('<div class="section-header">DistilBERT Sentiment Analysis</div>', unsafe_allow_html=True)

    analyze_mode = st.radio("Choose Tweet Input Mode", ["Select from Dataset", "Enter Your Own Tweet"], horizontal=True)

    selected_tweet = ""
    if analyze_mode == "Select from Dataset":
        selected_tweet = st.selectbox("Select a Tweet to Analyze", filtered_df['tweet'].tolist())
    else:
        selected_tweet = st.text_area("Enter your own tweet for sentiment analysis")

    if selected_tweet and st.button("Analyze with DistilBERT"):
        analyze_sentiment_with_distilbert(selected_tweet)

    st.markdown('<div class="section-header">Filtered Tweets Data</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df[['username', 'tweet', 'Sentiment']].reset_index(drop=True), height=300)

    plot_trend_over_time(filtered_df, person)
    trend_duration_prediction()

    #st.markdown('<div class="footer">Made with ❤️ using Streamlit | Twitter Sentiment Analyzer</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
