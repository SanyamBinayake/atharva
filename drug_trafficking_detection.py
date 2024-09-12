import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import random

# Initialize Faker
fake = Faker()
Faker.seed(0) 

# Set the number of records to generate
num_users = 1000
num_channels = 100
num_messages = 5000

# Custom list of device types
device_types = ['iPhone', 'Android', 'Windows PC', 'Mac', 'Linux', 'Tablet']

# Load drug keywords from CSV file
drug_keywords_df = pd.read_csv('drug_keywords.csv')
drug_keywords = drug_keywords_df['keyword'].tolist()

# Generate User Profiles
@st.cache_data
def generate_user_data():
    return pd.DataFrame({
        'user_id': range(1, num_users + 1),
        'username': [fake.user_name() for _ in range(num_users)],
        'ip_address': [fake.ipv4() for _ in range(num_users)],
        'device_info': [random.choice(device_types) for _ in range(num_users)],
        'mobile_number': [fake.phone_number() for _ in range(num_users)],
        'account_age_days': [random.randint(1, 1000) for _ in range(num_users)],
        'avg_daily_messages': [random.randint(1, 100) for _ in range(num_users)],
    })

# Generate Channels/Groups/Handles
@st.cache_data
def generate_channel_data():
    platforms = ['Telegram', 'WhatsApp', 'Instagram']
    return pd.DataFrame({
        'channel_id': range(1, num_channels + 1),
        'name': [fake.word() + random.choice(['_drugs', '_shop', '_market', '_group', '_chat']) for _ in range(num_channels)],
        'creation_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_channels)],
        'description': [fake.sentence() for _ in range(num_channels)],
        'members_count': [random.randint(50, 10000) for _ in range(num_channels)],
        'activity_level': [random.choice(['low', 'medium', 'high']) for _ in range(num_channels)],
        'is_private': [random.choice([True, False]) for _ in range(num_channels)],
        'platform': [random.choice(platforms) for _ in range(num_channels)],
    })

# Generate Messages/Posts
@st.cache_data
def generate_message_data(users_df, channels_df):
    def generate_content():
        if random.random() < 0.2:  # 20% chance of drug-related content
            return fake.sentence() + ' ' + random.choice(drug_keywords) + ' ' + random.choice([' Buy now!', ' Available!', ' DM for details.'])
        else:
            return fake.sentence()

    messages = pd.DataFrame({
        'message_id': range(1, num_messages + 1),
        'timestamp': [fake.date_time_between(start_date='-2y', end_date='now') for _ in range(num_messages)],
        'sender_id': [random.choice(users_df['user_id']) for _ in range(num_messages)],
        'channel_id': [random.choice(channels_df['channel_id']) for _ in range(num_messages)],
        'content': [generate_content() for _ in range(num_messages)],
        'engagement': [random.randint(0, 100) for _ in range(num_messages)],
    })
    
    messages['is_drug_related'] = messages['content'].apply(lambda x: 1 if any(keyword.lower() in x.lower() for keyword in drug_keywords) else 0)
    
    # Merge with channels_df to get platform information
    messages = messages.merge(channels_df[['channel_id', 'platform']], on='channel_id', how='left')
    
    return messages

# Train ML model
@st.cache_resource
def train_model(messages_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(messages_df['content'])
    y = messages_df['is_drug_related']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, vectorizer, classification_report(y_test, y_pred)

# Main app
def main():
    st.set_page_config(page_title="ML-Enhanced Drug Trafficking Detection Dashboard", layout="wide")

    st.title("ML-Enhanced Drug Trafficking Detection on Messaging Platforms")

    # Sidebar
    st.sidebar.title("Settings")
    platform = st.sidebar.selectbox("Select Platform", ["All", "Telegram", "WhatsApp", "Instagram"])

    date_range = st.sidebar.date_input("Select Date Range", 
                                    [pd.Timestamp.now() - pd.Timedelta(days=365), pd.Timestamp.now()],
                                    min_value=pd.Timestamp.now() - pd.Timedelta(days=730),
                                    max_value=pd.Timestamp.now())
    
    # Generate data
    users_df = generate_user_data()
    channels_df = generate_channel_data()
    messages_df = generate_message_data(users_df, channels_df)
    
    # Train ML model
    model, vectorizer, classification_report = train_model(messages_df)
    
    # Filter data based on sidebar inputs
    if platform != "All":
        channels_df = channels_df[channels_df['platform'] == platform]
        messages_df = messages_df[messages_df['platform'] == platform]

    messages_df = messages_df[(messages_df['timestamp'].dt.date >= date_range[0]) & (messages_df['timestamp'].dt.date <= date_range[1])]

    # Main content
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Problem Statement", "Overview", "User Profiles", "Channels/Groups", "Messages", "ML Insights"])

    with tab0:
        st.header("Problem Statement")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f"""
                <div style='background-color: #E6F3FF; padding: 20px; border-radius: 10px;'>
                    <h2>Problem</h2>
                    <p>Identifying drug-related messages on messaging platforms can be challenging due to the vast amount of data and diverse communication styles. Our solution uses machine learning to enhance the detection process, providing a more efficient way to flag and review potential drug-related content.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style='background-color: #E6F3FF; padding: 20px; border-radius: 10px;'>
                    <h2>Objectives</h2>
                    <ul>
                        <li>Utilize machine learning to classify messages as drug-related or not.</li>
                        <li>Visualize key metrics and trends in drug-related messaging.</li>
                        <li>Provide actionable insights to enhance monitoring and enforcement.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab1:
        st.header("Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("User Profiles")
            st.write(users_df.describe())

        with col2:
            st.subheader("Channel/Group Information")
            st.write(channels_df.describe())

    with tab2:
        st.header("User Profiles")
        
        st.write(users_df.head())

    with tab3:
        st.header("Channels/Groups")

        st.write(channels_df.head())

    with tab4:
        st.header("Messages")
        
        st.write(messages_df.head())

    with tab5:
        st.header("ML Insights")

        st.write("Classification Report:")
        st.text(classification_report)
        
        st.subheader("Test Message Classification")
        user_input = st.text_area("Enter a message to classify:", "")
        
        if user_input:
            prediction, probability = test_prediction(user_input, model, vectorizer)
            st.write(f"Prediction: {'Drug-Related' if prediction == 1 else 'Not Drug-Related'}")
            st.write(f"Probability of being drug-related: {probability:.2f}")

if __name__ == "__main__":
    main()
