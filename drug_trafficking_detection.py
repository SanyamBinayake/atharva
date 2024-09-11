import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import random

# Set a consistent color palette
color_palette = px.colors.qualitative.Bold

# Initialize Faker
fake = Faker()
Faker.seed(0)

# Set the number of records to generate
num_users = 1000
num_channels = 100
num_messages = 5000

# Custom list of device types
device_types = ['iPhone', 'Android', 'Windows PC', 'Mac', 'Linux', 'Tablet']

# List of drug-related keywords
drug_keywords = ['cocaine', 'heroin', 'meth', 'mdma', 'lsd', 'weed', 'pills', 'crack', 'ketamine', 'opioids', 'drug', 'drugs']

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
    
    messages['is_drug_related'] = messages['content'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in drug_keywords) else 0)
    
    # Merge with channels_df to get platform information
    messages = messages.merge(channels_df[['channel_id', 'platform']], on='channel_id', how='left')
    
    return messages

# Train ML model
@st.cache_resource
def train_model(messages_df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(messages_df['content'])
    y = messages_df['is_drug_related']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, vectorizer, classification_report(y_test, y_pred, output_dict=True)

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
        st.markdown("""
        This project aims to develop a machine-learning model to identify drug trafficking activity on messaging platforms like Telegram, WhatsApp, and Instagram.
        """)

    with tab1:
        st.header("Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users_df))
        with col2:
            st.metric("Total Channels", len(channels_df))
        with col3:
            st.metric("Total Messages", len(messages_df))
        with col4:
            st.metric("Suspected Drug-Related Messages", messages_df['is_drug_related'].sum())
        
        # Activity over time
        fig_activity = px.line(messages_df.groupby(messages_df['timestamp'].dt.date).size().reset_index(name='count'), 
                               x='timestamp', y='count', title="Activity Over Time")
        st.plotly_chart(fig_activity, use_container_width=True)

    with tab2:
        st.header("User Profiles")
        
        # Device distribution
       fig_devices = px.pie(
        values=users_df['device_info'].value_counts().values,  # Pie chart values
        names=users_df['device_info'].value_counts().index,    # Pie chart labels
        title="Device Distribution"
        )

        st.plotly_chart(fig_devices, use_container_width=True)

        st.dataframe(users_df)

    with tab3:
        st.header("Channels/Groups")
        st.dataframe(channels_df)

    with tab4:
        st.header("Messages")
        st.dataframe(messages_df)

    with tab5:
        st.header("Machine Learning Insights")
        st.subheader("Model Performance")
        st.json(classification_report)

if __name__ == '__main__':
    main()
