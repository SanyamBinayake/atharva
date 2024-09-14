import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Set a consistent color palette
color_palette = px.colors.qualitative.Bold

# Function to load CSV data
@st.cache_data
def load_csv_data(file_path):
    return pd.read_csv(file_path)

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
    
    return model, vectorizer, classification_report(y_test, y_pred)

# Main app
def main():
    st.set_page_config(page_title="ML-Enhanced Drug Trafficking Detection Dashboard", layout="wide")

    st.title("ML-Enhanced Drug Trafficking Detection on Messaging Platforms")

    # Direct file paths in your GitHub repository
    users_path = "users.csv"
    channels_path = "channels.csv"
    messages_path = "messages.csv"

    # Load data
    users_df = load_csv_data(users_path)
    channels_df = load_csv_data(channels_path)
    messages_df = load_csv_data(messages_path)

    # Convert timestamp to datetime
    messages_df['timestamp'] = pd.to_datetime(messages_df['timestamp'])

    # Sidebar
    st.sidebar.title("Settings")
    platform = st.sidebar.selectbox("Select Platform", ["All"] + list(messages_df['platform'].unique()))

    date_range = st.sidebar.date_input("Select Date Range", 
                                    [messages_df['timestamp'].min().date(), messages_df['timestamp'].max().date()],
                                    min_value=messages_df['timestamp'].min().date(),
                                    max_value=messages_df['timestamp'].max().date())
    
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
        # ... [Problem Statement content remains the same]

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
                               x='timestamp', y='count', title="Activity Over Time",
                               labels={'timestamp': 'Date', 'count': 'Number of Messages'},
                               color_discrete_sequence=[color_palette[0]])
        st.plotly_chart(fig_activity, use_container_width=True)
        
        # Drug-related messages over time
        fig_drug_activity = px.line(messages_df[messages_df['is_drug_related'] == 1].groupby(messages_df['timestamp'].dt.date).size().reset_index(name='count'),
                                    x='timestamp', y='count', title="Suspected Drug-Related Messages Over Time",
                                    labels={'timestamp': 'Date', 'count': 'Number of Drug-Related Messages'},
                                    color_discrete_sequence=[color_palette[1]])
        st.plotly_chart(fig_drug_activity, use_container_width=True)

    # ... [The rest of the tabs (2, 3, 4, 5) remain the same as in the previous version]

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This app analyzes CSV data from the connected GitHub repository to detect and analyze suspicious activities related to drug trafficking on messaging platforms. 
    
    In a real-world scenario, such tools must be used responsibly, with proper legal authorization, and with careful consideration of privacy rights and potential biases in the AI system.
    """)

if __name__ == "__main__":
    main()
