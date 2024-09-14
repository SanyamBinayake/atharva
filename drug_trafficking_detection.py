import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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
    X = vectorizer.fit_transform(messages_df['content'].astype(str))  # Ensure content is string type
    y = messages_df['is_drug_related']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, vectorizer, classification_report(y_test, y_pred)

# Function to debug predictions
def debug_prediction(user_input, model, vectorizer):
    # Transform the input
    input_vector = vectorizer.transform([user_input])
    
    # Get the prediction and probability
    prediction = model.predict(input_vector)[0]
    probability = model.predict_proba(input_vector)[0][1]
    
    # Print debug information
    print(f"Input: {user_input}")
    print(f"Input vector shape: {input_vector.shape}")
    print(f"Input vector non-zero elements: {input_vector.nnz}")
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability}")
    
    return prediction, probability

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

    with tab2:
        st.header("Channel Activity Levels")
        
        # Ensure 'activity_level' column exists
        if 'activity_level' in channels_df.columns:
            # Handle missing values
            channels_df['activity_level'].fillna('Unknown', inplace=True)
    
            # Activity level distribution
            activity_counts = channels_df['activity_level'].value_counts().reset_index()
            activity_counts.columns = ['activity_level', 'count']  # Rename columns for clarity
    
            # Check if there's data to plot
            if not activity_counts.empty:
                fig_activity = px.bar(activity_counts,
                                      x='activity_level',
                                      y='count',
                                      title="Channel Activity Levels",
                                      color_discrete_sequence=color_palette)
                st.plotly_chart(fig_activity, use_container_width=True)
            else:
                st.warning("No activity level data available.")
        else:
            st.warning("'activity_level' column is missing from the data.")

        
    


    with tab3:
        st.header("Channels/Groups")
        # Activity levels
        fig_activity = px.bar(channels_df['activity_level'].value_counts().reset_index(), 
                              x='index', y='activity_level', title="Channel Activity Levels",
                              labels={'index': 'Activity Level', 'activity_level': 'Number of Channels'},
                              color_discrete_sequence=color_palette)
        st.plotly_chart(fig_activity, use_container_width=True)
        
        # Member count distribution
        fig_members = px.histogram(channels_df, x='members_count', title="Channel Member Count Distribution",
                                   labels={'members_count': 'Number of Members', 'count': 'Number of Channels'},
                                   color_discrete_sequence=[color_palette[1]])
        st.plotly_chart(fig_members, use_container_width=True)
        
        # Private vs Public channels
        fig_privacy = px.pie(channels_df['is_private'].value_counts().reset_index(), 
                             values='is_private', names='index', title="Private vs Public Channels",
                             color_discrete_sequence=[color_palette[2], color_palette[3]])
        st.plotly_chart(fig_privacy, use_container_width=True)
        
        st.dataframe(channels_df)

    with tab4:
        st.header("Messages")
        # Drug-related vs non-drug-related messages
        fig_drug_related = px.pie(messages_df['is_drug_related'].value_counts().reset_index(), 
                                  values='is_drug_related', names='index', 
                                  title="Drug-Related vs Non-Drug-Related Messages",
                                  color_discrete_map={0: color_palette[0], 1: color_palette[1]})
        st.plotly_chart(fig_drug_related, use_container_width=True)
        
        # Engagement vs Drug-Related
        fig_engagement = px.box(messages_df, x='is_drug_related', y='engagement', 
                                title="Engagement vs Drug-Related Content",
                                labels={'is_drug_related': 'Is Drug Related', 'engagement': 'Engagement Level'},
                                color_discrete_sequence=color_palette)
        st.plotly_chart(fig_engagement, use_container_width=True)
        
        st.dataframe(messages_df)

    with tab5:
        st.header("Machine Learning Insights")
        st.subheader("Model Performance")
        st.text(classification_report)
        
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': vectorizer.get_feature_names_out(),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Top 20 Important Features for Drug-Related Message Detection",
                                color='importance',
                                color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_importance, use_container_width=True)

        st.subheader("Live Message Classification")
        user_input = st.text_area("Enter a message to classify:")
        if user_input:
            prediction, probability = debug_prediction(user_input, model, vectorizer)
            
            st.write(f"Prediction: {'Drug-Related' if prediction == 1 else 'Not Drug-Related'}")
            st.write(f"Probability of being drug-related: {probability:.2f}")
            
            st.write("Debug Information:")
            st.json({
                "Input": user_input,
                "Prediction": int(prediction),
                "Probability": float(probability),
                "Vectorizer Vocabulary Size": vectorizer.vocabulary_.__len__(),
                "Input Vector Non-Zero Elements": int(input_vector.nnz)
            })

if __name__ == "__main__":
    main()
