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

    with tab2:
        st.header("User Profiles")
        
        # Device distribution
        fig_devices = px.pie(users_df['device_info'].value_counts().reset_index(), 
                             values='count', names='device_info', title="Device Distribution",
                             color_discrete_sequence=color_palette)
        st.plotly_chart(fig_devices, use_container_width=True)
        st.markdown("""
        **Explanation:** This pie chart shows the distribution of device types used by users on the platform.
        
        **Example:** If a large portion of users are on mobile devices, it might indicate that drug deals are often arranged on-the-go.
        
        **Insights:**
        - A high proportion of a specific device type might indicate a preferred platform for drug dealers.
        - Unusual devices could be a red flag for suspicious activity.
        - This information can help tailor prevention and intervention strategies for specific platforms.
        """)
        
 
        # User activity
        fig_user_activity = px.scatter(users_df, x='account_age_days', y='avg_daily_messages', 
                                       title="User Activity", hover_data=['username'],
                                       labels={'account_age_days': 'Account Age (Days)', 'avg_daily_messages': 'Average Daily Messages'},
                                       color_discrete_sequence=[color_palette[3]])
        st.plotly_chart(fig_user_activity, use_container_width=True)
        st.markdown("""
        **Explanation:** This scatter plot shows the relationship between a user's account age and their average daily message count.
        
        **Example:** New accounts with very high message counts could be spam accounts or new drug dealers trying to establish themselves quickly.
        
        **Insights:**
        - Outliers (e.g., very new accounts with high activity) could indicate suspicious behavior.
        - Long-standing accounts with consistent, moderate activity might be established drug dealers.
        - This can help identify potential key players in drug trafficking networks.
        """)
        
        st.dataframe(users_df)

    with tab3:
        st.header("Channels/Groups")
        
        # Activity levels
        fig_activity = px.bar(channels_df['activity_level'].value_counts().reset_index(), 
                              x='activity_level', y='count', title="Channel Activity Levels",
                              labels={'activity_level': 'Activity Level', 'count': 'Number of Channels'},
                              color='activity_level',
                              color_discrete_sequence=color_palette)
        st.plotly_chart(fig_activity, use_container_width=True)
        st.markdown("""
        **Explanation:** This bar chart shows the distribution of channel activity levels.
        
        **Example:** A high number of very active channels could indicate a thriving drug marketplace.
        
        **Insights:**
        - High-activity channels may be prime targets for investigation.
        - Low-activity channels shouldn't be ignored as they might be used for more discreet transactions.
        - The overall distribution can give an idea of how vibrant the drug marketplace is on the platform.
        """)
        
        # Member count distribution
        fig_members = px.histogram(channels_df, x='members_count', title="Channel Member Count Distribution",
                                   labels={'members_count': 'Number of Members', 'count': 'Number of Channels'},
                                   color_discrete_sequence=[color_palette[1]])
        st.plotly_chart(fig_members, use_container_width=True)
        st.markdown("""
        **Explanation:** This histogram shows the distribution of channel sizes based on member count.
        
        **Example:** A large number of small channels might indicate many small-scale dealers, while a few very large channels could be major drug marketplaces.
        
        **Insights:**
        - Large channels might be more visible but also more likely to contain a mix of legitimate and illegal activity.
        - Small, private channels might be more likely to be purely for drug trafficking.
        - The overall distribution can give insights into the structure of drug trafficking networks on the platform.
        """)
        
        # Private vs Public channels
        fig_privacy = px.pie(channels_df['is_private'].value_counts().reset_index(), 
                             values='count', names='is_private', title="Private vs Public Channels",
                             color_discrete_sequence=[color_palette[2], color_palette[3]])
        st.plotly_chart(fig_privacy, use_container_width=True)
        st.markdown("""
        **Explanation:** This pie chart shows the proportion of private versus public channels.
        
        **Example:** A high proportion of private channels could indicate that most drug deals are conducted in secretive, invite-only groups.
        
        **Insights:**
        - Private channels are more likely to be used for illegal activities as they offer more control over who can view the content.
        - Public channels might be used for initial contact or advertising, with deals moving to private channels.
        - This information can help guide investigation strategies, focusing on infiltrating private channels or monitoring public ones for leads.
        """)
        
        st.dataframe(channels_df)

    with tab4:
        st.header("Messages")
        
        # Drug-related vs non-drug-related messages
        fig_drug_related = px.pie(messages_df['is_drug_related'].value_counts().reset_index(), 
                                  values='count', names='is_drug_related', 
                                  title="Drug-Related vs Non-Drug-Related Messages",
                                  labels={'is_drug_related': 'Is Drug Related', 'count': 'Number of Messages'},
                                  color_discrete_map={0: color_palette[0], 1: color_palette[1]})
        st.plotly_chart(fig_drug_related, use_container_width=True)
        st.markdown("""
        **Explanation:** This pie chart shows the proportion of messages that are suspected to be drug-related versus those that are not.
        
        **Example:** If 20% of messages are flagged as drug-related, it suggests a significant level of drug-related activity on the platform.
        
        **Insights:**
        - A high proportion of drug-related messages indicates a serious drug trafficking problem on the platform.
        - Even a small percentage can be significant if the overall message volume is high.
        - This ratio can help prioritize resources for monitoring and intervention.
        """)
        
        # Engagement vs Drug-Related
        fig_engagement = px.box(messages_df, x='is_drug_related', y='engagement', 
                                title="Engagement vs Drug-Related Content",
                                labels={'is_drug_related': 'Is Drug Related', 'engagement': 'Engagement Level'},
                                color='is_drug_related',
                                color_discrete_map={0: color_palette[2], 1: color_palette[3]})
        st.plotly_chart(fig_engagement, use_container_width=True)
        st.markdown("""
        **Explanation:** This box plot compares the engagement levels of drug-related messages versus non-drug-related messages.
        
        **Example:** If drug-related messages have higher engagement, it might indicate a high demand for drugs on the platform.
        
        **Insights:**
        - Higher engagement on drug-related messages could suggest an active and interested audience for drug content.
        - Lower engagement might indicate that drug-related content is being ignored or reported by most users.
        - Outliers in engagement could help identify particularly influential drug-related messages or users.
        """)
        
        st.dataframe(messages_df)

    with tab5:
        st.header("Machine Learning Insights")
        
        st.subheader("Model Performance")
        st.text(classification_report)
        st.markdown("""
        **Explanation:** This report shows how well our machine learning model is performing in identifying drug-related messages.
        
        **Example:** If the precision for drug-related messages is 0.85, it means that when the model predicts a message is drug-related, it's correct 85% of the time.
        
        **Insights:**
        - High precision reduces false positives, ensuring we don't wrongly accuse innocent conversations.
        - High recall ensures we're catching most of the actual drug-related messages.
        - The F1-score balances precision and recall, giving an overall measure of the model's performance.
        """)
        
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': vectorizer.get_feature_names_out(),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Top 20 Important Features for Drug-Related Message Detection",
                                labels={'importance': 'Importance Score', 'feature': 'Word or Phrase'},
                                color='importance',
                                color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown("""
        **Explanation:** This chart shows the words or phrases that are most indicative of drug-related messages, according to our model.
        
        **Example:** If 'cocaine' is at the top of the list, it means that the presence of this word is a strong indicator of a drug-related message.
        
        **Insights:**
        - These features can help understand the common language used in drug-related messages.
        - New or unexpected words in this list might reveal emerging trends or code words in drug trafficking.
        - This information can be used to update keyword lists for future monitoring and detection efforts.
        """)
        
        st.subheader("Live Message Classification")
        user_input = st.text_area("Enter a message to classify:")
        if user_input:
            prediction = model.predict(vectorizer.transform([user_input]))[0]
            probability = model.predict_proba(vectorizer.transform([user_input]))[0][1]
            
            st.write(f"Prediction: {'Drug-Related' if prediction == 1 else 'Not Drug-Related'}")
            st.write(f"Probability of being drug-related: {probability:.2f}")
            st.markdown("""
            **Explanation:** This tool allows you to input a message and see whether our model classifies it as drug-related or not.
            
            **Example:** If you input "Hey, want to meet for coffee?", the model should classify it as not drug-related with a low probability.
            
            **Insights:**
            - This can be used to quickly assess suspicious messages.
            - The probability gives an idea of how confident the model is in its prediction.
            - If the model frequently misclassifies certain types of messages, it might need further training or refinement.
            """)


    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This app analyzes CSV data from the connected GitHub repository to detect and analyze suspicious activities related to drug trafficking on messaging platforms. 
    
    In a real-world scenario, such tools must be used responsibly, with proper legal authorization, and with careful consideration of privacy rights and potential biases in the AI system.
    """)

if __name__ == "__main__":
    main()
