import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import random




# Now use 'drug_keywords' as usual


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
                <h3 style='color: #0066CC;'>Quick Info</h3>
                <p><strong>ID:</strong> 1674</p>
                <p><strong>Organization:</strong> Narcotics Control Bureau (NCB)</p>
                <p><strong>Department:</strong> Narcotics Control Bureau (NCB)</p>
                <p><strong>Category:</strong> Software</p>
                <p><strong>Theme:</strong> Blockchain & Cybersecurity</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style='background-color: #FFF5E6; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #CC6600;'>Software solutions to identify users behind Telegram, WhatsApp and Instagram based drug trafficking</h3>
                <h4 style='color: #FF8C00;'>Background:</h4>
                <p>Use of encrypted messaging/social media apps like Telegram, WhatsApp and Instagram for drug trafficking are on the rise. Channels operating on these platforms are blatantly being misused by drug traffickers for offering various narcotic drugs and psychotropic substances for sale.</p>
                <h4 style='color: #FF8C00;'>Key Points:</h4>
                <ul>
                <li>Drug traffickers create channels and handles to offer drugs for sale to subscribers.</li>
                <li>Customized Telegram bots are used by some traffickers to sell drugs.</li>
                <li>Majority of drugs offered are dangerous synthetic drugs like MDMA, LSD, Mephedrone etc.</li>
                <li>These apps are also used for drug-related communication.</li>
                </ul>
                <h4 style='color: #FF8C00;'>Expected Solution:</h4>
                <p>Development of a software solution to identify live channels/bots/handles offering drugs for sale in India, focusing on triangulating identifiable parameters like IP address, mobile number, email id etc. of the users behind these channels.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

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
        st.markdown("""
        **Explanation:** This line chart shows the total number of messages sent each day over time. 
        
        **Example:** If you see a spike on a particular date, it could indicate increased activity, possibly due to a new drug shipment or increased enforcement efforts.
        
        **Insights:** 
        - Look for unusual patterns or spikes in activity.
        - Regular patterns might indicate scheduled drug deals or shipments.
        - Sudden drops could suggest law enforcement interventions or technical issues with the platform.
        """)

        # Drug-related messages over time
        fig_drug_activity = px.line(messages_df[messages_df['is_drug_related'] == 1].groupby(messages_df['timestamp'].dt.date).size().reset_index(name='count'),
                                    x='timestamp', y='count', title="Suspected Drug-Related Messages Over Time",
                                    labels={'timestamp': 'Date', 'count': 'Number of Drug-Related Messages'},
                                    color_discrete_sequence=[color_palette[1]])
        st.plotly_chart(fig_drug_activity, use_container_width=True)
        st.markdown("""
        **Explanation:** This line chart focuses specifically on messages suspected to be drug-related over time.
        
        **Example:** A sudden increase in drug-related messages could indicate a new supplier entering the market or a change in drug availability.
        
        **Insights:**
        - Compare this chart with the overall activity chart to see if drug-related messages follow the same patterns.
        - Peaks could indicate times when dealers are most active, helping law enforcement plan operations.
        - Consistent levels might suggest ongoing, established drug trafficking operations.
        """)

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
    **Note:** This demo app showcases how to use machine learning to detect and analyze suspicious activities related to drug trafficking on messaging platforms using synthetic data. 
    
    In a real-world scenario, such tools must be used responsibly, with proper legal authorization, and with careful consideration of privacy rights and potential biases in the AI system.
    """)

if __name__ == "__main__":
    main()
