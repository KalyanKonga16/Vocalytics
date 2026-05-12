import streamlit as st
import psycopg2
import pandas as pd
import boto3
import os
import plotly.express as px

# Sidebar: File Upload
st.sidebar.header("Upload New Call")
uploaded_file = st.sidebar.file_uploader("Upload Audio (mp3/wav)", type=['mp3', 'wav'])

if uploaded_file is not None:
    if st.sidebar.button("Process Call"):
        # Ensure your AWS credentials and S3 bucket name are in Streamlit secrets!
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
            aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
        )
        s3.upload_fileobj(uploaded_file, st.secrets["S3_BUCKET"], uploaded_file.name)
        st.sidebar.success("File uploaded to S3! Analysis agent triggered.")
        st.experimental_rerun() # This will force the app to rerun and fetch new data

# Connect to Database
import time
import psycopg2
import pandas as pd
import streamlit as st

# Function to fetch data safely with auto-reconnect
def fetch_analytics_data():
    db_url = st.secrets["DATABASE_URL"]
    
    # Try fetching up to 3 times to handle Neon cold-starts
    for i in range(3):
        try:
            # 1. Open a fresh connection
            conn = psycopg2.connect(db_url)
            
            # 2. Grab the data
            df = pd.read_sql("SELECT * FROM call_analytics ORDER BY processed_at DESC", conn)
            
            # 3. Close the connection immediately (Serverless friendly!)
            conn.close()
            return df
            
        except psycopg2.OperationalError as e:
            if i < 2:
                time.sleep(2)  # Wait 2 seconds for Neon to wake up
                continue
            else:
                raise e

# Fetch the data using our new bulletproof function
df = fetch_analytics_data()

if not df.empty:
    # Top Row Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Calls Analyzed", len(df))
    negative_calls = len(df[df['customer_sentiment'] == 'negative'])
    col2.metric("Negative Sentiment", f"{negative_calls}")
    resolved_pct = (len(df[df['problem_resolved'] == True]) / len(df)) * 100
    col3.metric("Resolution Rate", f"{resolved_pct:.1f}%")

    # Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Calls by Topic")
        topic_counts = df['topic'].value_counts().reset_index()
        topic_counts.columns = ['topic', 'count']
        fig = px.pie(topic_counts, values='count', names='topic')
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.subheader("Sentiment Distribution")
        sent_counts = df['customer_sentiment'].value_counts().reset_index()
        sent_counts.columns = ['sentiment', 'count']
        fig = px.bar(sent_counts, x='sentiment', y='count', color='sentiment')
        st.plotly_chart(fig, use_container_width=True)

    # Raw Data Table
    st.subheader("Recent Transcripts & Analysis")
    st.dataframe(df[['filename', 'topic', 'customer_sentiment', 'problem_resolved', 'transcript']])
else:
    st.info("No data yet. Upload an audio file to begin!")