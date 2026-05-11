import streamlit as st
import psycopg2
import pandas as pd
import boto3
import os
import plotly.express as px

st.set_page_config(page_title="Support Auditor", layout="wide")
st.title("🎧 Multi-Modal Customer Support Auditor")

# Connect to Database
@st.cache_resource
def init_connection():
    return psycopg2.connect(st.secrets["DATABASE_URL"])

conn = init_connection()

# Sidebar: File Upload
st.sidebar.header("Upload New Call")
uploaded_file = st.sidebar.file_uploader("Upload Audio (mp3/wav)", type=['mp3', 'wav'])

if uploaded_file is not None:
    if st.sidebar.button("Process Call"):
        # Upload to S3 directly from Streamlit
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
            aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
        )
        s3.upload_fileobj(uploaded_file, st.secrets["S3_BUCKET"], uploaded_file.name)
        st.sidebar.success("File uploaded to S3! Analysis agent triggered.")

# Main Dashboard
df = pd.read_sql("SELECT * FROM call_analytics", conn)

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