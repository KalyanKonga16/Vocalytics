import streamlit as st
import psycopg2
import pandas as pd
import boto3
import time
import plotly.express as px
import hashlib

st.set_page_config(page_title="Support Auditor", layout="wide")
st.title("🎧 Multi-Modal Customer Support Auditor")


# ==================== DB HELPERS ====================
def get_connection():
    return psycopg2.connect(st.secrets["DATABASE_URL"])


def find_existing_audio(audio_hash):
    for _ in range(3):
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT id, filename, topic, customer_sentiment, processed_at, duplicate_count
                FROM call_analytics
                WHERE audio_hash = %s
                LIMIT 1
            """, (audio_hash,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            return row
        except psycopg2.OperationalError:
            time.sleep(2)
    return None


def fetch_analytics_data():
    for _ in range(3):
        try:
            conn = get_connection()
            df = pd.read_sql_query("""
                SELECT *
                FROM call_analytics
                ORDER BY processed_at DESC
            """, conn)
            conn.close()
            return df
        except psycopg2.OperationalError:
            time.sleep(2)
    return pd.DataFrame()


# ==================== SIDEBAR - FILE UPLOAD ====================
st.sidebar.header("📤 Upload New Call")
uploaded_file = st.sidebar.file_uploader("Upload Audio (mp3 / wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    audio_hash = hashlib.sha256(uploaded_bytes).hexdigest()

    existing_audio = find_existing_audio(audio_hash)

    if existing_audio:
        st.sidebar.warning(
            f"⚠️ This exact audio was already analyzed.\n\n"
            f"Filename: {existing_audio[1]}\n"
            f"Topic: {existing_audio[2]}\n"
            f"Sentiment: {existing_audio[3]}\n"
            f"Duplicates so far: {existing_audio[5]}"
        )
    else:
        if st.sidebar.button("Process Call"):
            try:
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
                    aws_secret_access_key=st.secrets["AWS_SECRET_KEY"],
                    region_name=st.secrets["AWS_REGION"]
                )

                # Use hash inside S3 key to make uploads cleaner
                s3_key = f"uploads/{audio_hash}_{uploaded_file.name}"

                uploaded_file.seek(0)
                s3.upload_fileobj(uploaded_file, st.secrets["S3_BUCKET"], s3_key)

                st.sidebar.success("✅ File uploaded! Analysis started.")
                st.sidebar.info("⏳ Wait 10–20 seconds, then refresh the page.")
            except Exception as e:
                st.sidebar.error(f"Upload failed: {str(e)}")


# ==================== DASHBOARD ====================
df = fetch_analytics_data()

if not df.empty:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Unique Calls", len(df))

    negative_calls = len(df[df["customer_sentiment"] == "negative"])
    col2.metric("Negative Sentiment", negative_calls)

    resolved_pct = (
        len(df[df["problem_resolved"] == True]) / len(df) * 100
        if len(df) > 0 else 0
    )
    col3.metric("Resolution Rate", f"{resolved_pct:.1f}%")

    blocked_duplicates = int(df["duplicate_count"].fillna(0).sum()) if "duplicate_count" in df.columns else 0
    col4.metric("Duplicate Uploads Blocked", blocked_duplicates)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Calls by Topic")
        topic_counts = df["topic"].value_counts().reset_index()
        topic_counts.columns = ["topic", "count"]
        fig = px.pie(topic_counts, values="count", names="topic")
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.subheader("Sentiment Distribution")
        sent_counts = df["customer_sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig = px.bar(sent_counts, x="sentiment", y="count", color="sentiment")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Transcripts & Analysis")

    display_cols = [
        col for col in [
            "filename",
            "last_uploaded_filename",
            "topic",
            "customer_sentiment",
            "problem_resolved",
            "duplicate_count",
            "last_seen_at",
            "transcript"
        ] if col in df.columns
    ]

    st.dataframe(df[display_cols], use_container_width=True)

else:
    st.info("No data yet. Upload an audio file to begin!")
