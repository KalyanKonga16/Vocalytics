import streamlit as st
import psycopg2
import pandas as pd
import boto3
import time
import hashlib
import plotly.express as px

st.set_page_config(page_title="Vocalytics", layout="wide")
st.title("🎧 Multi-Modal Customer Support Auditor")

# ==================== DB ====================
def get_connection():
    for i in range(3):
        try:
            return psycopg2.connect(st.secrets["DATABASE_URL"])
        except psycopg2.OperationalError:
            if i < 2:
                time.sleep(2)
            else:
                raise

def is_duplicate(audio_hash: str) -> dict | None:
    """Returns existing record info if duplicate, else None"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT filename, topic, customer_sentiment, problem_resolved
            FROM call_analytics
            WHERE audio_hash = %s
            LIMIT 1
        """, (audio_hash,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {
                "filename": row[0],
                "topic": row[1],
                "sentiment": row[2],
                "resolved": row[3]
            }
        return None
    except Exception:
        return None

def fetch_data() -> pd.DataFrame:
    try:
        conn = get_connection()
        df = pd.read_sql_query("""
            SELECT
                filename         AS "File",
                topic            AS "Topic",
                customer_sentiment AS "Sentiment",
                problem_resolved AS "Resolved",
                processed_at     AS "Analyzed At",
                transcript       AS "Transcript"
            FROM call_analytics
            ORDER BY processed_at DESC
        """, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Could not load data: {str(e)}")
        return pd.DataFrame()

# ==================== SIDEBAR ====================
st.sidebar.header("📤 Upload New Call")
uploaded_file = st.sidebar.file_uploader(
    "Select an audio file",
    type=["mp3", "wav"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    audio_hash = hashlib.md5(file_bytes).hexdigest()
    existing = is_duplicate(audio_hash)

    if existing:
        resolved_label = "✅ Yes" if existing["resolved"] else "❌ No"
        st.sidebar.warning(
            "⚠️ **Already Analyzed**\n\n"
            f"**File:** {existing['filename']}\n\n"
            f"**Topic:** {existing['topic']}\n\n"
            f"**Sentiment:** {existing['sentiment']}\n\n"
            f"**Resolved:** {resolved_label}"
        )
    else:
        if st.sidebar.button("🚀 Process Call"):
            try:
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY"),
                    aws_secret_access_key=st.secrets.get("AWS_SECRET_KEY"),
                    region_name=st.secrets.get("AWS_REGION", "us-east-1")
                )
                s3_key = f"uploads/{audio_hash}_{uploaded_file.name}"
                uploaded_file.seek(0)
                s3.upload_fileobj(uploaded_file, st.secrets["S3_BUCKET"], s3_key)
                st.sidebar.success("✅ Uploaded! Analysis in progress.")
                st.sidebar.info("⏳ Wait 15–20 seconds, then refresh the page.")
            except Exception as e:
                st.sidebar.error(f"Upload failed: {str(e)}")

# ==================== DASHBOARD ====================
df = fetch_data()

if not df.empty:

    # --- KPI Row ---
    total     = len(df)
    negative  = len(df[df["Sentiment"] == "negative"])
    resolved  = len(df[df["Resolved"] == True])
    res_pct   = round((resolved / total) * 100, 1) if total > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📞 Total Calls",       total)
    k2.metric("😡 Negative Calls",    negative)
    k3.metric("✅ Resolved",          resolved)
    k4.metric("📈 Resolution Rate",   f"{res_pct}%")

    st.divider()

    # --- Charts ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🏷️ Calls by Topic")
        topic_df = df["Topic"].value_counts().reset_index()
        topic_df.columns = ["Topic", "Count"]
        fig = px.pie(
            topic_df,
            values="Count",
            names="Topic",
            hole=0.3
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("😊 Sentiment Breakdown")
        sent_df = df["Sentiment"].value_counts().reset_index()
        sent_df.columns = ["Sentiment", "Count"]
        color_map = {
            "negative": "#EF4444",
            "neutral":  "#F59E0B",
            "positive": "#10B981"
        }
        fig = px.bar(
            sent_df,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map=color_map
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Calls Table (Manager View) ---
    st.subheader("📋 Call Records")
    st.dataframe(
        df[[
            "File",
            "Topic",
            "Sentiment",
            "Resolved",
            "Analyzed At",
            "Transcript"
        ]],
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("📭 No calls analyzed yet. Upload an audio file to begin!")
