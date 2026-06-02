import streamlit as st
import psycopg2
import pandas as pd
import boto3
import time
import hashlib
import plotly.express as px
import zipfile
import io
import os
import re
from datetime import datetime

st.set_page_config(page_title="Vocalytics", layout="wide")
st.title("🎧 Multi-Modal Customer Support Auditor")

# ==================== CONFIGURATION ====================
# How long to wait between checks when processing
POLL_INTERVAL_SECONDS = 5

# How many total calls before current session started
# Used to detect when NEW data appears
initial_count = None

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

def fetch_existing_records_by_hash(audio_hashes):
    if not audio_hashes:
        return {}
    try:
        conn = get_connection()
        cur = conn.cursor()
        placeholders = ",".join(["%s"] * len(audio_hashes))
        query = f"""
            SELECT audio_hash, filename, topic, customer_sentiment, problem_resolved
            FROM call_analytics
            WHERE audio_hash IN ({placeholders})
        """
        cur.execute(query, audio_hashes)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        result = {}
        for row in rows:
            result[row[0]] = {
                "filename": row[1],
                "topic": row[2],
                "sentiment": row[3],
                "resolved": row[4]
            }
        return result
    except Exception as e:
        st.error(f"DB check failed: {str(e)}")
        return {}

def fetch_data() -> pd.DataFrame:
    try:
        conn = get_connection()
        df = pd.read_sql_query("""
            SELECT
                filename AS "File",
                topic AS "Topic",
                customer_sentiment AS "Sentiment",
                problem_resolved AS "Resolved",
                processed_at AS "Analyzed At",
                transcript AS "Transcript"
            FROM call_analytics
            ORDER BY processed_at DESC
        """, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Could not load data: {str(e)}")
        return pd.DataFrame()

def get_total_row_count():
    """Get only count for efficient lightweight checks"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM call_analytics")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except:
        return -1

# ==================== HELPERS ====================
def sanitize_filename(filename):
    filename = os.path.basename(filename)
    filename = filename.replace(" ", "_")
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    return filename

def build_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY"),
        aws_secret_access_key=st.secrets.get("AWS_SECRET_KEY"),
        region_name=st.secrets.get("AWS_REGION", "us-east-1")
    )

def make_file_item(filename, file_bytes):
    safe_name = sanitize_filename(filename)
    audio_hash = hashlib.md5(file_bytes).hexdigest()
    return {
        "original_name": filename,
        "safe_name": safe_name,
        "bytes": file_bytes,
        "audio_hash": audio_hash
    }

def upload_audio_to_s3(s3_client, item):
    bucket = st.secrets["S3_BUCKET"]
    s3_key = f"uploads/{item['audio_hash']}_{item['safe_name']}"
    
    content_type = "audio/mpeg"
    if item["safe_name"].lower().endswith(".wav"):
        content_type = "audio/wav"

    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=item["bytes"],
        ContentType=content_type
    )
    return s3_key

def extract_audio_files_from_zip(uploaded_zip):
    extracted_files = []
    skipped_non_audio = []

    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                base_name = os.path.basename(member.filename)
                if not base_name:
                    continue
                if not base_name.lower().endswith((".mp3", ".wav")):
                    skipped_non_audio.append(base_name)
                    continue
                file_bytes = z.read(member.filename)
                extracted_files.append(make_file_item(base_name, file_bytes))
        return extracted_files, skipped_non_audio, None
    except zipfile.BadZipFile:
        return [], [], "Invalid ZIP file."

def process_files(file_items):
    """Smart batch processor:
    - Skips duplicates inside this batch
    - Skips duplicates already in DB  
    - Continues even if one file fails/duplicate
    - Returns detailed report"""
    if not file_items:
        return {"uploaded": 0, "duplicate_db": 0, "duplicate_batch": 0, "failed": 0, "details": []}

    results = {
        "uploaded": 0,
        "duplicate_db": 0,
        "duplicate_batch": 0,
        "failed": 0,
        "details": []
    }

    all_hashes = [item["audio_hash"] for item in file_items]
    existing_map = fetch_existing_records_by_hash(all_hashes)

    seen_in_this_batch = set()
    s3 = build_s3_client()
    total_files = len(file_items)

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for idx, item in enumerate(file_items, start=1):
        status_text.info(f"Uploading {idx}/{total_files}: {item['original_name']}...")
        audio_hash = item["audio_hash"]

        # Duplicate inside same upload?
        if audio_hash in seen_in_this_batch:
            results["duplicate_batch"] += 1
            results["details"].append({
                "File": item["original_name"], 
                "Status": "Skipped", 
                "Reason": "Duplicate in this batch"
            })
            progress_bar.progress(idx / total_files)
            continue
        
        seen_in_this_batch.add(audio_hash)

        # Already processed in DB?
        if audio_hash in existing_map:
            existing = existing_map[audio_hash]
            results["duplicate_db"] += 1
            results["details"].append({
                "File": item["original_name"], 
                "Status": "Skipped", 
                "Reason": f"Already analyzed ({existing['topic']}, {existing['sentiment']})"
            })
            progress_bar.progress(idx / total_files)
            continue

        # New file: upload!
        try:
            s3_key = upload_audio_to_s3(s3, item)
            results["uploaded"] += 1
            results["details"].append({
                "File": item["original_name"], 
                "Status": "Processing...", 
                "Reason": f"Sent to S3"
            })
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "File": item["original_name"], 
                "Status": "Failed", 
                "Reason": str(e)
            })

        progress_bar.progress(idx / total_files)

    progress_bar.empty()
    status_text.empty()
    return results

def smart_wait_for_new_data(timeout_seconds=120):
    """
    After upload, check DB periodically until new rows appear.
    Returns True once fresh data detected, or False on timeout.
    This replaces the ugly auto-refresh approach.
    """
    global initial_count
    
    start_time = time.time()
    max_wait = timeout_seconds
    
    placeholder = st.empty()
    
    # Step 1: Record how many rows exist BEFORE we started waiting
    pre_upload_count = get_total_row_count()
    
    # If DB was empty initially, we will watch for first non-empty result
    watching_for_first_data = False
    if pre_upload_count == 0:
        watching_for_first_data = True
        # Wait a few extra seconds for S3 trigger + Lambda cold start + Whisper + Llama + DB insert
        initial_sleep = 8
        time.sleep(initial_sleep)
    
    polls = 0
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait:
            # Timeout reached
            placeholder.success("✅ Analysis pipeline triggered! (Checking for completion...)")
            time.sleep(5)  # Allow one final check then we just end the wait gracefully
            return True
            
        # Light sleep to avoid heavy CPU usage
        time.sleep(POLL_INTERVAL_SECONDS)
        polls += 1
        
        # Get fresh count efficiently
        current_count = get_total_row_count()
        
        # Build the spinner status message
        if watching_for_first_data:
            if current_count > 0:
                placeholder.success(f"🎉 New data arrived! Found {current_count} call(s). Loading dashboard...")
                time.sleep(2)
                return True
            else:
                remaining = int(max_wait - elapsed)
                placeholder.info(
                    f"⏳ Awaiting AI Analysis... "
                    f"(Waiting for serverless pipeline: Lambda + Groq Whisper + Llama)\n"
                    f"⏱️ Elapsed: {int(elapsed)}s | ⏰ Timeout in: {remaining}s | 🔍 Check #{polls})"
                )
        else:
            # Watching for specific number of rows to increase
            if current_count > pre_upload_count:
                new_rows = current_count - pre_upload_count
                placeholder.success(f"🎉 Success! {new_rows} new call(s) analyzed and saved.")
                time.sleep(2)
                return True
            else:
                remaining = int(max_wait - elapsed)
                placeholder.info(
                    f"⏳ Processing audio via Cloud AI agents...\n"
                    f"⏱️ Elapsed: {int(elapsed)}s | ⏰ Timeout in: {remaining}s | 🔍 Poll #{polls} | "
                    f"(Current DB: {current_count} | Expected: >{pre_upload_count})"
                )

# ==================== SIDEBAR ====================
st.sidebar.header("📤 Upload Mode")

upload_mode = st.sidebar.radio(
    "Choose upload mode",
    ["Single File", "Multiple Files", "ZIP Batch"]
)

if upload_mode == "Single File":
    uploaded_file = st.sidebar.file_uploader(
        "Select an audio file",
        type=["mp3", "wav"],
        key="single_audio"
    )

    if uploaded_file is not None:
        file_item = make_file_item(uploaded_file.name, uploaded_file.getvalue())
        
        # Quick local check before showing anything
        dup_map = fetch_existing_records_by_hash([file_item["audio_hash"]])
        
        if file_item["audio_hash"] in dup_map:
            existing = dup_map[file_item["audio_hash"]]
            resolved_label = "✅ Yes" if existing["resolved"] else "❌ No"
            
            st.sidebar.warning(
                "⚠️ **Already Analyzed**\n\n"
                f"**File:** `{existing['filename']}`\n\n"
                f"**Topic:** `{existing['topic']}`\n\n"
                f"**Sentiment:** `{existing['sentiment']}`\n\n"
                f"**Resolved:** {resolved_label}\n\n"
                f"*This exact audio content already exists in database."
            )
        elif st.sidebar.button("🚀 Analyze Now"):
            report = process_files([file_item])
            
            if report["uploaded"] == 1:
                st.sidebar.success("✅ Audio sent to cloud!")
                
                # Trigger smooth freeze/wait state
                smart_wait_for_new_data(timeout_seconds=120)
                
            elif report["failed"] > 0:
                st.sidebar.error("❌ Upload failed. Check S3 permissions.")

elif upload_mode == "Multiple Files":
    uploaded_files = st.sidebar.file_uploader(
        "Select multiple files",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        key="multiple_audio"
    )

    if uploaded_files:
        st.sidebar.info(f"📦 {len(uploaded_files)} file(s) selected")
        
        if st.sidebar.button("🚀 Process All"):
            items_list = []
            for f in uploaded_files:
                items_list.append(make_file_item(f.name, f.getvalue()))
            
            report = process_files(items_list)
            
            summary_msg = ""
            if report["uploaded"] > 0:
                summary_msg += f"✅ {report['uploaded']} sent for analysis\n\n"
            if report["duplicate_db"] > 0:
                summary_msg += f"⚠️ {report['duplicate_db']} already processed (skipped)\n\n"
            if report["duplicate_batch"] > 0:
                summary_msg += f"⚠️ {report['duplicate_batch']} duplicates skipped\n\n"
            if report["failed"] > 0:
                summary_msg += f"❌ {report['failed']} failed\n\n"
            
            st.sidebar.text(summary_msg)
            
            if report["uploaded"] > 0:
                smart_wait_for_new_data(timeout_seconds=180)  # Longer timeout for multiple files

elif upload_mode == "ZIP Batch":
    uploaded_zip = st.sidebar.file_uploader(
        "Upload ZIP",
        type=["zip"],
        key="zip_upload"
    )

    if uploaded_zip is not None:
        files_from_zip, skipped, zip_err = extract_audio_files_from_zip(uploaded_zip)
        
        if zip_err:
            st.sidebar.error(zip_err)
        elif not files_from_zip:
            st.sidebar.warning("No valid audio files found in ZIP.")
        else:
            st.sidebar.info(f"📦 Extracted {len(files_from_zip)} valid audio file(s)")
            
            if skipped:
                st.sidebar.warning(f"{len(skipped)} non-audio files ignored.")
            
            if st.sidebar.button("🚀 Process ZIP Content"):
                report = process_files(files_from_zip)
                
                summary_msg = ""
                if report["uploaded"] > 0:
                    summary_msg += f"✅ {report['uploaded']} files sent for analysis\n\n"
                if report["duplicate_db"] > 0:
                    summary_msg += f"⚠️ {report['duplicate_db']} already processed (skipped)\n\n"
                if report["duplicate_batch"] > 0:
                    summary_msg += f"⚠️ {report['duplicate_batch']} duplicates skipped\n\n"
                if report["failed"] > 0:
                    summary_msg += f"❌ {report['failed']} failed\n\n"
                
                st.sidebar.text(summary_msg)
                
                if report["uploaded"] > 0:
                    smart_wait_for_new_data(timeout_seconds=240)  # Even longer for big ZIP batches

# ==================== REPORT DRAWER ====================
if "last_upload_report" in st.session_state:
    rpt = st.session_state["last_upload_report"]
    
    with st.expander(f"📊 Last Upload Report (Uploaded: {rpt['uploaded']})"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📤 Sent to AI", rpt["uploaded"])
        c2.metric("⏭️ Already Done", rpt["duplicate_db"])
        c3.metric("⚠️ Duplicates in Batch", rpt["duplicate_batch"])
        c4.metric("❌ Failed", rpt["failed"])
        
        if rpt["details"]:
            st.dataframe(pd.DataFrame(rpt["details"]), hide_index=True, use_container_width=True)

# ==================== DASHBOARD ====================
df = fetch_data()

if not df.empty:

    total     = len(df)
    negative  = len(df[df["Sentiment"] == "negative"])
    resolved  = len(df[df["Resolved"] == True])
    res_pct   = round((resolved / total) * 100, 1) if total > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📞 Total Calls", total)
    k2.metric("😡 Negative Calls", negative)
    k3.metric("✅ Resolved", resolved)
    k4.metric("📈 Resolution Rate", f"{res_pct}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🏷️ Calls by Topic")
        topic_df = df["Topic"].value_counts().reset_index()
        topic_df.columns = ["Topic", "Count"]
        fig = px.pie(topic_df, values="Count", names="Topic", hole=0.3)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("😊 Sentiment Breakdown")
        sent_df = df["Sentiment"].value_counts().reset_index()
        sent_df.columns = ["Sentiment", "Count"]

        color_map = {
            "negative": "#EF4444",
            "neutral": "#F59E0B",
            "positive": "#10B981"
        }
        fig = px.bar(sent_df, x="Sentiment", y="Count", color="Sentiment", color_discrete_map=color_map)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

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
