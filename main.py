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

st.set_page_config(page_title="Vocalytics", layout="wide", page_icon="🎧")
st.title("🎧 Multi-Modal Customer Support Auditor")

# ==================== CONFIGURATION ====================
POLL_INTERVAL_SECONDS = 4
MAX_RECOMMENDED_BATCH_SIZE = 10  # Warn only if exceeds this

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
        "audio_hash": audio_hash,
        "status": "pending",  # pending, uploaded, transcribing, analyzing, complete, skipped, failed
        "message": ""
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

def render_status_badge(status):
    """Render beautiful status badge for queue UI"""
    badges = {
        "pending": ("⏳", "Pending", "#6B7280"),
        "uploaded": ("📤", "Uploaded to S3", "#3B82F6"),
        "transcribing": ("🎙️", "Transcribing...", "#8B5CF6"),
        "analyzing": ("🧠", "Analyzing...", "#F59E0B"),
        "complete": ("✅", "Complete", "#10B981"),
        "skipped": ("⏭️", "Skipped (Duplicate)", "#6B7280"),
        "failed": ("❌", "Failed", "#EF4444")
    }
    
    if status in badges:
        icon, label, color = badges[status]
        return f"""
        <div style="
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            background-color: {color}20;
            color: {color};
            font-size: 12px;
            font-weight: 600;
            border: 1px solid {color}40;
        ">
            {icon} {label}
        </div>
        """
    return status

def render_queue_ui(file_items, title="📋 Upload Queue"):
    """Beautiful queue-like progress UI for uploaded files"""
    st.subheader(title)
    
    if not file_items:
        st.info("No files in queue.")
        return
    
    # Summary stats
    status_counts = {}
    for item in file_items:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    
    # Summary row
    cols = st.columns(min(5, len(status_counts)))
    status_labels = {
        "pending": "⏳ Pending",
        "uploaded": "📤 Uploaded",
        "transcribing": "🎙️ Transcribing",
        "analyzing": "🧠 Analyzing",
        "complete": "✅ Complete",
        "skipped": "⏭️ Skipped",
        "failed": "❌ Failed"
    }
    
    for idx, (status, count) in enumerate(status_counts.items()):
        with cols[idx % 5]:
            st.metric(status_labels.get(status, status), count)
    
    st.divider()
    
    # Individual file cards
    for idx, item in enumerate(file_items, start=1):
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"**{idx}. {item['original_name']}**")
                if item.get("message"):
                    st.caption(item["message"])
            
            with col2:
                st.markdown(render_status_badge(item["status"]), unsafe_allow_html=True)
            
            with col3:
                # Progress bar for this file
                progress_map = {
                    "pending": 0.0,
                    "uploaded": 0.25,
                    "transcribing": 0.5,
                    "analyzing": 0.75,
                    "complete": 1.0,
                    "skipped": 1.0,
                    "failed": 0.5
                }
                prog = progress_map.get(item["status"], 0.0)
                st.progress(prog)
            
            st.divider()

def process_files_with_queue(file_items, queue_container):
    """
    Process files with real-time queue UI updates.
    Each file moves through stages visibly.
    """
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

    # Initial render
    with queue_container:
        render_queue_ui(file_items, "📋 Processing Queue")

    for idx, item in enumerate(file_items, start=1):
        audio_hash = item["audio_hash"]

        # Update UI: currently processing
        item["message"] = f"Processing {idx}/{total_files}..."
        with queue_container:
            render_queue_ui(file_items, "📋 Processing Queue")

        # Duplicate inside same upload?
        if audio_hash in seen_in_this_batch:
            item["status"] = "skipped"
            item["message"] = "Duplicate file in this batch"
            results["duplicate_batch"] += 1
            results["details"].append({
                "File": item["original_name"], 
                "Status": "Skipped", 
                "Reason": "Duplicate in this batch"
            })
            with queue_container:
                render_queue_ui(file_items, "📋 Processing Queue")
            continue
        
        seen_in_this_batch.add(audio_hash)

        # Already processed in DB?
        if audio_hash in existing_map:
            existing = existing_map[audio_hash]
            item["status"] = "skipped"
            item["message"] = f"Already analyzed: {existing['topic']}, {existing['sentiment']}"
            results["duplicate_db"] += 1
            results["details"].append({
                "File": item["original_name"], 
                "Status": "Skipped", 
                "Reason": f"Already analyzed ({existing['topic']}, {existing['sentiment']})"
            })
            with queue_container:
                render_queue_ui(file_items, "📋 Processing Queue")
            continue

        # New file: upload to S3!
        try:
            item["status"] = "uploaded"
            item["message"] = "Uploading to S3..."
            with queue_container:
                render_queue_ui(file_items, "📋 Processing Queue")
            
            s3_key = upload_audio_to_s3(s3, item)
            
            # Simulate stage progression for visual feedback
            item["message"] = "Triggering AI transcription..."
            item["status"] = "transcribing"
            with queue_container:
                render_queue_ui(file_items, "📋 Processing Queue")
            
            time.sleep(1)  # Visual pause for effect
            
            item["message"] = "Running sentiment & topic analysis..."
            item["status"] = "analyzing"
            with queue_container:
                render_queue_ui(file_items, "📋 Processing Queue")
            
            time.sleep(1)  # Visual pause for effect
            
            # Mark as complete (actual completion happens via Lambda → DB)
            item["status"] = "complete"
            item["message"] = f"Sent for processing → {s3_key}"
            results["uploaded"] += 1
            results["details"].append({
                "File": item["original_name"], 
                "Status": "Processing", 
                "Reason": f"S3: {s3_key}"
            })
            
            with queue_container:
                render_queue_ui(file_items, "📋 Processing Queue")
                
        except Exception as e:
            item["status"] = "failed"
            item["message"] = str(e)
            results["failed"] += 1
            results["details"].append({
                "File": item["original_name"], 
                "Status": "Failed", 
                "Reason": str(e)
            })
            with queue_container:
                render_queue_ui(file_items, "📋 Processing Queue")

    return results

def smart_wait_for_new_data(queue_container, uploaded_count, timeout_seconds=120):
    """
    After upload, poll DB until new rows appear.
    Updates queue UI with live status.
    """
    start_time = time.time()
    max_wait = timeout_seconds
    
    pre_upload_count = get_total_row_count()
    watching_for_first_data = (pre_upload_count == 0)
    
    # Initial delay for Lambda cold start + S3 trigger
    initial_sleep = 6
    time.sleep(initial_sleep)
    
    polls = 0
    status_box = queue_container.empty()
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait:
            with status_box:
                st.success("""
                ### ✅ Pipeline Triggered Successfully!
                
                Your audio files have been sent to the AI analysis pipeline.
                
                **What happens next:**
                1.  S3 stores your audio securely
                2. 🎙️ Whisper transcribes speech to text
                3. 🧠 Llama 3.1 analyzes sentiment & topics
                4. 💾 Results saved to database
                
                *Refresh the page in 30-60 seconds to see completed results.*
                """)
            return True
            
        polls += 1
        current_count = get_total_row_count()
        remaining = int(max_wait - elapsed)
        
        if watching_for_first_data:
            if current_count > 0:
                with status_box:
                    st.success(f"""
                    ### 🎉 First Data Arrived!
                    
                    Found **{current_count}** call(s) in database.
                    
                    *Refreshing dashboard...*
                    """)
                time.sleep(2)
                return True
            else:
                with status_box:
                    st.info(f"""
                    ### ⏳ Awaiting AI Analysis...
                    
                    **Pipeline Status:**
                    - 🎙️ Whisper Transcription: In Progress
                    - 🧠 Llama Analysis: Queued
                    - 💾 Database Insert: Pending
                    
                    **Time Elapsed:** {int(elapsed)}s | **Timeout:** {remaining}s | **Check:** #{polls}
                    """)
        else:
            if current_count > pre_upload_count:
                new_rows = current_count - pre_upload_count
                with status_box:
                    st.success(f"""
                    ### 🎉 Analysis Complete!
                    
                    **{new_rows}** new call(s) analyzed and saved.
                    
                    *Updating dashboard with fresh statistics...*
                    """)
                time.sleep(2)
                return True
            else:
                with status_box:
                    st.info(f"""
                    ### ⏳ Processing Audio via Cloud AI...
                    
                    **Pipeline Progress:**
                    - 📤 S3 Upload: ✅ Complete
                    - 🎙️ Whisper Transcription: In Progress
                    - 🧠 Llama Analysis: Queued
                    - 💾 Database Insert: Pending
                    
                    **Files Sent:** {uploaded_count} | **Elapsed:** {int(elapsed)}s | **Timeout:** {remaining}s
                    """)
        
        time.sleep(POLL_INTERVAL_SECONDS)

# ==================== SIDEBAR ====================
st.sidebar.header("📤 Upload Mode")

upload_mode = st.sidebar.radio(
    "Choose upload mode",
    ["Single File", "Multiple Files", "ZIP Batch"]
)

# Session state for queue
if "current_queue" not in st.session_state:
    st.session_state.current_queue = []
if "show_queue" not in st.session_state:
    st.session_state.show_queue = False

# -------- SINGLE FILE MODE --------
if upload_mode == "Single File":
    uploaded_file = st.sidebar.file_uploader(
        "Select an audio file",
        type=["mp3", "wav"],
        key="single_audio"
    )

    if uploaded_file is not None:
        file_item = make_file_item(uploaded_file.name, uploaded_file.getvalue())
        
        dup_map = fetch_existing_records_by_hash([file_item["audio_hash"]])
        
        if file_item["audio_hash"] in dup_map:
            existing = dup_map[file_item["audio_hash"]]
            resolved_label = "✅ Yes" if existing["resolved"] else "❌ No"
            
            st.sidebar.warning(
                "⚠️ **Already Analyzed**\n\n"
                f"**File:** `{existing['filename']}`\n\n"
                f"**Topic:** `{existing['topic']}`\n\n"
                f"**Sentiment:** `{existing['sentiment']}`\n\n"
                f"**Resolved:** {resolved_label}"
            )
        elif st.sidebar.button("🚀 Analyze Now"):
            st.session_state.show_queue = True
            st.session_state.current_queue = [file_item]
            
            with st.container():
                queue_container = st.empty()
                report = process_files_with_queue([file_item], queue_container)
                
                if report["uploaded"] == 1:
                    st.sidebar.success("✅ Audio sent to cloud AI pipeline!")
                    smart_wait_for_new_data(queue_container, 1, timeout_seconds=120)
                elif report["failed"] > 0:
                    st.sidebar.error("❌ Upload failed. Check S3 permissions.")

# -------- MULTIPLE FILES MODE --------
elif upload_mode == "Multiple Files":
    uploaded_files = st.sidebar.file_uploader(
        "Select multiple files",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        key="multiple_audio"
    )

    if uploaded_files:
        file_count = len(uploaded_files)
        
        # Batch size warning ONLY when exceeds limit
        if file_count > MAX_RECOMMENDED_BATCH_SIZE:
            st.sidebar.warning(f"""
            ⚠️ **Large Batch Detected**
            
            You've selected **{file_count}** files.
            
            **Recommended:** {MAX_RECOMMENDED_BATCH_SIZE} files or less
            
            **Estimated Time:** ~{file_count * 10}-{file_count * 20} seconds
            
            *Processing will continue in parallel.*
            """)
        else:
            st.sidebar.info(f"📦 {file_count} file(s) selected")
        
        if st.sidebar.button("🚀 Process All"):
            st.session_state.show_queue = True
            items_list = []
            for f in uploaded_files:
                items_list.append(make_file_item(f.name, f.getvalue()))
            
            st.session_state.current_queue = items_list
            
            with st.container():
                queue_container = st.empty()
                report = process_files_with_queue(items_list, queue_container)
                
                summary_msg = ""
                if report["uploaded"] > 0:
                    summary_msg += f"✅ {report['uploaded']} sent for analysis\n"
                if report["duplicate_db"] > 0:
                    summary_msg += f"⚠️ {report['duplicate_db']} already processed\n"
                if report["duplicate_batch"] > 0:
                    summary_msg += f"⚠️ {report['duplicate_batch']} duplicates skipped\n"
                if report["failed"] > 0:
                    summary_msg += f"❌ {report['failed']} failed\n"
                
                st.sidebar.text(summary_msg)
                
                if report["uploaded"] > 0:
                    smart_wait_for_new_data(queue_container, report["uploaded"], timeout_seconds=180)

# -------- ZIP BATCH MODE --------
elif upload_mode == "ZIP Batch":
    uploaded_zip = st.sidebar.file_uploader(
        "Upload ZIP file",
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
            file_count = len(files_from_zip)
            
            # Batch size warning ONLY when exceeds limit
            if file_count > MAX_RECOMMENDED_BATCH_SIZE:
                st.sidebar.warning(f"""
                ⚠️ **Large ZIP Batch Detected**
                
                ZIP contains **{file_count}** audio files.
                
                **Recommended:** {MAX_RECOMMENDED_BATCH_SIZE} files or less
                
                **Estimated Time:** ~{file_count * 10}-{file_count * 20} seconds
                
                *Processing will continue in parallel.*
                """)
            else:
                st.sidebar.info(f"📦 Extracted {file_count} valid audio file(s)")
            
            if skipped:
                st.sidebar.warning(f"{len(skipped)} non-audio files ignored.")
            
            if st.sidebar.button("🚀 Process ZIP Content"):
                st.session_state.show_queue = True
                st.session_state.current_queue = files_from_zip
                
                with st.container():
                    queue_container = st.empty()
                    report = process_files_with_queue(files_from_zip, queue_container)
                    
                    summary_msg = ""
                    if report["uploaded"] > 0:
                        summary_msg += f"✅ {report['uploaded']} files sent for analysis\n"
                    if report["duplicate_db"] > 0:
                        summary_msg += f"⚠️ {report['duplicate_db']} already processed\n"
                    if report["duplicate_batch"] > 0:
                        summary_msg += f"⚠️ {report['duplicate_batch']} duplicates skipped\n"
                    if report["failed"] > 0:
                        summary_msg += f"❌ {report['failed']} failed\n"
                    
                    st.sidebar.text(summary_msg)
                    
                    if report["uploaded"] > 0:
                        smart_wait_for_new_data(queue_container, report["uploaded"], timeout_seconds=240)

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

# ==================== FOOTER ====================
st.divider()
st.caption("""
**Vocalytics** | Multi-Modal Customer Support Auditor | 
Powered by AWS Lambda + Groq Whisper + Llama 3.1 + Neon PostgreSQL
""")
