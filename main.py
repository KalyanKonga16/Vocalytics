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

st.set_page_config(page_title="Vocalytics", layout="wide")
st.title("🎧 Multi-Modal Customer Support Auditor")

# ==================== CONFIGURATION ====================
POLL_INTERVAL_SECONDS = 4
MAX_TOTAL_BATCH_MB = 50          # Warn only if combined upload exceeds this
SAFE_FILE_COUNT = 15             # Soft warning if too many files at once

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
        "size_mb": len(file_bytes) / (1024 * 1024)
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

# ==================== BATCH SIZE CHECK ====================
def check_batch_limits(file_items):
    """Returns (is_too_big, warning_message). Warns only when limits crossed."""
    total_mb = sum(item["size_mb"] for item in file_items)
    count = len(file_items)

    warnings = []
    block = False

    if total_mb > MAX_TOTAL_BATCH_MB:
        warnings.append(
            f"🚫 Total upload size is **{total_mb:.1f} MB**, "
            f"which exceeds the **{MAX_TOTAL_BATCH_MB} MB** batch limit. "
            f"Please upload fewer/smaller files."
        )
        block = True

    if count > SAFE_FILE_COUNT and not block:
        warnings.append(
            f"⚠️ You are uploading **{count} files** at once. "
            f"Processing may take a few minutes. You can still proceed."
        )

    return block, warnings, total_mb

# ==================== QUEUE-STYLE RENDERER ====================
STATUS_STYLE = {
    "queued":    ("⏳", "Queued",      "#6B7280"),
    "uploading": ("📤", "Uploading",   "#3B82F6"),
    "analyzing": ("🤖", "Analyzing",   "#8B5CF6"),
    "done":      ("✅", "Done",        "#10B981"),
    "skipped":   ("⏭️", "Skipped",     "#F59E0B"),
    "failed":    ("❌", "Failed",      "#EF4444"),
}

def render_queue(queue_items, container):
    """Renders the live queue UI inside a given container."""
    html = "<div style='display:flex;flex-direction:column;gap:8px;'>"
    for q in queue_items:
        icon, label, color = STATUS_STYLE[q["status"]]
        reason = f"<span style='color:#9CA3AF;font-size:12px;'> — {q['reason']}</span>" if q.get("reason") else ""
        html += f"""
        <div style='
            display:flex;align-items:center;justify-content:space-between;
            padding:10px 14px;border-radius:10px;
            background:rgba(255,255,255,0.03);
            border-left:4px solid {color};'>
            <div style='font-size:14px;color:#E5E7EB;overflow:hidden;
                        text-overflow:ellipsis;white-space:nowrap;max-width:60%;'>
                🎵 {q['name']}
            </div>
            <div style='font-size:13px;font-weight:600;color:{color};'>
                {icon} {label}{reason}
            </div>
        </div>
        """
    html += "</div>"
    container.markdown(html, unsafe_allow_html=True)

# ==================== SMART PROCESSOR WITH QUEUE ====================
def process_files_with_queue(file_items):
    """
    Processes files showing a live queue.
    - Skips duplicates (DB + in-batch)
    - Uploads new files
    - Then waits and marks them 'done' as DB rows appear
    """
    if not file_items:
        return

    st.subheader("📥 Upload Queue")
    queue_container = st.empty()

    # Build initial queue
    queue_items = [
        {"name": item["original_name"], "status": "queued", "reason": "", "hash": item["audio_hash"]}
        for item in file_items
    ]
    render_queue(queue_items, queue_container)
    time.sleep(0.4)

    # Pre-fetch existing DB hashes
    all_hashes = [item["audio_hash"] for item in file_items]
    existing_map = fetch_existing_records_by_hash(all_hashes)

    seen_in_batch = set()
    s3 = build_s3_client()

    uploaded_hashes = []   # hashes we actually sent to S3 (need to await analysis)

    # ---- PHASE 1: Upload / Skip ----
    for i, item in enumerate(file_items):
        h = item["audio_hash"]

        # Duplicate inside same batch
        if h in seen_in_batch:
            queue_items[i]["status"] = "skipped"
            queue_items[i]["reason"] = "Duplicate in this upload"
            render_queue(queue_items, queue_container)
            continue
        seen_in_batch.add(h)

        # Already in DB
        if h in existing_map:
            ex = existing_map[h]
            queue_items[i]["status"] = "skipped"
            queue_items[i]["reason"] = f"Already analyzed ({ex['topic']})"
            render_queue(queue_items, queue_container)
            continue

        # Upload
        queue_items[i]["status"] = "uploading"
        render_queue(queue_items, queue_container)
        try:
            upload_audio_to_s3(s3, item)
            queue_items[i]["status"] = "analyzing"
            queue_items[i]["reason"] = "Sent to AI pipeline"
            uploaded_hashes.append(h)
            render_queue(queue_items, queue_container)
        except Exception as e:
            queue_items[i]["status"] = "failed"
            queue_items[i]["reason"] = str(e)[:40]
            render_queue(queue_items, queue_container)

        time.sleep(0.3)

    # ---- PHASE 2: Await analysis completion ----
    if uploaded_hashes:
        status_banner = st.empty()
        pending = set(uploaded_hashes)
        start = time.time()
        timeout = 60 + (len(uploaded_hashes) * 20)  # scale with batch size

        # small head start for Lambda cold start
        time.sleep(6)

        while pending and (time.time() - start) < timeout:
            done_map = fetch_existing_records_by_hash(list(pending))

            for i, q in enumerate(queue_items):
                if q["hash"] in done_map and q["status"] == "analyzing":
                    info = done_map[q["hash"]]
                    queue_items[i]["status"] = "done"
                    queue_items[i]["reason"] = f"{info['topic']} | {info['sentiment']}"

            render_queue(queue_items, queue_container)

            pending = {h for h in pending if h not in done_map}

            if not pending:
                break

            elapsed = int(time.time() - start)
            remaining = int(timeout - elapsed)
            status_banner.info(
                f"🤖 AI agents working... {len(uploaded_hashes) - len(pending)}/{len(uploaded_hashes)} completed "
                f"| ⏱️ {elapsed}s elapsed | ⏰ {remaining}s left"
            )
            time.sleep(POLL_INTERVAL_SECONDS)

        if pending:
            # Some still not done (slow Lambda) — mark gracefully
            for i, q in enumerate(queue_items):
                if q["hash"] in pending and q["status"] == "analyzing":
                    queue_items[i]["reason"] = "Still processing (will appear shortly)"
            render_queue(queue_items, queue_container)
            status_banner.warning("⏳ Some files are still processing. Dashboard will show them shortly.")
        else:
            status_banner.success("🎉 All uploaded calls analyzed successfully!")

        time.sleep(2)

# ==================== SIDEBAR ====================
st.sidebar.header("📤 Upload Mode")

upload_mode = st.sidebar.radio(
    "Choose upload mode",
    ["Single File", "Multiple Files", "ZIP Batch"]
)

selected_items = []
trigger_process = False

# -------- SINGLE FILE --------
if upload_mode == "Single File":
    uploaded_file = st.sidebar.file_uploader(
        "Select an audio file",
        type=["mp3", "wav"],
        key="single_audio"
    )
    if uploaded_file is not None:
        item = make_file_item(uploaded_file.name, uploaded_file.getvalue())
        dup_map = fetch_existing_records_by_hash([item["audio_hash"]])

        if item["audio_hash"] in dup_map:
            ex = dup_map[item["audio_hash"]]
            resolved_label = "✅ Yes" if ex["resolved"] else "❌ No"
            st.sidebar.warning(
                "⚠️ **Already Analyzed**\n\n"
                f"**File:** `{ex['filename']}`\n\n"
                f"**Topic:** `{ex['topic']}`\n\n"
                f"**Sentiment:** `{ex['sentiment']}`\n\n"
                f"**Resolved:** {resolved_label}"
            )
        else:
            if st.sidebar.button("🚀 Analyze Now"):
                selected_items = [item]
                trigger_process = True

# -------- MULTIPLE FILES --------
elif upload_mode == "Multiple Files":
    uploaded_files = st.sidebar.file_uploader(
        "Select multiple files",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        key="multiple_audio"
    )
    if uploaded_files:
        items = [make_file_item(f.name, f.getvalue()) for f in uploaded_files]
        block, warns, total_mb = check_batch_limits(items)

        st.sidebar.info(f"📦 {len(items)} file(s) | {total_mb:.1f} MB total")
        for w in warns:
            st.sidebar.warning(w)

        if not block:
            if st.sidebar.button("🚀 Process All"):
                selected_items = items
                trigger_process = True

# -------- ZIP BATCH --------
elif upload_mode == "ZIP Batch":
    uploaded_zip = st.sidebar.file_uploader("Upload ZIP", type=["zip"], key="zip_upload")
    if uploaded_zip is not None:
        files_from_zip, skipped, zip_err = extract_audio_files_from_zip(uploaded_zip)
        if zip_err:
            st.sidebar.error(zip_err)
        elif not files_from_zip:
            st.sidebar.warning("No valid audio files found in ZIP.")
        else:
            block, warns, total_mb = check_batch_limits(files_from_zip)
            st.sidebar.info(f"📦 {len(files_from_zip)} audio file(s) | {total_mb:.1f} MB")
            if skipped:
                st.sidebar.warning(f"{len(skipped)} non-audio file(s) ignored.")
            for w in warns:
                st.sidebar.warning(w)

            if not block:
                if st.sidebar.button("🚀 Process ZIP Content"):
                    selected_items = files_from_zip
                    trigger_process = True

# ==================== RUN PROCESSING ====================
if trigger_process and selected_items:
    process_files_with_queue(selected_items)

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
