import json
import urllib.parse
import boto3
import os
import requests
import psycopg2
import hashlib
import re

s3 = boto3.client("s3")


def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, remove punctuation for content comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def safe_get_column(cur, table, column):
    """Check if a column exists in a table (prevents crashes if schema is old)."""
    try:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s
            LIMIT 1
        """, (table, column))
        return cur.fetchone() is not None
    except Exception:
        return False


def lambda_handler(event, context):
    conn = None
    cursor = None

    try:
        # =====================================================
        # 1. EXTRACT S3 EVENT INFORMATION
        # =====================================================
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(
            event["Records"][0]["s3"]["object"]["key"],
            encoding="utf-8"
        )
        filename_only = key.split("/")[-1]
        download_path = f"/tmp/{filename_only}"

        print(f"[INFO] Processing: s3://{bucket}/{key}")

        # =====================================================
        # 2. DOWNLOAD FILE & COMPUTE AUDIO HASH
        # =====================================================
        s3.download_file(bucket, key, download_path)
        print("[INFO] File downloaded successfully.")

        with open(download_path, "rb") as f:
            audio_bytes = f.read()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        print(f"[INFO] Audio hash: {audio_hash}")

        # =====================================================
        # 3. CONNECT TO DATABASE & CHECK SCHEMA
        # =====================================================
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cursor = conn.cursor()

        has_audio_hash = safe_get_column(cursor, "call_analytics", "audio_hash")
        has_transcript_hash = safe_get_column(cursor, "call_analytics", "transcript_hash")
        has_duplicate_count = safe_get_column(cursor, "call_analytics", "duplicate_count")
        has_last_seen_at = safe_get_column(cursor, "call_analytics", "last_seen_at")
        has_last_filename = safe_get_column(cursor, "call_analytics", "last_uploaded_filename")
        has_normalized = safe_get_column(cursor, "call_analytics", "normalized_transcript")

        print(f"[INFO] Schema check: audio_hash={has_audio_hash}, "
              f"transcript_hash={has_transcript_hash}, "
              f"duplicate_count={has_duplicate_count}, "
              f"last_seen_at={has_last_seen_at}")

        # =====================================================
        # 4. EXACT AUDIO DUPLICATE CHECK
        # =====================================================
        if has_audio_hash:
            cursor.execute("""
                SELECT id, duplicate_count FROM call_analytics
                WHERE audio_hash = %s LIMIT 1
            """, (audio_hash,))
            exact_dup = cursor.fetchone()

            if exact_dup:
                existing_id, current_dup_count = exact_dup
                print(f"[INFO] Exact audio duplicate found. Row ID: {existing_id}")

                if has_duplicate_count and has_last_seen_at and has_last_filename:
                    cursor.execute("""
                        UPDATE call_analytics
                        SET duplicate_count = COALESCE(duplicate_count, 0) + 1,
                            last_seen_at = CURRENT_TIMESTAMP,
                            last_uploaded_filename = %s
                        WHERE id = %s
                    """, (filename_only, existing_id))
                else:
                    # Minimal update if advanced columns don't exist
                    cursor.execute("""
                        UPDATE call_analytics
                        SET filename = %s
                        WHERE id = %s
                    """, (filename_only, existing_id))

                conn.commit()
                print("[INFO] Duplicate count incremented.")
                return {"statusCode": 200, "body": "Exact duplicate. Updated."}

        # =====================================================
        # 5. WHISPER TRANSCRIPTION (Multilingual)
        # =====================================================
        groq_api_key = os.environ["GROQ_API_KEY"]
        headers = {"Authorization": f"Bearer {groq_api_key}"}

        print("[INFO] Calling Groq Whisper API...")
        with open(download_path, "rb") as file:
            files = {"file": (filename_only, file)}
            data = {"model": "whisper-large-v3"}
            whisper_resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers, files=files, data=data
            )

        whisper_resp.raise_for_status()
        transcript = whisper_resp.json()["text"]
        print(f"[INFO] Transcription: {transcript[:120]}...")

        # =====================================================
        # 6. TRANSCRIPT CONTENT DUPLICATE CHECK
        # =====================================================
        normalized_transcript = normalize_text(transcript)
        transcript_hash = hashlib.sha256(normalized_transcript.encode("utf-8")).hexdigest()

        if has_transcript_hash:
            cursor.execute("""
                SELECT id, duplicate_count FROM call_analytics
                WHERE transcript_hash = %s LIMIT 1
            """, (transcript_hash,))
            content_dup = cursor.fetchone()

            if content_dup:
                existing_id, _ = content_dup
                print(f"[INFO] Transcript duplicate found. Row ID: {existing_id}")

                if has_duplicate_count and has_last_seen_at and has_last_filename:
                    cursor.execute("""
                        UPDATE call_analytics
                        SET duplicate_count = COALESCE(duplicate_count, 0) + 1,
                            last_seen_at = CURRENT_TIMESTAMP,
                            last_uploaded_filename = %s
                        WHERE id = %s
                    """, (filename_only, existing_id))
                else:
                    cursor.execute("""
                        UPDATE call_analytics SET filename = %s WHERE id = %s
                    """, (filename_only, existing_id))

                conn.commit()
                print("[INFO] Duplicate count incremented.")
                return {"statusCode": 200, "body": "Content duplicate. Updated."}

        # =====================================================
        # 7. LLAMA ANALYSIS (Multilingual, English Topic)
        # =====================================================
        prompt = f"""You are an expert customer support analyst.

Analyze the following customer support transcript. The transcript can be in any language (English, Hindi, Spanish, French, etc.), but you must understand the meaning.

Return ONLY a valid JSON object with exactly these three keys:

1. "customer_sentiment": Must be exactly "positive", "negative", or "neutral"
2. "topic": Must be in English. Choose from: billing_issue, technical_support, refund, pricing_inquiry, login_issue, service_cancellation, shipping_delay, investment_opportunity, product_complaint, other
3. "problem_resolved": Must be exactly true or false

Rules:
- Understand sentiment and context from any language
- Always return "topic" in English only
- Set problem_resolved=true only if the agent clearly solved the issue before the call ended

Transcript:
{transcript}
"""

        llama_data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a precise JSON-only customer support analyzer."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        print("[INFO] Calling Groq Llama API...")
        llama_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=llama_data
        )

        if llama_resp.status_code != 200:
            print(f"[ERROR] Llama rejected: {llama_resp.text}")

        llama_resp.raise_for_status()

        analysis_content = llama_resp.json()["choices"][0]["message"]["content"]
        analysis = json.loads(analysis_content)
        print(f"[INFO] Analysis: {analysis}")

        # Normalize values for DB
        customer_sentiment = str(analysis.get("customer_sentiment", "neutral")).strip().lower()
        topic = str(analysis.get("topic", "other")).strip().lower()

        raw_resolved = analysis.get("problem_resolved", False)
        if isinstance(raw_resolved, str):
            problem_resolved = raw_resolved.strip().lower() == "true"
        else:
            problem_resolved = bool(raw_resolved)

        # =====================================================
        # 8. INSERT NEW UNIQUE ROW (only if all advanced cols exist)
        # =====================================================
        if has_audio_hash and has_transcript_hash:
            cursor.execute("""
                INSERT INTO call_analytics (
                    filename, last_uploaded_filename, transcript, normalized_transcript,
                    audio_hash, transcript_hash, customer_sentiment, topic,
                    problem_resolved, duplicate_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
            """, (
                filename_only, filename_only, transcript, normalized_transcript,
                audio_hash, transcript_hash, customer_sentiment, topic, problem_resolved
            ))
        else:
            # Fallback for old schema
            cursor.execute("""
                INSERT INTO call_analytics (
                    filename, transcript, customer_sentiment, topic, problem_resolved
                )
                VALUES (%s, %s, %s, %s, %s)
            """, (filename_only, transcript, customer_sentiment, topic, problem_resolved))

        conn.commit()
        print("[INFO] Data successfully saved.")

        return {"statusCode": 200, "body": "Success"}

    except Exception as e:
        print(f"[FATAL] Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
