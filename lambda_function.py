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
    """Normalize transcript for content-based dedup comparison"""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def lambda_handler(event, context):
    conn = None
    cursor = None

    try:
        # STEP 1: Extract S3 file info
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(
            event["Records"][0]["s3"]["object"]["key"],
            encoding="utf-8"
        )
        filename_only = key.split("/")[-1]
        download_path = f"/tmp/{filename_only}"
        print(f"STEP 1: File received: s3://{bucket}/{key}")

        # STEP 2: Download audio file
        s3.download_file(bucket, key, download_path)
        print("STEP 2: Downloaded successfully.")

        # STEP 3: Compute audio hash for exact dedup
        with open(download_path, "rb") as f:
            audio_hash = hashlib.md5(f.read()).hexdigest()
        print(f"STEP 3: Audio hash: {audio_hash}")

        # STEP 4: Connect to database
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL is not set.")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        print("STEP 4: DB connected.")

        # STEP 5: Exact audio duplicate check
        cursor.execute(
            "SELECT id FROM call_analytics WHERE audio_hash = %s LIMIT 1",
            (audio_hash,)
        )
        if cursor.fetchone():
            print("STEP 5: Exact audio duplicate. Skipping.")
            return {"statusCode": 200, "body": "Duplicate audio. Skipped."}
        print("STEP 5: No audio duplicate found.")

        # STEP 6: Transcribe with Groq Whisper (FORCE ENGLISH OUTPUT)
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set.")

        headers = {"Authorization": f"Bearer {groq_key}"}
        print("STEP 6: Calling Whisper (English translation mode)...")

        with open(download_path, "rb") as audio_file:
            whisper_resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files={"file": (filename_only, audio_file)},
                data={
                    "model": "whisper-large-v3",
                    "task": "translate",          # <-- FORCES ENGLISH OUTPUT
                    "response_format": "json"     # <-- ensures clean JSON return
                }
            )

        if whisper_resp.status_code != 200:
            print(f"Whisper error: {whisper_resp.text}")
        whisper_resp.raise_for_status()

        transcript = whisper_resp.json().get("text", "").strip()
        print(f"STEP 6: English Transcript: {transcript[:100]}...")
        
        # STEP 7: Content-based duplicate check
        normalized = normalize_text(transcript)
        transcript_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

        cursor.execute(
            "SELECT id FROM call_analytics WHERE transcript_hash = %s LIMIT 1",
            (transcript_hash,)
        )
        if cursor.fetchone():
            print("STEP 7: Same content already analyzed. Skipping.")
            return {"statusCode": 200, "body": "Duplicate content. Skipped."}
        print("STEP 7: No content duplicate found.")

        # STEP 8: Analyze with Llama 3.1
        print("STEP 8: Calling Llama...")
        llama_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a customer support call analyst. "
                            "Always respond with valid JSON only. "
                            "All output values must be in English regardless of transcript language."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
Analyze this customer support call transcript carefully.
The audio may be in any language. Return values in English only.

Return ONLY this exact JSON:
{{
  "customer_sentiment": "positive" or "negative" or "neutral",
  "topic": exactly one of: billing_issue, technical_support, refund, pricing_inquiry, login_issue, service_cancellation, shipping_delay, product_complaint, other,
  "problem_resolved": true or false
}}

Base problem_resolved on whether the agent fully solved the issue before the call ended.

Transcript:
{transcript}
"""
                    }
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "max_tokens": 150
            }
        )

        if llama_resp.status_code != 200:
            print(f"Llama error: {llama_resp.text}")
        llama_resp.raise_for_status()

        analysis = json.loads(llama_resp.json()["choices"][0]["message"]["content"])
        print(f"STEP 8: Analysis result: {analysis}")

        # STEP 9: Sanitize values
        sentiment = str(analysis.get("customer_sentiment", "neutral")).strip().lower()
        topic = str(analysis.get("topic", "other")).strip().lower()
        resolved = analysis.get("problem_resolved", False)
        if isinstance(resolved, str):
            resolved = resolved.strip().lower() == "true"

        # STEP 10: Insert into database
        print("STEP 10: Inserting into DB...")
        cursor.execute("""
            INSERT INTO call_analytics (
                filename,
                topic,
                customer_sentiment,
                problem_resolved,
                transcript,
                processed_at,
                audio_hash,
                transcript_hash
            )
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s)
        """, (
            filename_only,
            topic,
            sentiment,
            resolved,
            transcript,
            audio_hash,
            transcript_hash
        ))

        conn.commit()
        print("STEP 10: Successfully saved to DB.")
        return {"statusCode": 200, "body": "Success"}

    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {str(e)}")
        raise e

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("FINAL: DB connection closed.")
