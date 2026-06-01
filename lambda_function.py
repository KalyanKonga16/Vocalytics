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
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def lambda_handler(event, context):
    conn = None
    cursor = None

    try:
        # 1. Get S3 event data
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(
            event["Records"][0]["s3"]["object"]["key"],
            encoding="utf-8"
        )
        filename_only = key.split("/")[-1]
        download_path = f"/tmp/{filename_only}"

        print(f"Processing S3 object: s3://{bucket}/{key}")

        # 2. Download file
        s3.download_file(bucket, key, download_path)
        print("File downloaded successfully.")

        # 3. Compute exact audio hash first
        with open(download_path, "rb") as f:
            audio_bytes = f.read()

        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        print(f"Audio hash: {audio_hash}")

        # 4. Connect to DB
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cursor = conn.cursor()

        # 5. Exact audio duplicate check
        cursor.execute("""
            SELECT id, duplicate_count
            FROM call_analytics
            WHERE audio_hash = %s
            LIMIT 1
        """, (audio_hash,))
        exact_duplicate = cursor.fetchone()

        if exact_duplicate:
            existing_id = exact_duplicate[0]

            cursor.execute("""
                UPDATE call_analytics
                SET duplicate_count = COALESCE(duplicate_count, 0) + 1,
                    last_seen_at = CURRENT_TIMESTAMP,
                    last_uploaded_filename = %s
                WHERE id = %s
            """, (filename_only, existing_id))
            conn.commit()

            print(f"Exact duplicate audio detected. Updated existing row id={existing_id}")
            return {
                "statusCode": 200,
                "body": "Exact duplicate detected. Existing record updated."
            }

        # 6. Whisper transcription
        groq_api_key = os.environ["GROQ_API_KEY"]
        headers = {"Authorization": f"Bearer {groq_api_key}"}

        print("Calling Groq Whisper API...")
        with open(download_path, "rb") as file:
            files = {"file": (filename_only, file)}
            data = {"model": "whisper-large-v3"}
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data
            )

        response.raise_for_status()
        transcript = response.json()["text"]
        print(f"Whisper success: {transcript[:100]}...")

        # 7. Transcript-based duplicate detection
        normalized_transcript = normalize_text(transcript)
        transcript_hash = hashlib.sha256(
            normalized_transcript.encode("utf-8")
        ).hexdigest()

        cursor.execute("""
            SELECT id, duplicate_count
            FROM call_analytics
            WHERE transcript_hash = %s
            LIMIT 1
        """, (transcript_hash,))
        content_duplicate = cursor.fetchone()

        if content_duplicate:
            existing_id = content_duplicate[0]

            cursor.execute("""
                UPDATE call_analytics
                SET duplicate_count = COALESCE(duplicate_count, 0) + 1,
                    last_seen_at = CURRENT_TIMESTAMP,
                    last_uploaded_filename = %s
                WHERE id = %s
            """, (filename_only, existing_id))
            conn.commit()

            print(f"Transcript duplicate detected. Updated existing row id={existing_id}")
            return {
                "statusCode": 200,
                "body": "Transcript duplicate detected. Existing record updated."
            }

        # 8. Llama analysis
        prompt = f"""
        Analyze this customer support transcript. Return ONLY a valid JSON object with exactly these keys:
        "customer_sentiment" (positive, negative, or neutral),
        "topic" (e.g., billing_issue, technical_support, refund, pricing_inquiry, login_issue, service_cancellation, shipping_delay, other),
        "problem_resolved" (true or false).

        Transcript:
        {transcript}
        """

        llama_data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful AI that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        print("Calling Groq Llama API...")
        llama_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=llama_data
        )

        if llama_resp.status_code != 200:
            print(f"Groq Llama error: {llama_resp.text}")

        llama_resp.raise_for_status()

        analysis = json.loads(llama_resp.json()["choices"][0]["message"]["content"])
        print(f"Llama success: {analysis}")

        # 9. Normalize boolean
        problem_resolved = analysis.get("problem_resolved", False)
        if isinstance(problem_resolved, str):
            problem_resolved = problem_resolved.strip().lower() == "true"

        customer_sentiment = str(analysis.get("customer_sentiment", "neutral")).strip().lower()
        topic = str(analysis.get("topic", "other")).strip().lower()

        # 10. Insert only unique analyzed row
        cursor.execute("""
            INSERT INTO call_analytics (
                filename,
                last_uploaded_filename,
                transcript,
                normalized_transcript,
                audio_hash,
                transcript_hash,
                customer_sentiment,
                topic,
                problem_resolved,
                duplicate_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
        """, (
            filename_only,
            filename_only,
            transcript,
            normalized_transcript,
            audio_hash,
            transcript_hash,
            customer_sentiment,
            topic,
            problem_resolved
        ))

        conn.commit()
        print("Data successfully saved to PostgreSQL.")

        return {
            "statusCode": 200,
            "body": "Success"
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise e

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
