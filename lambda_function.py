import json
import urllib.parse
import boto3
import os
import requests
import psycopg2
import hashlib
import re
import numpy as np
import pickle

s3 = boto3.client("s3")

# ==================== LOAD LIGHTWEIGHT SENTIMENT MODEL ====================
print("Loading lightweight sentiment model...")
with open("sentiment_light.pkl", "rb") as f:
    LIGHT_MODEL = pickle.load(f)

VOCABULARY = LIGHT_MODEL["vocabulary"]
IDF = np.array(LIGHT_MODEL["idf"])
COEF = np.array(LIGHT_MODEL["coef"])
INTERCEPT = np.array(LIGHT_MODEL["intercept"])
CLASSES = LIGHT_MODEL["classes"]

print("Lightweight sentiment model loaded successfully.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def predict_sentiment(text):
    """Predict sentiment using lightweight NumPy model"""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    
    # Create TF-IDF vector (sparse simulation)
    vector = np.zeros(len(VOCABULARY))
    for token in tokens:
        if token in VOCABULARY:
            idx = VOCABULARY[token]
            vector[idx] = 1.0
    
    # TF-IDF transformation
    vector = vector * IDF
    
    # Logistic Regression prediction
    scores = np.dot(COEF, vector) + INTERCEPT
    pred_idx = np.argmax(scores)
    return CLASSES[pred_idx]

def lambda_handler(event, context):
    conn = None
    cursor = None

    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"], encoding="utf-8")
        filename_only = key.split("/")[-1]
        download_path = f"/tmp/{filename_only}"

        print(f"STEP 1: File received: s3://{bucket}/{key}")

        s3.download_file(bucket, key, download_path)
        print("STEP 2: Downloaded successfully.")

        with open(download_path, "rb") as f:
            audio_hash = hashlib.md5(f.read()).hexdigest()
        print(f"STEP 3: Audio hash: {audio_hash}")

        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL is not set.")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        print("STEP 4: DB connected.")

        # Duplicate check
        cursor.execute("SELECT id FROM call_analytics WHERE audio_hash = %s LIMIT 1", (audio_hash,))
        if cursor.fetchone():
            print("STEP 5: Duplicate audio. Skipping.")
            return {"statusCode": 200, "body": "Duplicate skipped."}

        # Whisper Translation (Always English)
        headers = {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"}
        print("STEP 6: Calling Whisper Translation...")

        with open(download_path, "rb") as audio_file:
            whisper_resp = requests.post(
                "https://api.groq.com/openai/v1/audio/translations",
                headers=headers,
                files={"file": (filename_only, audio_file)},
                data={"model": "whisper-large-v3"}
            )
        whisper_resp.raise_for_status()
        transcript = whisper_resp.json().get("text", "").strip()
        print(f"STEP 6: English transcript: {transcript[:100]}...")

        # Content duplicate check
        normalized = normalize_text(transcript)
        transcript_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

        cursor.execute("SELECT id FROM call_analytics WHERE transcript_hash = %s LIMIT 1", (transcript_hash,))
        if cursor.fetchone():
            print("STEP 7: Content duplicate. Skipping.")
            return {"statusCode": 200, "body": "Content duplicate skipped."}

        # === ML SENTIMENT PREDICTION (The Change) ===
        print("STEP 8: Predicting sentiment with ML model...")
        sentiment = predict_sentiment(transcript)
        print(f"STEP 8: ML Sentiment = {sentiment}")

        # Llama for Topic + Resolution
        print("STEP 9: Calling Llama for topic & resolution...")
        llama_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "Return only valid JSON in English."},
                    {"role": "user", "content": f"""
Analyze this transcript and return ONLY this JSON:
{{
  "topic": one of: billing_issue, technical_support, refund, pricing_inquiry, login_issue, service_cancellation, shipping_delay, product_complaint, other,
  "problem_resolved": true or false
}}
Transcript: {transcript}
"""}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            }
        )
        analysis = json.loads(llama_resp.json()["choices"][0]["message"]["content"])

        topic = str(analysis.get("topic", "other")).strip().lower()
        resolved = analysis.get("problem_resolved", False)
        if isinstance(resolved, str):
            resolved = resolved.strip().lower() == "true"

        # Final Insert
        print("STEP 10: Saving to database...")
        cursor.execute("""
            INSERT INTO call_analytics (
                filename, topic, customer_sentiment, problem_resolved, 
                transcript, audio_hash, transcript_hash, processed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """, (filename_only, topic, sentiment, resolved, transcript, audio_hash, transcript_hash))

        conn.commit()
        print("✅ Successfully saved to database.")
        return {"statusCode": 200, "body": "Success"}

    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {str(e)}")
        raise e
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
