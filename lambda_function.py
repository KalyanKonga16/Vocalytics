import json
import urllib.parse
import boto3
import os
import requests
import psycopg2

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # 1. Get the bucket and file name from the S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    download_path = f'/tmp/{key}'
    
    # 2. Download file to Lambda's temporary storage
    s3.download_file(bucket, key, download_path)
    
    # 3. Transcribe Audio using Groq Whisper API
    groq_api_key = os.environ['GROQ_API_KEY']
    headers = {"Authorization": f"Bearer {groq_api_key}"}
    
    with open(download_path, "rb") as file:
        files = {"file": (key, file)}
        data = {"model": "whisper-large-v3"}
        response = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers, files=files, data=data)
    
    transcript = response.json()['text']
    
    # 4. Analyze with Llama 3
           # 4. Analyze with Llama 3.1 - MULTILINGUAL VERSION
        prompt = f"""
        You are an expert customer support analyst. Analyze the following transcript.

        Transcript Language: Detect automatically.
        Task: Return ONLY a valid JSON object with exactly these keys:
        - "customer_sentiment": "positive", "negative", or "neutral"
        - "topic": Use English only. Common topics: billing_issue, technical_support, refund, pricing_inquiry, product_complaint, service_cancellation, login_issue, shipping_delay, other.
        - "problem_resolved": true or false

        Rules:
        - Understand the sentiment and context even if the transcript is in Hindi, Tamil, Spanish, French, or any other language.
        - Always return topic in English.
        - Be accurate with whether the agent's solution satisfied the customer.

        Transcript:
        {transcript}
        """

        llama_data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a precise JSON-only customer support analyzer. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 300
        }
    
    llama_resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=llama_data)
    analysis = json.loads(llama_resp.json()['choices'][0]['message']['content'])
    
    # 5. Save to PostgreSQL Database
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO call_analytics (filename, transcript, customer_sentiment, topic, problem_resolved)
        VALUES (%s, %s, %s, %s, %s)
    """, (key, transcript, analysis['customer_sentiment'], analysis['topic'], analysis['problem_resolved']))
    conn.commit()
    cursor.close()
    conn.close()
    
    return {"statusCode": 200, "body": "Success"}
