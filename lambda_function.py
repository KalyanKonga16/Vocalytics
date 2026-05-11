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
    prompt = f"""
    Analyze this customer support transcript. Return ONLY a valid JSON object with exactly these keys:
    "customer_sentiment" (positive, negative, or neutral),
    "topic" (e.g., billing_issue, technical_support, refund, etc.),
    "problem_resolved" (true or false).
    Transcript: {transcript}
    """
    
    llama_data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
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