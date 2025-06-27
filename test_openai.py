import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key from env: {api_key[:20]}...{api_key[-10:] if api_key else 'None'}")
print(f"Key length: {len(api_key) if api_key else 0}")

if api_key and api_key.startswith('sk-'):
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ API Key works!")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ API Error: {e}")
else:
    print("❌ Invalid API key format")
