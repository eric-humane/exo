#!/usr/bin/env python3
"""
Test script to check if we can make API calls to OpenAI models through tinychat.
"""

import requests
import json
import os

# Endpoint - tinychat is served on a random port, but we can find it from the console output
TINYCHAT_PORT = input("Enter the tinychat port (from console output): ")

# Construct the URL
API_URL = f"http://localhost:{TINYCHAT_PORT}/v1/chat/completions"

# Prepare request data
data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! What's your name?"}
    ],
    "temperature": 0.7,
    "max_tokens": 50
}

# Make request
print(f"Sending completion request to {API_URL}...")
response = requests.post(API_URL, json=data)

# Print response
print(f"Status code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print("\nResponse:")
    print(json.dumps(result, indent=2))
    
    # Extract just the assistant's message
    content = result['choices'][0]['message']['content']
    print("\nAssistant's message:")
    print(content)
else:
    print(f"Error: {response.text}")