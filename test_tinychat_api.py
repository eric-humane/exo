#!/usr/bin/env python3
"""
Test script to check if the OpenAI API is working in tinychat.
"""

import requests
import json
import os

# Endpoint - tinychat is served on a random port, but we can find it from the console output
TINYCHAT_PORT = input("Enter the tinychat port (from console output): ")

# Construct the URL
API_URL = f"http://localhost:{TINYCHAT_PORT}/v1/models"

# Make request
print(f"Querying models from {API_URL}...")
response = requests.get(API_URL)

# Print response
print(f"Status code: {response.status_code}")
if response.status_code == 200:
    models = response.json()
    print(f"Found {len(models['data'])} models:")
    
    # Print all models
    for model in models['data']:
        print(f"  - {model['id']} (owned by {model.get('owned_by', 'unknown')})")
        
    # Print OpenAI models specifically
    openai_models = [m for m in models['data'] if m.get('owned_by') == 'openai']
    print(f"\nOpenAI models ({len(openai_models)}):")
    for model in openai_models:
        print(f"  - {model['id']}")
else:
    print(f"Error: {response.text}")