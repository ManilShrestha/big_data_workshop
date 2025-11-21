import requests
from openai import OpenAI

# Test both endpoints
endpoints = [
    ('http://96.245.177.243:12410/v1', 'Port 12410'),
    ('http://96.245.177.243:12302/v1', 'Port 12302')
]

print("=" * 60)
print("Testing Model Endpoints")
print("=" * 60)

for base_url, name in endpoints:
    print(f"\n{name}: {base_url}")
    print("-" * 60)

    # Method 1: Using requests to call /v1/models
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                print(f"Available models ({len(data['data'])}):")
                for model in data['data']:
                    print(f"  - {model.get('id', 'unknown')}")
            else:
                print("Response:", data)
        else:
            print(f"Error: Status {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error with requests: {e}")

    # Method 2: Using OpenAI client
    print("\nUsing OpenAI client:")
    try:
        client = OpenAI(base_url=base_url, api_key="none")
        models = client.models.list()
        print(f"Available models ({len(models.data)}):")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e:
        print(f"Error with OpenAI client: {e}")

# Test the specific completion endpoint for port 12302
print("\n" + "=" * 60)
print("Testing Completion on Port 12302")
print("=" * 60)

try:
    response = requests.post(
        'http://96.245.177.243:12302/v1/completions',
        headers={'Content-Type': 'application/json'},
        json={
            'model': 'cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-8bit',
            'prompt': 'San Francisco is a',
            'max_tokens': 7,
            'temperature': 0
        },
        timeout=30
    )

    if response.status_code == 200:
        print("Success!")
        print("Response:", response.json())
    else:
        print(f"Error: Status {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
