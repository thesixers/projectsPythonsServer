import requests

# Define the URL for the prediction
url = 'http://127.0.0.1:5000/predict'

# Create a dummy data payload
data = {
    'weight': 70,         # Example weight in kg
    'sleepHours': 8,      # Example sleep hours
    'bpm': 75,            # Example beats per minute
    'systolic': 50,      # Example systolic blood pressure
    'diastolic': 20       # Example diastolic blood pressure
}

# Send a POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
