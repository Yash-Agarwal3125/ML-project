import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "features": [2, 5, 1, 2.33, 1, 0]  # example input
}

response = requests.post(url, json=data)
print("Predicted price:", response.json())
