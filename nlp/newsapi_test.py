import requests

endpoint = "https://newsapi.org/v2/everything"

headers = {"x-api-key": "..."}
params = {
    "sources": "cnn",
    "language": "en",
    "q": "Tesla",
    "sortBy": "publishedAt"
}

# Fetch from newsapi.org
response = requests.get(endpoint, params=params, headers=headers)
data = response.json()

print(data)