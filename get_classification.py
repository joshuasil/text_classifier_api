import requests
import json

url = "https://text-classifier-blcz.onrender.com/c4hprediction"

payload = json.dumps({"text_to_classify": "How can i be fit?"})

headers = {"Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, data=payload)
response = json.loads(response.text)
print(response.get('prediction'))