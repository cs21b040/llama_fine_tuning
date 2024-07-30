import requests

url = 'http://localhost:5000/romanize'
url2='http://localhost:5000/transliterate'
payload = {
    'input': 'ඔයාට කොහොම ද',
}
payload2={
    'input':'oyata kohoma dha'
}
response = requests.post(url, json=payload)
response2 = requests.post(url2, json=payload2)

print(response.json()['romanized'])
print(response2.json()['transliterated'])

