from datetime import datetime, timezone
from transformers import pipeline
import os
from newspaper import Article, Config
import requests

import json
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.documents as documents
import azure.cosmos.http_constants as http_constants
from uuid import uuid4

def utc_to_local(utc_dt):
    if(((  datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()) > 0):
        print("Yes")
    return ( datetime.now() - datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()


## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
summarizer = pipeline("summarization")

config = {
    "endpoint": "https://bdadata.documents.azure.com:443/",
    "primarykey": "BV5BBMTsSqC8zSRuYxK6h4qY4Vx7wodnKFe4hJ7svs2DkK6UCrfQGhEQee76hsNLIUOb9VaWfB6iB71bQyuMMg=="
}
client = cosmos_client.CosmosClient(url=config["endpoint"], credential=config["primarykey"])
db = client.get_database_client("BDA")
summaries_collection = db.get_container_client("Summaries")

url = ('https://newsapi.org/v2/top-headlines?category=business&country=in&apiKey=2ceabdef954d44be8274db059da36369')
response = requests.get(url)

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

summaries= []
i = 0
for article in (response.json())['articles']:
    art_link = article["url"]
    if("youtube" in art_link):
        print("shit")
        continue
    news_extract = Article(art_link, language="en", config = config) # en for English
    news_extract.download()
    news_extract.parse()
    summary = summarizer(news_extract.text[:3423], max_length=100, min_length=50, do_sample=False)[0]['summary_text']
    summaries.append(summary)
    summaries_collection.create_item({"id": str(uuid4()),  \
                                        "Title": article["title"], \
                                        "Date": article["publishedAt"], \
                                        "Summary": summary, \
                                        "Link": article["url"]})
    print(i)
    i = i+1
    if(i==20):
      break

print(summaries)