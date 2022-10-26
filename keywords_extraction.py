import os
from re import I
from newspaper import Article, Config
import requests
from transformers import pipeline
import json
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.documents as documents
import azure.cosmos.http_constants as http_constants
from uuid import uuid4
from regex import R
from textblob import TextBlob


config = {
    "endpoint": "https://bdadata.documents.azure.com:443/",
    "primarykey": "BV5BBMTsSqC8zSRuYxK6h4qY4Vx7wodnKFe4hJ7svs2DkK6UCrfQGhEQee76hsNLIUOb9VaWfB6iB71bQyuMMg=="
}
client = cosmos_client.CosmosClient(url=config["endpoint"], credential=config["primarykey"])
db = client.get_database_client("BDA")
summaries_collection = db.get_container_client("Summaries")
edgelist = []
for summary in summaries_collection.read_all_items():
    blob = TextBlob(summary["Summary"])
    for i in range(0, len(list(blob.noun_phrases))):
        for noun in list(blob.noun_phrases):
            edgelist.append([blob.noun_phrases[i], noun])
    

import pandas as pd
df = pd.DataFrame(edgelist,columns=['Source','Dest'])
import networkx as nx
G = nx.from_pandas_edgelist(df, source="Source", target = "Dest")
from pyvis.network import Network
net = Network(notebook = True, height = '500px', width = '500px')
net.barnes_hut()
net.from_nx(G)
frame = pd.DataFrame((nx.pagerank_numpy(G).items()))
frame.columns = ["Item", "Rank"]

print((frame.sort_values(by = ["Rank"], ascending=False)).head())

net.show("eg.html")



def utc_to_local(utc_dt):
    if(((  datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()) > 0):
        print("Yes")
    return ( datetime.now() - datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()


## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
summarizer = pipeline("summarization")


keywords_collection = db.get_container_client("Summaries")
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10



for keyword in (((frame.sort_values(by = ["Rank"], ascending=False)).head())["Item"]):
    url = ('https://newsapi.org/v2/everything?q={0}&apiKey=2ceabdef954d44be8274db059da36369'.format(str(keyword)))
    print(url)
    response = requests.get(url)
    summaries= []
    i = 0
    for article in (response.json())['articles']:
        art_link = article["url"]
        if("youtube" in art_link):
            continue
        
        news_extract = Article(art_link, language="en", config = config) # en for English
        news_extract.download()
        news_extract.parse()
        summary = summarizer(news_extract.text[:3423], max_length=100, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
        keywords_collection.create_item({"id": str(uuid4()),  \
                                        "isKeyworded": True, \
                                        "Title": article["title"], \
                                        "Date": article["publishedAt"], \
                                        "Summary": summary, \
                                        "Link": article["url"]})
        i = i+1
        if(i==5):
            break

    print(summaries)

