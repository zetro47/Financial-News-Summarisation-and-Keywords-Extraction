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
print(frame.sort_values(by = ["Rank"], ascending=False))
net.show("eg.html")



