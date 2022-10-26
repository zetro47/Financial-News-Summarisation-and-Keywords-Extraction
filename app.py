from flask import Flask, jsonify

import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.documents as documents
import azure.cosmos.http_constants as http_constants
from uuid import uuid4


app = Flask(__name__)

@app.route('/', methods=['GET'])
def getHeadlines():
    config = {
    "endpoint": "https://bdadata.documents.azure.com:443/",
    "primarykey": "BV5BBMTsSqC8zSRuYxK6h4qY4Vx7wodnKFe4hJ7svs2DkK6UCrfQGhEQee76hsNLIUOb9VaWfB6iB71bQyuMMg=="
    }
    client = cosmos_client.CosmosClient(url=config["endpoint"], credential=config["primarykey"])
    db = client.get_database_client("BDA")
    summaries_collection = db.get_container_client("Summaries")

    #print(collection_name.find_one(sort=[( '_id', pymongo.DESCENDING )])    )
    headlines = []
    for summary in summaries_collection.read_all_items():
        headlines.append({"Title": summary["Title"], \
                        "Date": summary["Date"], \
                        "Summary": summary["Summary"], \
                        "Link": summary["Link"] \
                        })
    return jsonify(headlines)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)

