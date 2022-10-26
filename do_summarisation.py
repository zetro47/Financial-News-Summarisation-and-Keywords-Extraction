from datetime import datetime, timezone
import os
from newspaper import Article, Config
import requests
import json
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.documents as documents
import azure.cosmos.http_constants as http_constants
from uuid import uuid4
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
num_add_toks = tokenizer.add_special_tokens(special_tokens)

#model = Transformer(embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads,
#                   num_encoder_layers, num_decoder_layers, froward_expansion, dropout, max_len, device).to(device)
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

model.load_state_dict(torch.load("/Users/zetro7744/Downloads/avalon.pth", map_location=torch.device('cpu')))
model.eval()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_seq(model, context, length, device, temperature=1, top_k=0, top_p=0.0):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():  
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

test_sentence = ''' In first half of 20th century, factions of Indian National Congress continued to remain getting identified with "Hindu politics" and ideas of a Hindu nation. The word "Hindu", throughout history, had been used as an inclusive description that lacked a definition and was used to refer to the native traditions and people of India. It was only in the late 18th century that the word "Hindu" came to be used extensively with religious connotation, while still being used as a synecdoche describing the indigenous traditions. Hindu nationalist ideologies and political languages were very diverse both linguistically and socially. Since Hinduism does not represent an identifiable religious group, the terms such as 'Hindu nationalism', 'Hindu', are considered problematic in the case of religious and nationalism discourse. As Hindus were identifiable as a homogeneous community, some individual Congress leaders were able to induce a symbolism with "Hindu" meaning inside the general stance of a secular nationalism.[12][13]

The diversity of Indian cultural groups and moderate positions of Hindu nationalism have sometimes made it regarded as cultural nationalism than a religious one.'''

#context = (tokenizer.encode(test_sentence.text[:1023]))
#generated_text = sample_seq(model, context, 100, device, 1, 10, 0.5)
#generated_text = generated_text[0, len(context):].tolist()
#text = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
#text = tokenizer.convert_tokens_to_string(text)
#print(text)



def utc_to_local(utc_dt):
    if(((  datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()) > 0):
        print("Yes")
    return ( datetime.now() - datetime.strptime(utc_dt, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()



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
    news_extract = Article(art_link, language="en", config = config) 
    news_extract.download()
    news_extract.parse()
    context = (tokenizer.encode(news_extract.text[:1023]))
    generated_text = sample_seq(model, context, 100, device, 1, 10, 0.5)
    generated_text = generated_text[0, len(context):].tolist()
    text = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
    summary = tokenizer.convert_tokens_to_string(text)
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
