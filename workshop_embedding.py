import openai
from openai.embeddings_utils import get_embedding, cosine_similarity # must pip install openai[embeddings]
import pandas as pd
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
import time


# load in variables from .env 
load_dotenv()


# set keys and configure Azure OpenAI
openai.api_type = 'azure'
openai.api_version = os.environ['AZURE_OPENAI_VERSION']
openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
openai.api_key = os.environ['AZURE_OPENAI_KEY']

 
# import os
# import openai
# openai.api_type = "azure"
# openai.api_base = "https://ss-aoai-ce.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
# openai.api_key = os.getenv("OPENAI_API_KEY")

# response = openai.ChatCompletion.create(
#   engine="ss-gpt-4-ce",
#   messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"Who is the CEO of Microsoft"}],
#   temperature=0.7,
#   max_tokens=800,
#   top_p=0.95,
#   frequency_penalty=0,
#   presence_penalty=0,
#   stop=None)

# print(response)

# read the data file to be embed 
df = pd.read_csv('microsoft-earnings.csv')
print(df)


# calculate word embeddings
df['embedding'] = df['text'].apply(lambda x:get_embedding(x, engine='text-embedding-ada-002-ce'))
df.to_csv('microsoft-earnings_embeddings.csv')
time.sleep(3)
print(df)



