"""
This script uses the Azure OpenAI API to generate text embeddings for a given dataset.

Functions:
- get_embedding(text, engine='text-embedding-ada-002-ce'): Retrieves the embedding for a given text using the specified engine.
    - text: The input text to generate the embedding for.
    - engine: The engine to use for generating the embedding (default: 'text-embedding-ada-002-ce').

Variables:
- client: An instance of the AzureOpenAI class configured with the API version, endpoint, and key from the environment variables.
- df: A pandas DataFrame containing the data read from the 'microsoft-earnings.csv' file.
    - Columns: 'text', 'embedding'
- load_dotenv(): Loads the environment variables from the '.env' file.

Usage:
1. Set the required environment variables: AZURE_OPENAI_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY.
2. Call the 'get_embedding' function to generate embeddings for the text data in the DataFrame.
3. Save the DataFrame with the embeddings to a new CSV file named 'microsoft-earnings_embeddings.csv'.
"""

from openai import AzureOpenAI
import pandas as pd
import os
from dotenv import load_dotenv
import time


# load in variables from .env 
load_dotenv()


# configure Azure OpenAI client
client = AzureOpenAI(api_version=os.environ['AZURE_OPENAI_VERSION'],
azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
api_key=os.environ['AZURE_OPENAI_KEY'])


# Function to get embeddings
def get_embedding(text, engine='text-embedding-ada-002-ce'):
    response = client.embeddings.create(model='text-embedding-ada-002-ce',
        input=text
    )
    return response.data[0].embedding


# read the data file to be embed 
df = pd.read_csv('microsoft-earnings.csv')
print(df)


# calculate word embeddings 
df['embedding'] = df['text'].apply(lambda x: get_embedding(x))
df.to_csv('microsoft-earnings_embeddings.csv')
time.sleep(3)
print(df)



