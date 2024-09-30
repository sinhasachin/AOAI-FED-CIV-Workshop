"""
This file contains a script for searching and retrieving information using Azure OpenAI's text embedding model.
The script performs the following steps:
1. Imports necessary libraries and modules.
2. Configures the Azure OpenAI client using the provided API version, endpoint, and API key.
3. Reads in the embeddings from a CSV file and converts the 'embedding' column values to numpy arrays.
4. Defines a function to get embeddings using the Azure OpenAI client.
5. Prompts the user to enter a search term.
6. Calculates the embedding vector for the search term using the defined function.
7. Calculates the cosine similarity between the search term vector and the vectors in the dataframe.
8. Sorts the dataframe based on similarity scores and selects the top 5 most similar entries.
9. Prints the most similar entry, its similarity score, and a newline character.

Note: This script requires the Azure OpenAI API version, endpoint, and API key to be set as environment variables.
"""
import os
import numpy as np
from openai import AzureOpenAI
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# load in variables from .env 
load_dotenv()

# configure Azure OpenAI client
client = AzureOpenAI(api_version=os.environ['AZURE_OPENAI_VERSION'],
                    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                    api_key=os.environ['AZURE_OPENAI_KEY'])

gpt_model_deployment = os.environ['AZURE_GPT_DEPLOYMENT']
embedding_model_deployment = os.environ['AZURE_EMBEDDINGS_DEPLOYMENT']

# read in the embeddings .csv 
# convert elements in 'embedding' column back to numpy array
df = pd.read_csv('microsoft-earnings_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

# Function to get embeddings
def get_embedding(text):
    response = client.embeddings.create(model=embedding_model_deployment, input=text)
    return response.data[0].embedding

# calculate user query embedding 
search_term = input("Enter a search term: ")
if search_term:
    search_term_vector = np.array(get_embedding(search_term))

    # find similarity between query and vectors 
    df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x.reshape(1, -1), search_term_vector.reshape(1, -1))[0][0])
    df1 = df.sort_values("similarities", ascending=False).head(5)

    # output the response 
    print('\n')
    print('Answer: ', df1['text'].loc[df1.index[0]])
    print('\n')
    print('Similarity Score: ', df1['similarities'].loc[df1.index[0]]) 
    print('\n')



  # send search_term to Azure OpenAI as prompt and df1 as context to generate output
    response = client.chat.completions.create(
        model=gpt_model_deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You provide answers to questions based on the information in the data. However, your answers are really succinct and to the point in as few words as possible."},
            {"role": "user", "content": f"{search_term}\n\n{df1['text'].loc[df1.index[0]]}"}
        ],
        max_tokens=50
    )

    # print the generated output
    print('\n')
    print('Generated Output: ', response.choices[0].message.content)
    print('\n')
