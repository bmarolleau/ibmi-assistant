# Import environment loading library
from dotenv import load_dotenv
# Import IBMGen Library 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms.base import LLM
# Import lang Chain Interface object
from langChainInterface import LangChainInterface
# Import langchain prompt templates
from langchain.prompts import PromptTemplate
# Import system libraries
import os
# Import streamlit for the UI 
import streamlit as st

#import fitz
import os
import re
import requests

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from chromadb.api.types import EmbeddingFunction
from dotenv import load_dotenv

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from typing import Literal, Optional, Any

def pdf_to_text(path: str, 
                start_page: int = 1, 
                end_page: Optional[int | None] = None) -> list[str]:
    """
    Converts PDF to plain text.

    Params:
        path (str): Path to the PDF file.
        start_page (int): Page to start getting text from.
        end_page (int): Last page to get text from.
    """
    loader = PyPDFLoader("rbam6pdf.pdf")
    pages = loader.load()
    total_pages = len(pages)

    if end_page is None:
        end_page = len(pages)

    text_list = []
    for i in range(start_page-1, end_page):
        text = pages[i].page_content
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text_list.append(text)

    return text_list
def text_to_chunks(texts: list[str], 
                   word_length: int = 150, 
                   start_page: int = 1) -> list[list[str]]:
    """
    Splits the text into equally distributed chunks.

    Args:
        texts (str): List of texts to be converted into chunks.
        word_length (int): Maximum number of words in each chunk.
        start_page (int): Starting page number for the chunks.
    """
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip() 
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
            
    return chunks
def get_text_embedding(texts: list[list[str]], 
                       batch: int = 1000) -> list[Any]:
        """
        Get the embeddings from the text.

        Args:
            texts (list(str)): List of chucks of text.
            batch (int): Batch size.
        """
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            # Embeddings model
            emb_batch = emb_function(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
emb_function = MiniLML6V2EmbeddingFunction()

def build_prompt(question):
    prompt = ""
    prompt += 'Search results:\n'
    
    for c in topn_chunks:
        prompt += c + '\n\n'
    
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
            "Cite each reference at the end of each sentence. If the search results mention multiple subjects "\
            "with the same name, create separate answers for each. Only include information found in the results and "\
            "don't add any additional information. Make sure the answer is correct and don't output false content. "\
            "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
            "search results which has nothing to do with the question. Only answer what is asked. The "\
            "answer should be short and concise."
    
    prompt += f"\n\n\nQuery: {question}\n\nAnswer: "
    
    return prompt

# Load environment vars
load_dotenv()

# Define credentials 
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

text_list = pdf_to_text("rbam6pdf.pdf")
print(text_list)


chunks = text_to_chunks(text_list)
for chunk in chunks:
    print(chunk + '\n')

embeddings = get_text_embedding(chunks)

print(embeddings.shape)
print(f"Our text was embedded into {embeddings.shape[1]} dimensions")

# Define generation parameters 
params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 30,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.TEMPERATURE: 1.0,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
}

# define LangChainInterface model
llm = LangChainInterface(model='google/flan-ul2', credentials=creds, params=params, project_id=project_id)

# Title for the app
st.title('🤖 IBM i AI Assistant')
# Prompt box 
question = st.text_input('Enter your question here')

# If a user hits enter
if question: 
    emb_question = emb_function([question])
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(embeddings)
    neighbors = nn.kneighbors(emb_question, return_distance=False)
    neighbors
    topn_chunks = [chunks[i] for i in neighbors.tolist()[0]]
    prompt = build_prompt(question)
    print(prompt)
   
    #print(context)
    
    # Pass the prompt to the llm
    response = llm(prompt)
    # Write the output to the screen
    st.write(response)