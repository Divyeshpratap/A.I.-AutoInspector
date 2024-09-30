# models/nlp/nlp_model.py

import os
import numpy as np
import faiss
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
import logging
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path)

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
modelName = 'gpt-4o-mini'

# Initialize the appropriate LLM and embeddings based on the model name
if modelName.startswith("gpt"):
    LLM = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=modelName, temperature=0.2)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
else:
    LLM = Ollama(model=modelName)
    embeddings = OllamaEmbeddings(model=modelName, temperature=0.2)

# Functions related to LLM processing

def rephrase_question_with_history(chat_history, question):
    template = """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    
    <chat_history>
      {chat_history}
    </chat_history>
    
    Follow Up Input: {question}
    Standalone question:
    """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(chat_history=chat_history, question=question)
    response = LLM(formatted_prompt)
    return response.content if hasattr(response, 'content') else str(response)


def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    documents = text_splitter.split_documents(pages)
    return [doc.page_content.replace('\n', ' ') for doc in documents]


def process_web_links(links):
    loader = UnstructuredURLLoader(urls=links)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    documents = text_splitter.split_documents(documents)
    return [doc.page_content.replace('\n', ' ') for doc in documents]


def process_documents(document_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    return text_splitter.split_documents([Document(page_content=text) for text in document_texts])


def prepare_vectors(documents):
    document_texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(document_texts)
    return document_texts, np.array(vectors)


def create_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def generate_answer(query, combined_texts, index, chat_history):
    standalone_question = rephrase_question_with_history(chat_history, query)
    query_vector = embeddings.embed_query(standalone_question)
    query_vector = np.array(query_vector).reshape((1, -1))
    positions = index.search(query_vector, k=4)[1]
    final_context = ' '.join([combined_texts[pos] for pos in positions[0]])
    
    template = """
    You are an expert researcher. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    If the question is not related to the context or chat history, politely respond that you are tuned to only answer questions that are related to the context.
    
    <context>
      {context}
    </context>
    
    <chat_history>
      {chat_history}
    </chat_history>
    
    Question: {question}
    Helpful answer in markdown:
    """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(context=final_context, chat_history=chat_history, question=standalone_question)
    response = LLM(formatted_prompt)
    return response.content if hasattr(response, 'content') else str(response)
