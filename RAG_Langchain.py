# Import necessary libraries for document processing, embeddings, and LLM integration
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI  # NEW

import os
import requests
from typing import List
from langchain.schema import Document

# Download the PDF file from a given URL and save it locally
pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
response = requests.get(pdf_url)
pdf_path = 'attention_is_all_you_need.pdf'

with open(pdf_path, 'wb') as file:
    file.write(response.content)

# Function to extract text from the downloaded PDF file
def pdf_extract(pdf_path: str) -> List[Document]:
    print("PDF file text is extracted...")
    loader = PyPDFLoader(pdf_path)
    pdf_text = loader.load()
    return pdf_text

# Extract text from the PDF file
pdf_text = pdf_extract(pdf_path)

# Function to split the extracted text into smaller chunks for processing
def pdf_chunk(pdf_text: List[Document]) -> List[Document]:
    print("PDF file text is chunked....")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pdf_text)
    return chunks

# Chunk the extracted PDF text into smaller segments
chunks = pdf_chunk(pdf_text)

# Set the directory path for storing the Chroma vector database
current_dir = os.path.join(os.getcwd(), "rag")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_pdf")
os.makedirs(persistent_directory, exist_ok=True)

# Function to create a Chroma vector store from the text chunks
def create_vector_store(chunks: List[Document], db_path: str) -> Chroma:
    print("Chroma vector store is created...\n")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=db_path)
    return db

# Create a Chroma vector store using the extracted chunks
db = create_vector_store(chunks, persistent_directory)

# Function to retrieve relevant chunks from the vector store based on a query
def retrieve_context(db: Chroma, query: str) -> List[Document]:
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    print("Relevant chunks are retrieved...\n")
    relevant_chunks = retriever.invoke(query)
    return relevant_chunks

# Function to build a context string from the retrieved chunks
def build_context(relevant_chunks: List[Document]) -> str:
    print("Context is built from relevant chunks")
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    return context

# Function to handle the entire process of extracting context from the PDF and vector store
def get_context(inputs: dict) -> dict:
    pdf_path, query, db_path = inputs['pdf_path'], inputs['query'], inputs['db_path']

    if not os.path.exists(db_path):
        print("Creating a new vector store...\n")
        pdf_text = pdf_extract(pdf_path)
        chunks = pdf_chunk(pdf_text)
        db = create_vector_store(chunks, db_path)
    else:
        print("Loading the existing vector store\n")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    relevant_chunks = retrieve_context(db, query)
    context = build_context(relevant_chunks)
    return {'context': context, 'query': query}

# Define the prompt template for the RAG chain
template = """
You are a helpful AI assistant. Answer the user's question using only the provided context.
If the answer is not in the context, say: "The answer to this question is not available in the provided content."

Context:
{context}

Question:
{query}

Answer:
"""

rag_prompt = ChatPromptTemplate.from_template(template)

# Set up the OpenRouter LLM with the specified model and API key
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_api_key="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",  # Replace with your actual key
    model="mistralai/mixtral-8x7b",  # Or any model supported by OpenRouter
    temperature=0.5
)

str_parser = StrOutputParser()

# Define the RAG chain by combining context retrieval, prompt, LLM, and output parsing
rag_chain = (
    RunnableLambda(get_context)
    | rag_prompt
    | llm
    | str_parser
)

# Run the RAG chain with a sample query and print the generated answer
query = 'Explain transformer model in three line'
answer = rag_chain.invoke({'pdf_path': pdf_path, 'query': query, 'db_path': persistent_directory})

print(f"Query: {query}\n")
print(f"Generated answer: {answer}")
