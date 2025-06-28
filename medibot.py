from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECCONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

#extracting data from the pdf file
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",  # Correct keyword
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

#extracting the data from the file
extracted_data=load_pdf_file(data='Data/')


# split the data into text chunks\
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return(text_chunks)
text_chunks= text_split(extracted_data)


# downloading the hugging face embeddingd
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return(embeddings)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings=download_hugging_face_embeddings()

# importing pinecone vector store
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'medicalchatbot-1'


pc.create_index(
    name=index_name,
    dimension=384,  
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

#embedding each vectoe chunk and uploading it into pinecone vector storage
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings

)

#loading existing index from pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

#storing all the docs in an object called retriever
retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k":3} )

# initializing the open ai model
llm = ChatOpenAI(
    temperature=0.4,
    max_tokens = 500,
    model_name="gpt-3.5-turbo",
    openai_api_key="OPENAI_API_KEY"
)

#generating system prompt and human input using chains
system_prompt = (
    " you are an assistant for question-answering tasks."
    "use the folowing pieces of retrieved context to answer the question"
    "if you dont know the answer, say that you dont know. "
    "use three sentence maximum and kepp the answer concise"
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]

)

#setting up a rag pipeline using langchain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain )

# storing the output of ragchain into an object responce
responce = rag_chain.invoke({"input": "what is asthama"})
print(responce["answer"])


