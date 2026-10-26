from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "data" / "sample.pdf"



#PDF loader
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()


#Splitting the doc into chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(documents=docs)

# Generate embeddings using Gemini
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# creating a vector store using qdrant

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333"
)

print("Vector store created and documents embedded successfully.")

