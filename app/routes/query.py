from fastapi import APIRouter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

router = APIRouter(
    prefix="/query",
    tags=["query"],
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding = embedding_model,
    collection_name = "sample_collection",
    url = "http://localhost:6333"
)

@router.post("/")
async def query(question : str):
    search_results = vector_db.similarity_search(query=question)
    print(f"Search results: {search_results}")

    context = " ".join([result.page_content for result in search_results])

    system_prompt = f"You are a helpful assistant that provides answers based on the provided context. If the context does not contain the answer, respond with 'I don't know.' {context}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=question
    )
    return {"question": question, "answer": response.text}
