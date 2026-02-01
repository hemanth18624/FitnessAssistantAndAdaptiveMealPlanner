# =========================
# Imports
# =========================
from dotenv import load_dotenv
import os

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# =========================
# Environment Setup
# =========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "developer-quickstart-py"
TEXT_FIELD = "text"

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")

# =========================
# Helper Functions
# =========================
def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return splitter.split_documents(documents)


def download_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

# =========================
# Pinecone Initialization
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)
print("Index status:", index.describe_index_stats())

# =========================
# Load & Store PDFs
# =========================
print("Loading PDFs...")
docs = load_pdf("pdfs/")

print("Splitting text...")
chunks = text_split(docs)

print(f"Storing {len(chunks)} chunks in Pinecone...")

embeddings = download_embeddings()

PineconeVectorStore.from_texts(
    texts=[c.page_content for c in chunks],
    embedding=embeddings,
    index_name=INDEX_NAME
)

print(" PDF ingestion completed successfully")
