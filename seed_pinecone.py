import os
from google import genai
from pinecone import Pinecone
from dotenv import load_dotenv

# Load parameters from the local .env configuration file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cx-ai-agent-realestate-leadqualifier")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")

# 1. Initialize API Clients
print("Connecting to secure cloud infrastructure panels...")
ai_client = genai.Client(api_key=GOOGLE_API_KEY)
pc_client = Pinecone(api_key=PINECONE_API_KEY)

# 2. Target the Serverless Index Instance
index = pc_client.Index(PINECONE_INDEX_NAME)

# 3. Define the Luxury Property Data Cluster
MOCK_PROPERTIES = [
    {
        "id": "prop-001",
        "name": "The Address Residences — Downtown Dubai Penthouse",
        "location": "Downtown Dubai",
        "type": "Penthouse",
        "bedrooms": 4,
        "price_aed": 28500000,
        "highlights": "Full Burj Khalifa & fountain view, private rooftop pool, butler service.",
    },
    {
        "id": "prop-002",
        "name": "Palm Jumeirah Signature Villa — Frond D",
        "location": "Palm Jumeirah",
        "type": "Villa",
        "bedrooms": 6,
        "price_aed": 45000000,
        "highlights": "Private beach frontage, 20m infinity pool, direct marina berth.",
    },
    {
        "id": "prop-003",
        "name": "Emaar Beachfront — Sunrise Bay Tower",
        "location": "Dubai Harbour",
        "type": "Apartment",
        "bedrooms": 2,
        "price_aed": 5200000,
        "highlights": "Private beach access, skyline + sea views, resort-style amenities.",
    },
    {
        "id": "prop-004",
        "name": "Bulgari Lighthouse — Jumeirah Bay Island",
        "location": "Jumeirah Bay Island",
        "type": "Penthouse",
        "bedrooms": 5,
        "price_aed": 95000000,
        "highlights": "Ultra-exclusive island address, private marina, Bulgari-branded interiors.",
    }
]

def format_property_text(p: dict) -> str:
    return (
        f"{p['name']} is a {p['bedrooms']}-bedroom {p['type']} located in {p['location']}, Dubai. "
        f"Price: AED {p['price_aed']:,}. Highlights: {p['highlights']}"
    )

# 4. Generate Cloud Vectors & Ingest to Pinecone
print(f"Embedding and uploading properties to index: {PINECONE_INDEX_NAME}...")
upsert_batch = []

for prop in MOCK_PROPERTIES:
    text_content = format_property_text(prop)
    
    # Generate vectors natively — defaults precisely to 768 measurements inside SDK
    embedding_res = ai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text_content,
        config=genai.types.EmbedContentConfig(output_dimensionality=768)
    )
    # Correct array access index to bypass list object tracking exceptions
    vector_values = embedding_res.embeddings[0].values
    
    # Package values into the official Pinecone upload schema
    upsert_batch.append({
        "id": prop["id"],
        "values": vector_values,
        "metadata": {
            "document": text_content,
            "name": prop["name"],
            "location": prop["location"],
            "type": prop["type"],
            "price_aed": prop["price_aed"]
        }
    })

# Commit batch to the serverless cluster
index.upsert(vectors=upsert_batch, namespace=PINECONE_NAMESPACE)
print("✅ Ingestion complete! The real estate vector registry is live.")
