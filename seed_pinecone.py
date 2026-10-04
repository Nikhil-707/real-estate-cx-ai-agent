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
    },
    {
        "id": "prop-005",
        "name": "Palm Jumeirah Signature Villa — Frond K",
        "location": "Palm Jumeirah",
        "type": "Villa",
        "bedrooms": 5,
        "price_aed": 32000000,
        "highlights": "Open-sea views, private pool, maid's room, direct beach access.",
    },
    {
        "id": "prop-006",
        "name": "One Za'abeel — The Penthouse Collection",
        "location": "Za'abeel",
        "type": "Penthouse",
        "bedrooms": 3,
        "price_aed": 22000000,
        "highlights": "Home to The Link skybridge, panoramic city views, private elevator access.",
    },
    {
        "id": "prop-007",
        "name": "Dubai Hills Estate — Golf Place Villa",
        "location": "Dubai Hills",
        "type": "Villa",
        "bedrooms": 5,
        "price_aed": 12800000,
        "highlights": "Golf course frontage, landscaped garden, community clubhouse and pool.",
    },
    {
        "id": "prop-008",
        "name": "Emirates Hills — Villa Lake View Mansion",
        "location": "Emirates Hills",
        "type": "Villa",
        "bedrooms": 7,
        "price_aed": 68000000,
        "highlights": "Lake and golf-course views, private cinema, 8-car garage, staff quarters.",
    },
    {
        "id": "prop-009",
        "name": "Dorchester Collection — Downtown Residences",
        "location": "Downtown Dubai",
        "type": "Apartment",
        "bedrooms": 3,
        "price_aed": 9800000,
        "highlights": "Branded residence, concierge service, Burj Khalifa district walkability.",
    },
    {
        "id": "prop-010",
        "name": "Bluewaters Island — Penthouse on the Wheel",
        "location": "Bluewaters Island",
        "type": "Penthouse",
        "bedrooms": 4,
        "price_aed": 24500000,
        "highlights": "Ain Dubai & sea views, private rooftop terrace, beach club access.",
    },
    {
        "id": "prop-011",
        "name": "Jumeirah Golf Estates — Fire Villa",
        "location": "Jumeirah Golf Estates",
        "type": "Villa",
        "bedrooms": 6,
        "price_aed": 18500000,
        "highlights": "Championship golf course frontage, private pool, home gym and sauna.",
    },
    {
        "id": "prop-012",
        "name": "DIFC — The Lofts Executive Residence",
        "location": "DIFC",
        "type": "Apartment",
        "bedrooms": 2,
        "price_aed": 4300000,
        "highlights": "Financial-district address, skyline views, walking distance to Gate Avenue.",
    },
    {
        "id": "prop-013",
        "name": "Jumeirah Bay Island — Garden Mansion Plot",
        "location": "Jumeirah Bay Island",
        "type": "Villa",
        "bedrooms": 6,
        "price_aed": 72000000,
        "highlights": "Private beach, yacht berth, triple-height majlis, smart-home integration.",
    },
    {
        "id": "prop-014",
        "name": "Dubai Marina — Marina Gate Sky Collection",
        "location": "Dubai Marina",
        "type": "Apartment",
        "bedrooms": 3,
        "price_aed": 6700000,
        "highlights": "Full marina & sea views, infinity pool deck, direct mall access.",
    },
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