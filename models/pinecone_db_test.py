from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
from .embedding_methos import generate_embeddings



load_dotenv()

def initialize_pinecone(index_name, dimension, metric="cosine"):
    """
    Initializes Pinecone client and index.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    existing_index_names = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_index_names:
        print(f"Index '{index_name}' not found. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Index '{index_name}' already exists.")

    # Connect to index
    index = pc.Index(index_name)
    print(f"✅ Connected to index: {index_name}")
    return index


def upsert_to_pinecone(docs, file_name, index, mongo_collection, embedding_model="MiniLM", chunking='RecursiveCharacterTextSplitter'):
    """
    Upserts embeddings to Pinecone and stores metadata in MongoDB.
    """
    embeddings_list = generate_embeddings(docs, model_name=embedding_model)


    pinecone_vectors = []
    mongo_documents = []

    for i, (doc, embedding) in enumerate(zip(docs, embeddings_list)):
        vector_id = f"{file_name}_{i}"

        pinecone_vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": doc.page_content,
                "chunk_index": i,
                "file_name": file_name,
                "embedding_model": embedding_model,
                "chunking": chunking
            }
        })

        mongo_documents.append({
            "_id": vector_id,
            "text": doc.page_content,
            "chunk_index": i,
            "file_name": file_name,
            "embedding_model": embedding_model,
            "chunking": chunking
        })

    index.upsert(vectors=pinecone_vectors)

    # Upsert with overwrite
    from pymongo import UpdateOne
    operations = [
        UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
        for doc in mongo_documents
    ]
    mongo_collection.bulk_write(operations)

    print(f"✅ Inserted {len(pinecone_vectors)} vectors with {embedding_model} embeddings.")
