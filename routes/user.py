from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from models.user import FileUpload
from models.doc_extract import load_file
from models.chunking_methods import chunking_methods
from models.embedding_methos import generate_embeddings
from models.pinecone_db_test import initialize_pinecone, upsert_to_pinecone
from config.db import metadata_collection
from pymongo.errors import BulkWriteError
from config.constant import MODEL_DIMENSIONS, INDEX_NAME
import os


user = APIRouter()



@user.get("/")
async def home():
    return {'message':'Upload your file in Pdf ot txt'}





@user.post("/")
async def upload_file(
    file: UploadFile = File(...),
    chunking: str = Query("recursive", enum=["recursive", "fixed"]),
    embedding_model: str = Query("MiniLM", enum=["MiniLM","DistilBERT"]),
    # index_name: str = Query("minilm-index-384d", enum=['minilm-index-384d','distilbert-768d'])
):
    """
    Upload file, select chunking & embedding model.
    """
    try: 
        validated = FileUpload(file=file) 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    file_location = f"uploads/{validated.file.filename}"
    with open(file_location, "wb") as f:
        f.write(await validated.file.read())
    
    
    try:
        doc = load_file()
        chunk_func = chunking_methods.get(chunking)
        chunks = chunk_func(doc)
        dimension = MODEL_DIMENSIONS[embedding_model]
        index_name = INDEX_NAME[embedding_model]
        index = initialize_pinecone(index_name=index_name, dimension=dimension)
        upsert_to_pinecone(
            chunks, 
            file_name=validated.file.filename, 
            index=index, 
            mongo_collection=metadata_collection,
            embedding_model=embedding_model,
            chunking=chunking
            
        )
    except BulkWriteError as bwe:
        error_details = bwe.details.get("writeErrors", [{}])[0]
        duplicate_key = error_details.get("keyValue", {}).get("_id", "unknown")
        errmsg = f"Duplicate metadata entry found for ID: {duplicate_key}. File may have been already uploaded."
        raise HTTPException(status_code=409, detail=errmsg)

    finally:
        os.remove(file_location)
    

    return JSONResponse(
        status_code=200, 
        content={"message": f"File processed successfully with {chunking} chunking."}
    )





@user.post("/search")
async def search(query: str,chunking: str = Query("recursive", enum=["recursive", "fixed"]), embedding_model: str = Query("MiniLM", enum=["MiniLM", "DistilBERT"])):
    """
    Search for relevant documents based on user query.
    """
    # Dynamically set Pinecone index dimension
    dimension = MODEL_DIMENSIONS[embedding_model]
    index_name = INDEX_NAME[embedding_model]
    index = initialize_pinecone(index_name=index_name, dimension=dimension)

    # Generate embedding for query (no .tolist())
    query_embedding = generate_embeddings(query, model_name=embedding_model)

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter={"chunking": {"$eq": chunking}}
    )

    # Collect matches
    matches = []
    for match in results["matches"]:
        doc = metadata_collection.find_one({"_id": match["id"]})
        matches.append({
            "score": match["score"],
            "text": (doc["text"] if doc else match["metadata"].get("text", "")),
            "file_name": (doc.get("file_name") if doc else "unknown")
        })

    return {"results": matches}
