from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_vector_db"]
metadata_collection = db["metadata"]
