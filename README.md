# Palm Mind Interview Task

## Document Retrieval System – FastAPI + Pinecone + MongoDB
This project provides a backend system for uploading PDF/TXT files, chunking text, generating vector embeddings, and storing them in a vector database (Pinecone) with metadata in MongoDB. It also includes an evaluation module to compare different chunking strategies and embedding models based on retrieval accuracy and latency.

## Features
- 📂 File upload API with Pydantic validation  
- ✂️ Multiple chunking strategies: Recursive, Fixed-size  
- 🔑 Multiple embedding models: MiniLM, DistilBERT  
- 🛢️ Vector storage using **Pinecone**  
- 🗄️ Metadata storage using **MongoDB**  
- 🔍 Semantic search with top-k retrieval  
- 📊 Evaluation script with precision and latency analysis  


---

## 📦 Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```


2️⃣ Create and activate a virtual environment
```
python -m venv .venv
source .venv/bin/activate 
```

3️⃣ Copy environment template and configure
```
cp .env_template .env
```

4️⃣ Install dependencies
```
pip install -r requirements.txt
```

▶️ Run the application
```
uvicorn index:app --reload
```
