import time
import requests
import pandas as pd

BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/"
SEARCH_ENDPOINT = f"{BASE_URL}/search"

CHUNKING_METHODS = ["recursive", "fixed"]
EMBEDDING_MODELS = ["MiniLM", "DistilBERT"]

# Test dataset
TEST_FILE = "sample_doc.pdf"
TEST_QUERIES = [
    {
        "query": "What is artificial intelligence?",
        "expected_keywords": ["artificial intelligence", "AI"]
    },
    {
        "query": "Explain deep learning",
        "expected_keywords": ["deep learning", "neural networks"]
    },
    {
        "query": "Machine learning applications",
        "expected_keywords": ["applications", "machine learning"]
    }
]

results = []

def evaluate():
    for chunking in CHUNKING_METHODS:
        for embedding in EMBEDDING_MODELS:
            
            # === 1️⃣ Upload document ===
            print(f"\n🚀 Testing {chunking} chunking + {embedding} embeddings")
            with open(TEST_FILE, "rb") as file:
                response = requests.post(
                    UPLOAD_ENDPOINT,
                    files={"file": file},
                    params={"chunking": chunking, "embedding_model": embedding}
                )
            if response.status_code != 200 and response.status_code != 409:
                print(f"Upload failed: {response.json()}")
                continue

            for test in TEST_QUERIES:
                query = test["query"]
                expected_keywords = test["expected_keywords"]

                # === 2️⃣ Search and measure latency ===
                start_time = time.time()
                search_res = requests.post(
                    SEARCH_ENDPOINT,
                    params={"query": query, "embedding_model": embedding, "chunking": chunking}
                ).json()
                latency = time.time() - start_time

                # === 3️⃣ Calculate precision ===
                retrieved_texts = " ".join([r["text"].lower() for r in search_res["results"]])
                hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_texts)
                precision = hits / len(expected_keywords)

                results.append({
                    "chunking": chunking,
                    "embedding_model": embedding,
                    "query": query,
                    "precision": precision,
                    "latency_sec": latency
                })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("\n✅ Evaluation completed. Results saved to evaluation_results.csv.")

if __name__ == "__main__":
    evaluate()
