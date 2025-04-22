from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
import time

# Example corpus (your documents)
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog is brown",
    "The fox is brown",
    "The quick fox jumped over the fence",
    "The dog chased the fox"
]

# Initialize BM25 and fit the corpus
bm25 = BM25Encoder()
bm25.fit(corpus)

# Encode documents into sparse vectors (6-dimensional)
doc_sparse_vectors = bm25.encode_documents(corpus)
print(f"Document Sparse Vectors: {doc_sparse_vectors}")

pc = Pinecone(api_key="pcsk_2MPq4R_GBiTYzgeDKyYWX79L3LLv9Eiu2YCdXvCeun7MDb6UPiaV7vU1dMLRAyJzabyWHC")

# Index name
index_name = 'pdf-chunks-index'

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=6,  # Dimension is set to 6 since we're using BM25 with 6-dimensional sparse vectors
        metric='cosine',  # Similarity metric
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1',
            capacity_mode='Serverless'
        )
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists.")

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

# Upsert documents into Pinecone
upsert_data = [
    (f"doc_{i}", {'indices': doc_vector['indices'], 'values': doc_vector['values']})
    for i, doc_vector in enumerate(doc_sparse_vectors)
]

# Ensure upsert_data is in the correct format (list of tuples)
upsert_data = [(f"doc_{i}", {'indices': doc_vector['indices'], 'values': doc_vector['values']})
               for i, doc_vector in enumerate(doc_sparse_vectors)]

# Attempt to upsert the data into Pinecone
try:
    index.upsert(vectors=upsert_data)
    print(f"Upserted {len(upsert_data)} documents into Pinecone.")
except Exception as e:
    print(f"Error during upsert: {e}")

# Now, you can encode a query
query = "What is the brown fox doing?"

# Encode the query as a sparse vector
query_sparse_vector = bm25.encode_queries([query])[0]
print(f"Query Sparse Vector: {query_sparse_vector}")

# Perform search: Query Pinecone with the sparse vector
try:
    results = index.query(
        queries=[{'indices': query_sparse_vector['indices'], 'values': query_sparse_vector['values']}],  # Sparse vector format
        top_k=3  # top_k=3 to get top 3 most similar documents
    )
    print("Search Results:")
    for match in results['matches']:
        print(f"Document ID: {match['id']}, Score: {match['score']}")
except Exception as e:
    print(f"Error during search: {e}")
