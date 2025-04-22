from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus (documents)
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog is brown",
    "The fox is brown"
]

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

# Function to encode documents into sparse vector format
def encode_document(doc):
    tfidf_vector = vectorizer.transform([doc])  # Transform the document
    indices = tfidf_vector.nonzero()[1]  # Get non-zero feature indices
    values = tfidf_vector.data  # Get corresponding TF-IDF values
    return {"indices": indices.tolist(), "values": values.tolist()}

# Encode a new document as a sparse vector
new_doc = "The brown fox is lazy"
doc_sparse_vector = encode_document(new_doc)

# Print the encoded sparse vector
print(doc_sparse_vector)

# # """
# from pinecone_text.sparse import BM25Encoder

# # Define corpus
# corpus = [
#     "The quick brown fox jumps over the lazy dog",
#     "The lazy dog is brown",
#     "The fox is brown",
#     "Seeta is my friend",
# ]

# # Initialize BM25 and fit the corpus
# bm25 = BM25Encoder()
# bm25.fit(corpus)

# # Encode a new document into a sparse vector
# doc_sparse_vector = bm25.encode_documents(["seeta is my friend"])

# print("BM25 Sparse Vector:", doc_sparse_vector)

# # Encode a query as a sparse vector for search
# query_sparse_vector = bm25.encode_queries(["who is seeta"])

# print("BM25 Query Sparse Vector:", query_sparse_vector)

# #"""