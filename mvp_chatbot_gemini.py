from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import google.generativeai as genai

# 1. Free embeddings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction

# 1. Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create wrapper class for Chroma
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents):
        return embedder.encode(input).tolist()

embedding_function = SentenceTransformerEmbeddingFunction()

# 3. Chroma client with embedding function
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="universities",
    embedding_function=embedding_function
)

# 3. Load CSV
df = pd.read_csv("universitetlar.csv")
for i, row in df.iterrows():
    text = f"""{row['name']} ({row['location']})
    Ranking: {row['ranking']}
    Fakultetlar: {row['faculties']}
    Website: {row['website']}
    Description: {row['description']}"""
    collection.add(documents=[text], ids=[str(row['id'])])

# 4. Gemini setup
genai.configure(api_key="")
model = genai.GenerativeModel("gemini-2.0-flash")

def ask_bot(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    context = "\n".join(results["documents"][0])

    prompt = f"Savol: {query}\n\nMa'lumotlar:\n{context}\n\nJavob ber:"
    response = model.generate_content(prompt)
    return response.text

# 5. Test
print(ask_bot("O'zbekistonda IT bo'yicha eng yaxshi universitet qaysi?"))
