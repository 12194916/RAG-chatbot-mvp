import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ======================
# 1. API KEY
# ======================
OPENAI_API_KEY = ""

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ======================
# 2. CSV o‘qish
# ======================
df = pd.read_csv("universitetlar.csv")

# ======================
# 3. Chroma client va embedding model
# ======================
chroma_client = chromadb.Client()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

collection = chroma_client.create_collection(
    name="universities",
    embedding_function=openai_ef
)

# ======================
# 4. CSV’dagi ma’lumotlarni yuklash
# ======================
for i, row in df.iterrows():
    text = f"""{row['name']} ({row['location']})
    Ranking: {row['ranking']}
    Fakultetlar: {row['faculties']}
    Sayt: {row['website']}
    Tavsif: {row['description']}"""
    collection.add(documents=[text], ids=[str(row['id'])])

print("✅ Ma'lumotlar ChromaDB'ga yuklandi!")

# ======================
# 5. Chatbot funktsiyasi
# ======================
def ask_bot(query, n_results=3):
    # Vector DB’dan eng mos ma’lumotlarni olish
    results = collection.query(query_texts=[query], n_results=n_results)
    context = "\n".join(results["documents"][0])

    # LLM prompt yaratish
    prompt = f"Savol: {query}\n\nMa'lumotlar:\n{context}\n\nJavob ber:"

    answer = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}]
    )

    return answer.choices[0].message.content

# ======================
# 6. Sinov
# ======================
if __name__ == "__main__":
    while True:
        query = input("\n❓ Savol ber (yoki 'exit' yoz): ")
        if query.lower() == "exit":
            break
        answer = ask_bot(query)
        print("\n🤖 Javob:", answer)
