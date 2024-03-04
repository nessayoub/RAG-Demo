import json
import torch
from sentence_transformers import SentenceTransformer
import faiss

# Load pre-trained RAG model
retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
generator = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")
sequence_generator = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-base")

# Load restaurant menu data from JSON
with open('menu.json', 'r') as f:
    menu_data = json.load(f)

# Load Sentence Transformers model
embedding_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Generate embeddings for menu items
menu_embeddings = []
menu_item_names = []
for item in menu_data:
    menu_item_names.append(item['name'])
    menu_embeddings.append(embedding_model.encode(item['name']))

menu_embeddings = torch.tensor(menu_embeddings)

# Build FAISS index for fast approximate nearest neighbor search
index = faiss.IndexFlatIP(menu_embeddings.size(1))
index.add(menu_embeddings.numpy())

def retrieve_entities(query, k=5):
    # Utilize pre-trained retriever model for efficient entity retrieval
    query_embedding = torch.tensor(embedding_model.encode(query)).unsqueeze(0)
    _, idx = index.search(query_embedding.numpy(), k)
    retrieved_entities = [menu_data[i] for i in idx[0]]
    return retrieved_entities

def generate_response(query, retrieved_entities):
    # Use RAG for response generation
    options = []
    for entity in retrieved_entities:
        option = entity['name']
        if 'sizes' in entity:
            option += f" ({', '.join(entity['sizes'])})"
        if 'calories' in entity:
            option += f", {entity['calories']} calories"
        options.append(option)

    if len(options) > 0:
        prompt = f"What are the options for {query}? {' '.join(options)}"
    else:
        prompt = f"Sorry, no options found for {query}."

    inputs = generator(prompt, return_tensors="pt")
    response = sequence_generator.generate(**inputs, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    while True:
        user_query = input("What would you like to order or ask (or type 'quit' to exit)? ")
        if user_query.lower() == 'quit':
            break

        retrieved_entities = retrieve_entities(user_query)
        response = generate_response(user_query, retrieved_entities)
        print(response)

if __name__ == "__main__":
    main()
