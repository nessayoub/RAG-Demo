import json
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import RagRetriever, RagTokenForGeneration, RagSequenceForGeneration

def load_models():
    try:
        retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
        generator = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")
        sequence_generator = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-base")
        return retriever, generator, sequence_generator
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def load_menu_data(file_path):
    try:
        with open(file_path, 'r') as f:
            menu_data = json.load(f)
        return menu_data
    except Exception as e:
        print(f"Error loading menu data: {e}")
        return []

def build_index(menu_embeddings):
    index = faiss.IndexFlatIP(menu_embeddings.size(1))
    index.add(menu_embeddings.numpy())
    return index

def retrieve_entities(query, embedding_model, index, menu_data, k=5):
    try:
        query_embedding = torch.tensor(embedding_model.encode(query)).unsqueeze(0)
        _, idx = index.search(query_embedding.numpy(), k)
        retrieved_entities = [menu_data[i] for i in idx[0]]
        return retrieved_entities
    except Exception as e:
        print(f"Error retrieving entities: {e}")
        return []

def generate_response(query, retrieved_entities, generator):
    try:
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
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

def main():
    retriever, generator, sequence_generator = load_models()
    if not all((retriever, generator, sequence_generator)):
        return

    menu_data = load_menu_data('menu.json')
    if not menu_data:
        return

    embedding_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    menu_embeddings = torch.tensor([embedding_model.encode(item['name']) for item in menu_data])
    index = build_index(menu_embeddings)

    while True:
        user_query = input("What would you like to order or ask (or type 'quit' to exit)? ")
        if user_query.lower() == 'quit':
            break

        retrieved_entities = retrieve_entities(user_query, embedding_model, index, menu_data)
        response = generate_response(user_query, retrieved_entities, generator)
        print(response)

if __name__ == "__main__":
    main()
