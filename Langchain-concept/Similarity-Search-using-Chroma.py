
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()

collection_name = "my_grocery_collection"

# Define the main function to interact with the Chroma DB
def main():
    try:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing grocery data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")

        texts = [
			'fresh red apples',
			'organic bananas',
			'ripe mangoes',
			'whole wheat bread',
			'farm-fresh eggs',
			'natural yogurt',
			'frozen vegetables',
			'grass-fed beef',
			'free-range chicken',
			'fresh salmon fillet',
			'aromatic coffee beans',
			'pure honey',
			'golden apple',
			'red fruit'
		]
        # Creating a unique id
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]

        collection.add(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )

        all_items = collection.get()

        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        def perform_similarity_search(collection, all_items):
            try:

                query_term = "apple"

                results = collection.query(
                    query_texts=[query_term],
                    n_results=3  # Retrieve top 3 results
                )
                print(f"Query results for '{query_term}':")
                print(results)

                # Check if no results are returned or if the results array is empty
                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    print(f'No documents found similar to "{query_term}"')
                    return
                
                print(f'Top 3 similar documents to "{query_term}":')
                for i in range(min(3, len(results['ids'][0]))):
                    doc_id = results['ids'][0][i]  
                    score = results['distances'][0][i]  
                    text = results['documents'][0][i]
                    if not text:
                        print(f'ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                    else:
                        print(f'ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
            except Exception as error:
                print(f"Error in similarity search: {error}")
        
        perform_similarity_search(collection, all_items)
    except Exception as error:  
        print(f"Error: {error}")

if __name__ == "__main__":
    main()
