"""
Example usage of ChromaDB Interface
Demonstrates programmatic usage of the ChromaDB interface.
"""

from chroma_interface import ChromaDBInterface


def main():
    # Initialize the interface (will use GPU if available)
    print("Initializing ChromaDB with GPU acceleration...")
    db = ChromaDBInterface(collection_name="example_collection")
    
    # Example 1: Add a single paragraph
    print("\n--- Example 1: Adding a single paragraph ---")
    text1 = """
    Artificial intelligence is transforming the way we interact with technology.
    Machine learning algorithms can now recognize patterns in data that would be
    impossible for humans to detect manually.
    """
    doc_id1 = db.add_paragraph(text1.strip(), metadata={"source": "AI Article", "category": "technology"})
    print(f"Added document with ID: {doc_id1}")
    
    # Example 2: Add multiple paragraphs in batch
    print("\n--- Example 2: Adding multiple paragraphs ---")
    paragraphs = [
        "Natural language processing enables computers to understand and generate human language. "
        "This technology powers chatbots, translation services, and voice assistants.",
        
        "Computer vision allows machines to interpret and understand visual information from the world. "
        "Applications include facial recognition, autonomous vehicles, and medical image analysis.",
        
        "Deep learning uses neural networks with multiple layers to learn complex patterns. "
        "It has achieved breakthrough results in image recognition, speech processing, and game playing.",
        
        "The Python programming language is widely used for data science and machine learning. "
        "Its simplicity and extensive library ecosystem make it ideal for AI development.",
        
        "Cloud computing provides scalable infrastructure for training large AI models. "
        "Services like AWS, Google Cloud, and Azure offer GPU instances for accelerated computing."
    ]
    
    metadatas = [
        {"source": "NLP Guide", "category": "technology"},
        {"source": "Vision Tutorial", "category": "technology"},
        {"source": "Deep Learning Book", "category": "education"},
        {"source": "Python Guide", "category": "programming"},
        {"source": "Cloud Computing", "category": "infrastructure"}
    ]
    
    doc_ids = db.add_paragraphs(paragraphs, metadatas)
    print(f"Added {len(doc_ids)} documents in batch")
    
    # Example 3: Search with a query
    print("\n--- Example 3: Searching with a query ---")
    query = "How do computers understand images?"
    results = db.search(query, n_results=3)
    
    print(f"\nQuery: '{query}'")
    print(f"Found {len(results['documents'])} results:\n")
    
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'], 
        results['distances'], 
        results['metadatas']
    ), 1):
        similarity = 1 - distance
        print(f"{i}. Similarity: {similarity:.4f}")
        print(f"   Source: {metadata.get('source', 'Unknown')}")
        print(f"   Category: {metadata.get('category', 'Unknown')}")
        print(f"   Text: {doc[:100]}...")
        print()
    
    # Example 4: Compare existing text
    print("\n--- Example 4: Comparing text with existing paragraphs ---")
    new_text = """
    Neural networks are computational models inspired by the human brain.
    They consist of interconnected nodes that process information in layers.
    """
    
    comparison_results = db.compare_text(new_text.strip(), n_results=3)
    
    print(f"Comparing text: '{new_text.strip()[:50]}...'")
    print(f"Most similar paragraphs:\n")
    
    for i, (doc, distance, metadata) in enumerate(zip(
        comparison_results['documents'],
        comparison_results['distances'],
        comparison_results['metadatas']
    ), 1):
        similarity = 1 - distance
        print(f"{i}. Similarity: {similarity:.4f}")
        print(f"   Source: {metadata.get('source', 'Unknown')}")
        print(f"   Text: {doc[:100]}...")
        print()
    
    # Example 5: Get document count
    print("\n--- Example 5: Document statistics ---")
    count = db.count_documents()
    print(f"Total documents in collection: {count}")
    
    # Example 6: View all documents (first 3)
    print("\n--- Example 6: Viewing all documents ---")
    all_docs = db.get_all_documents()
    print(f"Showing first 3 of {len(all_docs['documents'])} documents:\n")
    
    for i in range(min(3, len(all_docs['documents']))):
        print(f"{i+1}. ID: {all_docs['ids'][i]}")
        print(f"   Metadata: {all_docs['metadatas'][i]}")
        print(f"   Text: {all_docs['documents'][i][:80]}...")
        print()
    
    print("\n--- Example complete! ---")
    print(f"GPU acceleration: {'✓ Enabled' if db.device == 'cuda' else '✗ Not available (using CPU)'}")


if __name__ == "__main__":
    main()
