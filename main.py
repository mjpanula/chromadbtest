"""
Main CLI Application for ChromaDB Text Storage and Search
Provides a simple command-line interface for adding and searching text paragraphs.
"""

import sys
from chroma_interface import ChromaDBInterface


def print_menu():
    """Display the main menu."""
    print("\n" + "="*60)
    print("ChromaDB Text Paragraph Manager (GPU Accelerated)")
    print("="*60)
    print("1. Add a text paragraph")
    print("2. Add multiple paragraphs (batch)")
    print("3. Search by query")
    print("4. Compare text with existing paragraphs")
    print("5. View all documents")
    print("6. Document count")
    print("7. Delete a document")
    print("8. Reset collection")
    print("9. Exit")
    print("="*60)


def add_paragraph(db: ChromaDBInterface):
    """Add a single text paragraph."""
    print("\n--- Add Text Paragraph ---")
    print("Enter your text paragraph (press Enter twice to finish):")
    
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            lines.pop()  # Remove the last empty line
            break
        lines.append(line)
    
    text = "\n".join(lines).strip()
    
    if not text:
        print("Error: No text entered.")
        return
    
    # Optional metadata
    print("\nAdd metadata? (y/n): ", end="")
    if input().lower() == 'y':
        print("Enter source/title (optional): ", end="")
        source = input().strip()
        metadata = {"source": source} if source else {}
    else:
        metadata = {}
    
    doc_id = db.add_paragraph(text, metadata)
    print(f"\n✓ Paragraph added successfully! ID: {doc_id}")


def add_multiple_paragraphs(db: ChromaDBInterface):
    """Add multiple paragraphs in batch."""
    print("\n--- Add Multiple Paragraphs ---")
    print("How many paragraphs do you want to add? ", end="")
    
    try:
        count = int(input())
    except ValueError:
        print("Error: Invalid number.")
        return
    
    texts = []
    metadatas = []
    
    for i in range(count):
        print(f"\n--- Paragraph {i+1}/{count} ---")
        print("Enter text (press Enter twice to finish):")
        
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                lines.pop()
                break
            lines.append(line)
        
        text = "\n".join(lines).strip()
        if text:
            texts.append(text)
            
            print("Enter source/title (optional): ", end="")
            source = input().strip()
            metadatas.append({"source": source} if source else {})
    
    if texts:
        doc_ids = db.add_paragraphs(texts, metadatas)
        print(f"\n✓ {len(doc_ids)} paragraphs added successfully!")
    else:
        print("No paragraphs were added.")


def search_query(db: ChromaDBInterface):
    """Search for paragraphs using a query."""
    print("\n--- Search by Query ---")
    print("Enter your search query: ", end="")
    query = input().strip()
    
    if not query:
        print("Error: Empty query.")
        return
    
    print("Number of results (default 5): ", end="")
    n_results_input = input().strip()
    n_results = int(n_results_input) if n_results_input else 5
    
    results = db.search(query, n_results)
    
    print(f"\n--- Search Results (Found {len(results['documents'])} matches) ---")
    
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'], 
        results['distances'], 
        results['metadatas']
    ), 1):
        similarity = 1 - distance  # Convert distance to similarity
        print(f"\n{i}. Similarity: {similarity:.4f}")
        if metadata.get('source'):
            print(f"   Source: {metadata['source']}")
        print(f"   Text: {doc[:200]}{'...' if len(doc) > 200 else ''}")


def compare_text(db: ChromaDBInterface):
    """Compare text with existing paragraphs."""
    print("\n--- Compare Text ---")
    print("Enter text to compare (press Enter twice to finish):")
    
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            lines.pop()
            break
        lines.append(line)
    
    text = "\n".join(lines).strip()
    
    if not text:
        print("Error: No text entered.")
        return
    
    print("Number of results (default 5): ", end="")
    n_results_input = input().strip()
    n_results = int(n_results_input) if n_results_input else 5
    
    results = db.compare_text(text, n_results)
    
    print(f"\n--- Similar Paragraphs (Found {len(results['documents'])} matches) ---")
    
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'], 
        results['distances'], 
        results['metadatas']
    ), 1):
        similarity = 1 - distance
        print(f"\n{i}. Similarity: {similarity:.4f}")
        if metadata.get('source'):
            print(f"   Source: {metadata['source']}")
        print(f"   Text: {doc[:200]}{'...' if len(doc) > 200 else ''}")


def view_all_documents(db: ChromaDBInterface):
    """View all documents in the collection."""
    print("\n--- All Documents ---")
    
    results = db.get_all_documents()
    
    if not results['documents']:
        print("No documents in the collection.")
        return
    
    for i, (doc, metadata, doc_id) in enumerate(zip(
        results['documents'],
        results['metadatas'],
        results['ids']
    ), 1):
        print(f"\n{i}. ID: {doc_id}")
        if metadata.get('source'):
            print(f"   Source: {metadata['source']}")
        print(f"   Text: {doc[:200]}{'...' if len(doc) > 200 else ''}")


def delete_document(db: ChromaDBInterface):
    """Delete a document by ID."""
    print("\n--- Delete Document ---")
    print("Enter document ID: ", end="")
    doc_id = input().strip()
    
    if not doc_id:
        print("Error: No ID entered.")
        return
    
    try:
        db.delete_document(doc_id)
        print("✓ Document deleted successfully!")
    except Exception as e:
        print(f"Error deleting document: {e}")


def main():
    """Main application loop."""
    print("Initializing ChromaDB with GPU acceleration...")
    
    try:
        db = ChromaDBInterface()
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        sys.exit(1)
    
    while True:
        print_menu()
        choice = input("\nSelect an option (1-9): ").strip()
        
        try:
            if choice == '1':
                add_paragraph(db)
            elif choice == '2':
                add_multiple_paragraphs(db)
            elif choice == '3':
                search_query(db)
            elif choice == '4':
                compare_text(db)
            elif choice == '5':
                view_all_documents(db)
            elif choice == '6':
                count = db.count_documents()
                print(f"\nTotal documents: {count}")
            elif choice == '7':
                delete_document(db)
            elif choice == '8':
                print("\nAre you sure you want to reset the collection? (yes/no): ", end="")
                if input().lower() == 'yes':
                    db.reset_collection()
                else:
                    print("Reset cancelled.")
            elif choice == '9':
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid option. Please try again.")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
