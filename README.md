# ChromaDB Text Paragraph Manager

A Python project for storing and searching text paragraphs using ChromaDB with GPU-accelerated embeddings.

## Features

- **GPU Acceleration**: Uses sentence-transformers with CUDA support for fast embedding generation
- **Simple Interface**: Easy-to-use API for adding and searching text paragraphs
- **Semantic Search**: Find similar paragraphs based on meaning, not just keywords
- **Text Comparison**: Compare new text with existing paragraphs to find similarities
- **Batch Operations**: Efficiently add multiple paragraphs at once
- **Persistent Storage**: Data persists between sessions using ChromaDB
- **CLI Application**: Interactive command-line interface for easy use
- **Metadata Support**: Attach custom metadata to paragraphs

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, will fall back to CPU)
- CUDA Toolkit (for GPU acceleration)

## Installation

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU acceleration, ensure you have:
- NVIDIA GPU drivers installed
- CUDA Toolkit installed (compatible with your PyTorch version)

## Project Structure

```
chromadb/
├── chroma_interface.py   # Core ChromaDB interface class
├── main.py               # Interactive CLI application
├── example.py            # Example usage script
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Usage

### Interactive CLI Application

Run the interactive application:

```bash
python main.py
```

The CLI provides the following options:

1. **Add a text paragraph** - Add a single paragraph with optional metadata
2. **Add multiple paragraphs** - Batch add multiple paragraphs
3. **Search by query** - Search using a natural language query
4. **Compare text** - Find similar paragraphs to your input text
5. **View all documents** - List all stored paragraphs
6. **Document count** - See how many paragraphs are stored
7. **Delete a document** - Remove a paragraph by ID
8. **Reset collection** - Clear all data
9. **Exit** - Close the application

### Programmatic Usage

```python
from chroma_interface import ChromaDBInterface

# Initialize (automatically detects and uses GPU if available)
db = ChromaDBInterface()

# Add a single paragraph
doc_id = db.add_paragraph(
    "Your text paragraph here",
    metadata={"source": "Example", "category": "test"}
)

# Add multiple paragraphs
texts = ["First paragraph", "Second paragraph", "Third paragraph"]
metadatas = [{"source": "batch1"}, {"source": "batch1"}, {"source": "batch1"}]
doc_ids = db.add_paragraphs(texts, metadatas)

# Search with a query
results = db.search("your search query", n_results=5)
for doc, distance in zip(results['documents'], results['distances']):
    similarity = 1 - distance
    print(f"Similarity: {similarity:.4f} - {doc}")

# Compare text with existing paragraphs
results = db.compare_text("text to compare", n_results=5)

# Get document count
count = db.count_documents()

# View all documents
all_docs = db.get_all_documents()

# Delete a document
db.delete_document(doc_id)

# Reset collection (delete all)
db.reset_collection()
```

### Run Example Script

See the example script in action:

```bash
python example.py
```

This demonstrates:
- Adding single and multiple paragraphs
- Searching with queries
- Comparing text
- Viewing stored documents

## GPU Acceleration

The project automatically detects and uses GPU acceleration if available:

- **With GPU**: Uses CUDA for fast embedding generation
- **Without GPU**: Falls back to CPU (still functional, but slower)

To verify GPU usage, check the console output when initializing:
```
Using device: cuda
```

## How It Works

1. **Embeddings**: Text paragraphs are converted to high-dimensional vectors using sentence-transformers
2. **Storage**: Embeddings are stored in ChromaDB with the original text and metadata
3. **Search**: Query text is embedded and compared against stored embeddings using cosine similarity
4. **Results**: Most similar paragraphs are returned with similarity scores

## Configuration

You can customize the following parameters when initializing `ChromaDBInterface`:

- `collection_name`: Name of the ChromaDB collection (default: "text_paragraphs")
- `persist_directory`: Directory for data persistence (default: "./chroma_data")
- `model_name`: Sentence-transformers model (default: "all-MiniLM-L6-v2")

Example:
```python
db = ChromaDBInterface(
    collection_name="my_collection",
    persist_directory="./my_data",
    model_name="all-mpnet-base-v2"  # More accurate but slower
)
```

## Available Models

Some popular sentence-transformer models:

- `all-MiniLM-L6-v2` (default) - Fast and efficient, good for most use cases
- `all-mpnet-base-v2` - More accurate, but slower
- `multi-qa-MiniLM-L6-cos-v1` - Optimized for question-answering
- `paraphrase-multilingual-MiniLM-L12-v2` - Supports 50+ languages

## Troubleshooting

### GPU not detected
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure compatible PyTorch version: `pip install torch --upgrade`
- Check GPU drivers are up to date

### Out of memory errors
- Use a smaller batch size when adding multiple paragraphs
- Switch to a smaller model (e.g., MiniLM instead of mpnet)
- Reduce the number of search results

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Create a virtual environment to avoid conflicts

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to extend this project with additional features such as:
- Document update functionality
- Advanced filtering options
- Export/import capabilities
- Web interface
- Multi-collection management
