# Importing libraries
import wikipedia        # downloading source documents from wikipedia
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # for chunking - multiple separators
from langchain_huggingface import HuggingFaceEmbeddings        # for implementing the vector embedding model using a local model - no openai or tiktoken connection required
from langchain_chroma import Chroma                            # for storing vector embeddings in the vector databases

# Source Documents Collection
# Defining the wikipedia pages to download - creating a list
nuclear_countries=['Nuclear weapons of the United States', 
                   'Russia and weapons of mass destruction',
                   'Nuclear weapons of China',
                   'Pakistan and weapons of mass destruction',
                   'Nuclear weapons of the United Kingdom'
                   ]

# Defining the folder name where the downloaded text files will be saved
documents_folder = 'docs'

# Creating a loop to ensure all docs are downloaded for every single page title
for topic in nuclear_countries:

    # Error checking by using try block (if page doesn't exist)
    try:
        # Content fetching using the wikipedia library
        page_content = wikipedia.page(topic, auto_suggest=False).content   # .content provides the entire text and avoids messy html and citations that are likely attained from doing copy paste

        # Creating a clean file name for the text file 
        clean_filename = topic.replace('_','').replace('/','').replace(':','') + '.txt'

        # Path Generation - combining the folder name and the clean file name to create the full saving path
        file_path = os.path.join(documents_folder, clean_filename)

        # Writing the clean content in the file - saving the file
        # Opening, writing and ensuring that the file can correctly handle and save all the characters including special letters and symbols
        with open(file_path, 'w', encoding='utf-8') as f:
            # Writes the text content that was stored in the page_content variable into the file that was just opened and assigned to f
            f.write(page_content)

        print(f'Downloaded and saved: {clean_filename}')

    except wikipedia.exceptions.PageError:
        print(f"Error: Wikipedia page not found!! for '{topic}'")
    except Exception as e :
        print(f"An unexpected error occured for '{topic}': {e}")

#1. Loading the source documents from the docs folder
def load_documents(docs_path):
    print(f'Loading documents from the {docs_path} folder')

    # The package DirectoryLoader will scan the docs folder while the TextLoader reads each file
    loader = DirectoryLoader(
        path=docs_path,
        glob='*.txt',            # only look for .txt
        loader_cls=TextLoader,   # loader class   
        recursive=True           # for also searching inside the any subfolders apart from the docs folder
    )

    documents = loader.load()

    print(f'Successfully loaded {len(documents)} documents!')

    # Checking if there are any missing documents 
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add the nuclear country's documents!")

    # Printing the five source documents 
    for i, doc in enumerate (documents[:5]):
        print(f"\nDocument {i+1}:\n")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content Preview: {doc.page_content[:100]}...)")
        print(f"metadata: {doc.metadata}")      # metadata verifies which document the answer came from and allows us to narrow down the search in the documents we upload (filtering) 
    return documents

#2. Chunking Method
def split_documents(documents, chunk_size=800, chunk_overlap=0):  # chunk size kept at 800 as 1 token = 4 characters (800/4 = 200 tokens)
    """Split the documents into smaller chunks with overlap"""
    print("Splitting the source documents into chunks...")

    # Splitting the text in the source documents
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Checking whether the chunking process worked 
    if chunks:
        for i, chunk in enumerate (chunks[:5]):  # loop and slicing to show only the first five chunks
            print(f"\n---Chunk {i+1}---")        # Readable chunks eg. Chunk 1, Chunk 2 ...
            print(f"Source: {chunk.metadata['source']}")  # Which wikipedia file the text came from
            print(f"Length: {len(chunk.page_content)} characters") # Chunk character count
            print(f"Content:")   # 
            print(chunk.page_content) # Prints the actual text stored in the chunk 
            print("-" * 50)   # Line of dashes-helpful for seeing where the chunk overlap starts and ends
        
        # Total number of chunks - tells us the total scale of the data
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

#3. Implementing the Embedding Model 
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    # Initializing the embedding model
    embedding_model = HuggingFaceEmbeddings(                  # token limit - 256 tokens
        model_name="sentence-transformers/all-MiniLM-L6-v2"   # converts text into vectors with 384 dimensions
    )

    # Creating a ChromaDB vector store
    print("-- Creating Vector Store --")
    vectorstore = Chroma.from_documents(
        documents=chunks,                 # langchain documents in the form of chunks
        embedding=embedding_model,        
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"} # 
    )
    print("--- Finished creating the vector store ---")

    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


# Main Function
def main():
    print("Initiating the RAG Pipeline")

    #1. Loading the files
    documents = load_documents(docs_path = documents_folder)

    #2. Chunking
    chunks = split_documents(documents)

    #3. Embedding Model
    vectorstore = create_vector_store(chunks)

    # Testing whether the chunks were actually stored in the vector database
    print(f"Number of chunks in the database: {vectorstore._collection.count()}")

if __name__== "__main__":
    main()





