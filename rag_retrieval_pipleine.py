# Importing libraries
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# RETRIEVAL PHASE

# Directory where the vector embeddings are stored
persist_directory = "db/chroma_db"

# Loading embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Recreating the vector store
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}  # cosine similarity algorithm - smallest angles represent higher similarity
)

# Searching for relevant documents
query = "Tell me about the first nuclear test in the United States of America."

# Retriever component
retriever = db.as_retriever(search_kwargs={"k": 3}) # will provide the top 3 most relevant chunks 

relevant_documents = retriever.invoke(query) # the list of objects the Chromadb found

print(f"User query: {query}")

# Displaying the results - output of the retrieval process
print("Context")
for i, doc in enumerate(relevant_documents, 1):
    print(f"\nDocument {i}:\n\n{doc.page_content}\n")

# More synethetic questions

#1. In which year did Pakistan test its first nuclear weapon?
#2. In world ranking, where does the United Kingdom fall for testing its nuclear weapons?
#3. Tell me about the first nuclear test in the United States of America.
#4. Where are the research and production facilities of the nuclear weapons located in the United Kingdom?
#5. Tell me about the 1952 test.
#6. In which year did Islamabad (Pakistan) conduct its maiden atomic detonation?
#7. Besides the US and Russia, which country was the first to test a nuclear bomb?
#8. In which year did China test its first nuclear weapon?


# GENERATION PHASE (Answer Generation by the LLM model)

# Identifying and fetching relevant document content for a given query
combined_input = f"""Based on the following documents, please answer this question {query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_documents])}

Please provide a clear and factual answer using the information from all the documents. Analyze the dates in each and every document and create a thorough timeline before answering. Before answering the query, please carefully go through all the documents and answer the query asked properly. If you cannot find the answer in these documents, say "I do not have enough information based on the documents provided"
"""

# Creating a Llama3 LLM model - free LLM local model
model = ChatOllama(model="llama3",
                   temperature=0,    # To get the exact same answer everytime I run the script
                   seed=25           # To ensure reproducibility
                   )

# Defining messages for the model
messages= [
    SystemMessage(content="You are a helpful assistant!"),
    HumanMessage(content=combined_input),
]

# Invoking the model with the combined input
result=model.invoke(messages)

# Displaying the final result and content
print("\nRESPONSE GENERATED!!!\n")
print("\nFull Result:\n")
print(result)
print('\n---Content Only---\n')
print(result.content)