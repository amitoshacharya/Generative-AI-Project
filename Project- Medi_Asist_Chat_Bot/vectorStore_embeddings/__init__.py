import sys
# Add the path of the project to the sys.path
project_path = r'C:\\Users\\amitosh.acharya\\Desktop\\Self Projects and Learnings\\1. Text Chatbot\\Project 1\\Code'
sys.path.append(project_path)


"""This module create the embeddings from documents using the genAI models."""
import os
from genAI_models import embeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time


def load_documents(dir_path: str):
    """Load the documents from directory path.
    Args:
        dir_path: str: The directory path where the documents are stored.

    Returns:
        documents: List[Document]: The list of documents loaded from the directory path.
    """
    loader = PyPDFDirectoryLoader(dir_path)
    documents = loader.load()

    return documents

def transform_documents(documents, chunking_kwargs=None, chunking_strategy="semantic"):
    """Transform the documents into smaller chunks.
    Args:
        documents: List[Document]: The list of documents to be transformed.
        chunking_strategy: str: The strategy to be used for text splitting. Options: "semantic" or "recursive".
        chunking_kwargs: dict: The arguments for the text splitter.

    Returns:
        docs: List[Document]: The list of documents after transformation.
    """
    print("Transforming documents into smaller chunks...")
    start = time.time()
    if chunking_strategy == "semantic":
        if chunking_kwargs is None:
            chunking_kwargs = {"embeddings": embeddings}
        text_splitter = SemanticChunker(embeddings= chunking_kwargs["embeddings"],)
        docs = text_splitter.split_documents(documents)
    elif chunking_strategy == "recursive":
        if chunking_kwargs is None:
            chunking_kwargs = {"chunk_size": 4000, "chunk_overlap": 200}
        text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunking_kwargs["chunk_size"],
                                                       chunk_overlap= chunking_kwargs["chunk_overlap"]
                                                       )
        docs = text_splitter.split_documents(documents)
        

    # docs = text_splitter.create_documents([d.page_content for d in documents])
    print(f"Time taken to transform the {len(documents)} documents: {(time.time() - start)/60} minutes")
    return docs

def embed_and_store(documents, store_embeddings_path):
    """Embed the documents and store the embeddings in Vector DB.
    Args:
        docs: List[Document]: The list of documents to be embedded and stored.
    """

    print("Creating embeddings and storing in FAISS index...")
    start = time.time()
    try:
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(store_embeddings_path)

        print(f"Time taken to embed and store documents: {(time.time() - start)/60} minutes")

        message = f"Successfully created embeddings for {len(documents)} and stored in FAISS index \"{store_embeddings_path}\"."


    except Exception as e:
        message = f"Error in creating embeddings and storing in FAISS index: {e}"

    return message


def create_vectorDB(documents_path, vectorDB_path, chunking_strategy, chunking_kwargs=None):
    """Create an external Vector DB to perform RAG operation.
    Args:
        documents_path: str: The path where the documents are stored.
        vectorDB_path: str: The path where the Vector DB is to be stored.
    """
    # document loader: load the documents with Document Loader
    documents = load_documents(documents_path)
    print(f"Length of Loaded documents:{len(documents)}")

    # Text Splitting: transform the documents into smaller chunks
    docs = transform_documents(documents = documents, chunking_strategy = chunking_strategy, chunking_kwargs= chunking_kwargs)
    print(f"Length of documents after splitting:{len(docs)}")

    # Embed and Store: create vector store and save the embeddings
    message = embed_and_store(docs, vectorDB_path)
    return message


def get_retriever(vectorDB_path, search_type="similarity", search_kwargs={"k": 1}):
    """Use VectorDB as a retriever.
    Args:
        vectorDB_path: str: The path of the Vector DB.
        search_type: str: The type of search operation to be performed.
        search_kwargs: dict: The search arguments for the search operation.
    """
    vectorStore = FAISS.load_local(vectorDB_path, embeddings)
    retriever = vectorStore.as_retriever(search_type = search_type, 
                                         search_kwargs = search_kwargs)
    return retriever


if __name__ == "__main__":

    embed_documents_flag = input("\nDo you want to create VectorDB for the documents?\nHere, \n1. Enter '0' for No \n2. Enter '1' for Yes \nYour Input: ")
    local_dir_path = r"C:\Users\amitosh.acharya\Desktop\Self Projects and Learnings\1. Text Chatbot\Project 1\Code\documents"
        
    vectorDB_path = r"C:\Users\amitosh.acharya\Desktop\Self Projects and Learnings\1. Text Chatbot\Project 1\Code\VectorDB-(FAISS)"


    if int(embed_documents_flag):
        ## Create an external Vector DB to feed into the LLM
        # Step-1: Load the documents
        # Step-2: Transform the documents into smaller chunks
        chunking_strategy = "recursive" ## "semantic" or "recursive"
        chunking_kwargs = {"chunk_size": 4000, "chunk_overlap": 200}
        # Step-3: Embed the documents 
        # Step-4: Store the embeddings in Vector DB
        message = create_vectorDB(local_dir_path, vectorDB_path, 
                                chunking_strategy = chunking_strategy, 
                                chunking_kwargs=chunking_kwargs)
        print(message)
    
    else:
        print("Embedding of documents is disabled.")
        
    # Step-5: Retrieval of the relevant documents.
    search_type = "similarity" ## "similarity" or "mmr"
    search_kwargs = {"k":1, "score_threshold": 0.95}
    retriever = get_retriever(vectorDB_path = vectorDB_path, 
                              search_type = search_type, 
                              search_kwargs = search_kwargs)

    query = "I have a neurologic disorder and want to know treatment for it, as per doctor its Arteriovenous Malformations?"
    relevant_docs = retriever.invoke(query)

    print(f"Relevant Documents for the user query: '{query}' retrieved are {len(relevant_docs)} ", end="\n"*2)
    for doc in relevant_docs:
        print(doc.page_content)
        print("\n\n")