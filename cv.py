import csv
import re
from typing import Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import pickle
from sympy import content

from pdf_parser.parser import extract_from_folder

embedding = OllamaEmbeddings(model="nomic-embed-text")
PERSIST_PATH = "./faiss_db"
LLM_MODEL = "deepseek-r1:1.5b"  # or "mistral", "tinyllama"


def create_or_load_vector_store(documents, persist_path=PERSIST_PATH):
    index_file_path = os.path.join(persist_path, "index.faiss")
    pkl_file_path = os.path.join(persist_path, "index.pkl")

    if os.path.exists(index_file_path) and os.path.exists(pkl_file_path):
        print("[INFO] Loading existing FAISS vector store...")
        store = FAISS.load_local(persist_path, embeddings=embedding, allow_dangerous_deserialization=True)
        return store
    else:
        print("[INFO] Creating new FAISS vector store...")
        store = FAISS.from_documents(documents, embedding)
        store.save_local(persist_path)
        with open(f"{persist_path}.pkl", "wb") as f:
            pickle.dump({"index_name": "index"}, f)
        return store

def add_or_update_document(store: FAISS, new_doc: Document, unique_key: str = "email", persist_path=PERSIST_PATH):
    existing_docs = store.docstore._dict  # dict: doc_id -> Document
    new_key_value = new_doc.metadata.get(unique_key)
    
    to_remove = []
    for doc_id, doc in existing_docs.items():
        if doc.metadata.get(unique_key) == new_key_value:
            to_remove.append(doc_id)
    
    if to_remove:
        print(f"[INFO] Removing {len(to_remove)} document(s) with {unique_key}={new_key_value}")
        # Keep all except those to remove
        filtered_docs = [doc for doc_id, doc in existing_docs.items() if doc_id not in to_remove]
        # Add the new doc
        filtered_docs.append(new_doc)
        print("[INFO] Rebuilding FAISS index with updated documents...")
        store = FAISS.from_documents(filtered_docs, store.embedding_function)
    else:
        print("[INFO] Adding new document to FAISS index...")
        store.add_documents([new_doc])
    
    store.save_local(persist_path)
    print(f"[INFO] FAISS vector store saved at {persist_path}")
    return store


def get_content_from_doc(doc: Dict) -> Document:
    return Document(
        page_content=(
            f"Name: {doc.get('name', 'Unknown')}. "
            f"Email: {doc.get('email', 'Unknown')}. "
            f"Phone: {doc.get('phone', 'Unknown')}. "
            f"Location: {doc.get('location', 'Unknown')}. "
            f"Skills: {doc.get('skills', 'None')}. "
            f"Education: {doc.get('education', 'None')}. "
            f"Work Experience: {doc.get('work_experience', 'None')}."
        ),
        metadata={"email": doc.get("email", "Unknown")},
    ) if doc else None

def main():
    folder_path = "./pdf"  # Replace with your folder path
    all_data = extract_from_folder(folder_path)
    first_doc = get_content_from_doc(all_data[0]) if len(all_data)>0 else None
    if not first_doc:
        print("[ERROR] No valid documents found. Exiting...")
        return
    store = create_or_load_vector_store([first_doc])
    
    for data in all_data:
        doc = get_content_from_doc(data)
        if doc:
            store = add_or_update_document(store, doc, unique_key="email", persist_path=PERSIST_PATH)
    
    print(f"[INFO] Vector store contains {len(store.docstore._dict)} documents after update.")

    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    prompt_template = PromptTemplate.from_template("""
    You are a helpful assistant to provide information based on the following data:
    
    {context}

    Answer the question: {question}
    Don't provide anything outside the above given data.
    """)

    llm = ChatOllama(model=LLM_MODEL, temperature=0.2, max_tokens=512,streaming=True)
    chain = prompt_template | llm | StrOutputParser()

    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        print(f"[INFO] Retrieving relevant documents for query: {query}")
        relevant_docs = retriever.invoke(query)

        print(f"[INFO] Found {len(relevant_docs)} relevant documents.")

        context = "\n".join([doc.page_content for doc in relevant_docs])
        response = chain.invoke({"context": context, "question": query})
        print("AI:", response)


if __name__ == "__main__":
    main()