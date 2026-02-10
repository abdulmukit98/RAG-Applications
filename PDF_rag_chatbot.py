import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.prompts import PromptTemplate
import gradio as gr

# configuration
PDF_Folder = "data"
PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "phi3:mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def initialize_rag_system():
    
    """Initialize the RAG system with PDF documents"""

    print("Loading PDF documents...")
    documents = []
    for filename in os.listdir(PDF_Folder):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(PDF_Folder, filename)
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f" Loaded: {filename}")
            except Exception as e:
                print(f" Error loading {filename}: {e}")
    
    if not documents:
        raise ValueError("No PDF file found or loaded in the data folder!")

    # Step 2: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Created {len(chunks)} text chunks")


    # Step 3: Create embeddings and vector store
    print("Creating embeddings and vector database...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create or Load vector store
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print("  Loading existing vector database...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        print("  Creating new vector database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist()


    # Step 4: Initialize the LLM
    print(f"Initializing LLM ({MODEL_NAME})...")
    # llm = Ollama(model=MODEL_NAME)
    llm = OllamaLLM(model=MODEL_NAME)

    # Step 5: Create retrieval chain with custom prompt
    print("Setting up retrival chain...")

    # Custom prompt template
    prompt_template = """Use the following context to answer the question.
    If you don't know the answer based on the context, just say that you don't know.

    Context: {context}

    Question: {question}

    Answer based on context: """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    print("RAG system initialized successfully!")
    return qa_chain

def answer_question(question):
    """Function to process questions and return answers"""
    try:
        # get answer from RAG system
        result = qa_chain.invoke({"query": question})

        answer = result["result"]

        # Get source documents for context
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"][:2]: # show top 2 source
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                sources.append(f"Source: {os.path.basename(source)} (Page {page+1})")

        # Format response
        response = f"{answer}"
        if sources:
            response += f"\n\nSources:\n" + "\n".join(sources)
        
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

print("Hello World")
print("=" * 50)
print("Initializing RAG Chatbot...")
print("=" * 50)

qa_chain = initialize_rag_system()

# Create Gradio interface
print("Launching chatbot interface...")
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question about your PDF:", lines=2),
    outputs=gr.Textbox(label="Answer:", lines=10),
    title="Local RAG Chatbot",
    description="Ask questions about your PDF document. The chatbot will answer based on the content.",
    examples=["What is this document about?", "Summerize the main points", "List the key findings"]
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860, share=False)
