import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

class RAGBot:
    def __init__(self):
        self.model_name = "mistral:7b-instruct"
        self.embedding_model = "nomic-embed-text:latest"
        self.persist_dir = "./chroma_db"
        
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.llm = OllamaLLM(model=self.model_name, temperature=0.0)
        self.vectorstore = None
        
    def _init_vectorstore(self):
        """ONLY when DB exists"""
        if not os.path.exists(self.persist_dir) or not os.path.exists(f"{self.persist_dir}/chroma.sqlite3"):
            return
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        except:
            self.vectorstore = None
    
    def load_documents(self, folder_path="documents"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return []
        
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        for filename in pdf_files:  # Only process PDFs
            print(f"  📄 Processing {filename}...")
            filepath = os.path.join(folder_path, filename)
            
            try:
                loader = PyPDFLoader(filepath)
                raw_docs = loader.load_and_split(text_splitter)
                
                # Add filename metadata (no page numbers)
                for doc in raw_docs:
                    doc.metadata['source'] = filename
                
                docs.extend(raw_docs)
                print(f"    ✅ {filename} done ({len(raw_docs)} chunks)")
                
            except Exception as e:
                print(f"    ❌ {filename} failed: {e}")
                continue
        
        # Handle TXT files silently (no progress print)
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                loader = TextLoader(filepath, encoding='utf-8')
                raw_docs = loader.load_and_split(text_splitter)
                for doc in raw_docs:
                    doc.metadata['source'] = filename
                docs.extend(raw_docs)
        
        return docs
    
    
    def ingest_documents(self, folder_path="documents"):
        docs = self.load_documents(folder_path)
        if not docs:
            return "No documents found in folder"
        
        self.vectorstore = None
        
        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir, ignore_errors=True)
            except:
                pass
        
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.vectorstore.persist()
        return f"Ingested {len(docs)} chunks from {len(os.listdir(folder_path))} files"
    
    def query(self, question, k=4):
        if not self.vectorstore or self.vectorstore._collection.count() == 0:
            return {"answer": "Please ingest documents first.", "sources": []}
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(question)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""Use ONLY the following context to answer. If not in context, say "Not found in documents".

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.invoke(prompt)
        
        # ✅ NO PAGE NUMBERS - just unique filenames
        seen_sources = set()
        unique_sources = []
        for doc in relevant_docs:
            source_name = doc.metadata.get('source', 'Unknown')
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                unique_sources.append(doc)
        
        sources = [
            f"📄 {doc.metadata.get('source', 'Unknown')}"
            for doc in unique_sources[:3]
        ]
        
        return {
            "answer": response,
            "sources": sources
        }
