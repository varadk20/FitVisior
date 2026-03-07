import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

class RAGBot:
    def __init__(self):
        self.model_name = "mistral:7b-instruct"
        self.embedding_model = "nomic-embed-text:latest"
        self.persist_dir = "./chroma_db"
        
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.llm = OllamaLLM(model=self.model_name)
        self.vectorstore = None
        self._init_vectorstore()
    
    def _init_vectorstore(self):
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
            chunk_size=800,      # Smaller for precision
            chunk_overlap=100    # Reduced overlap
        )
        
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(filepath)
                raw_docs = loader.load_and_split(text_splitter)
                # ✅ Accurate page metadata
                for doc in raw_docs:
                    page_num = doc.metadata.get('page', 1)
                    doc.metadata['source'] = filename
                    doc.metadata['page'] = page_num
            elif filename.endswith('.txt'):
                loader = TextLoader(filepath, encoding='utf-8')
                raw_docs = loader.load_and_split(text_splitter)
                for doc in raw_docs:
                    doc.metadata['source'] = filename
                    doc.metadata['page'] = 'N/A'
            else:
                continue
            
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
                for root, dirs, files in os.walk(self.persist_dir, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                        except:
                            pass
                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except:
                            pass
                try:
                    os.rmdir(self.persist_dir)
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
        
        # ✅ UNIQUE sources only (no duplicates)
        seen_sources = set()
        unique_sources = []
        for doc in relevant_docs:
            source_key = f"{doc.metadata.get('source')}-{doc.metadata.get('page')}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(doc)
        
        sources = [
            f"📄 {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
            for doc in unique_sources[:3]  # Top 3 unique sources
        ]
        
        return {
            "answer": response,
            "sources": sources
        }
