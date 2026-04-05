#!/usr/bin/env python3
"""
Manual document ingestion script - run once, then use Streamlit app
Usage: python ingest.py
"""
import os
import shutil
from rag import RAGBot

def main():
    print("🔄 Starting document ingestion from 'documents/' folder...")
    
    # Create documents folder if it doesn't exist
    os.makedirs("documents", exist_ok=True)
    
    ragbot = RAGBot()
    
    # Process each PDF individually with progress
    folder_path = "documents"
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in 'documents/' folder")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF(s):")
    for pdf in pdf_files:
        print(f"  📄 {pdf}")
    
    print("\n🔄 Processing...")
    
    result = ragbot.ingest_documents()
    
    print("\n✅", result)
    print("📁 Vectorstore saved to ./chroma_db")
    print("\n🚀 Now you can run: streamlit run app.py")
    print("💡 To update documents: delete ./chroma_db and run this script again")

if __name__ == "__main__":
    main()
