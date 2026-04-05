import os
import nest_asyncio
from rag import RAGBot
from test_questions import TEST_SET
from langchain_ollama import OllamaEmbeddings
import pandas as pd
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nest_asyncio.apply()

def semantic_score(question, rag_answer, expected, verbose=False):
    """✅ TRUE SEMANTIC similarity using embeddings"""
    embedder = OllamaEmbeddings(model="nomic-embed-text", temperature=0.0)
    
    # Embed all 3 texts
    q_emb = np.array(embedder.embed_query(question)).reshape(1, -1)
    a_emb = np.array(embedder.embed_query(rag_answer)).reshape(1, -1)  
    e_emb = np.array(embedder.embed_query(expected)).reshape(1, -1)
    
    # Semantic similarities
    q_a_sim = cosine_similarity(q_emb, a_emb)[0][0]
    a_e_sim = cosine_similarity(a_emb, e_emb)[0][0]
    
    score = (q_a_sim * 0.4 + a_e_sim * 0.6)
    
    if verbose:
        return score, q_a_sim, a_e_sim, q_emb, a_emb, e_emb
    return score

def main():
    print("🚀 SEMANTIC RAG TEST (Multiple Questions)")
    
    # 🎯 SELECT HOW MANY QUESTIONS
    N_QUESTIONS = 30  # Change this: 1, 5, 10, or 'all'
    
    ragbot = RAGBot()
    ragbot._init_vectorstore()
    
    if not ragbot.vectorstore:
        print("❌ Run ingest.py first!")
        return
    
    # Handle 'all' case
    if N_QUESTIONS == 'all':
        n_questions = len(TEST_SET)
        test_set = TEST_SET
    else:
        n_questions = min(N_QUESTIONS, len(TEST_SET))
        test_set = TEST_SET[:n_questions]
        # test_set = TEST_SET[120:121]
    
    print(f"\n🤖 Testing {n_questions} questions...")
    results = []
    
    for i, (question, expected) in enumerate(test_set):
        print(f"\n{'='*70}")
        print(f"Q{i+1}: {question}")
        print(f"Expected: {expected}")
        
        start = time.time()
        result = ragbot.query(question)
        latency = time.time() - start
        
        # Get detailed semantic scores
        score, q_a_sim, a_e_sim, _, _, _ = semantic_score(question, result["answer"], expected, verbose=True)
        
        print(f"RAG: {result['answer'][:500]}...")
        print(f"✅ SCORE: {score:.3f} | Latency: {latency:.1f}s")
        print(f"🔍 Q→A: {q_a_sim:.3f} | A→Expected: {a_e_sim:.3f}")
        
        results.append({
            "question": question[:50] + "...",
            "expected": expected,
            "score": round(score, 3),
            "q_a_sim": round(q_a_sim, 3),
            "a_e_sim": round(a_e_sim, 3),
            "latency": round(latency, 1)
        })
    
    # 📊 AVERAGES & SUMMARY
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 SUMMARY TABLE")
    print("="*80)
    print(df[['question', 'score', 'q_a_sim', 'a_e_sim', 'latency']].round(3).to_string(index=False))
    
    print(f"\n🏆 AVERAGES:")
    print(f"   Overall Score:  {df['score'].mean():.3f}")
    print(f"   Q→A Similarity: {df['q_a_sim'].mean():.3f}")
    print(f"   A→Expected:     {df['a_e_sim'].mean():.3f}")
    print(f"   Avg Latency:    {df['latency'].mean():.1f}s")
    
    # Quality assessment
    avg_score = df['score'].mean()
    if avg_score > 0.80:
        print("🎉 EXCELLENT - Production ready!")
    elif avg_score > 0.70:
        print("✅ GOOD - Minor retrieval tuning")
    else:
        print("⚠️ NEEDS WORK - Improve chunking/retrieval")
    
    # df.to_csv("rag_test_last.csv", index=False)
    # print("\n💾 Saved: rag_multi_test.csv")

if __name__ == "__main__":
    main()
