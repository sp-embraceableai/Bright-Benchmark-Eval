#!/usr/bin/env python3
"""
Diagnostic script to investigate model performance issues
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bright_evaluator import BrightEvaluator
from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

def diagnose_model():
    """Diagnose the model performance issues"""
    print("🔍 MODEL DIAGNOSIS")
    print("=" * 50)
    
    # 1. Test model loading
    print("\n1. Testing model loading...")
    try:
        evaluator = BrightEvaluator()
        print("✓ Model loaded successfully")
        
        # Check model properties
        print(f"   - Model name: {evaluator.model_name}")
        print(f"   - Batch size: {evaluator.batch_size}")
        print(f"   - Device: {evaluator.model.device}")
        
        # Test basic encoding
        test_texts = ["This is a test query", "This is another test"]
        embeddings = evaluator.model.encode(test_texts, batch_size=1)
        print(f"   - Embedding shape: {embeddings.shape}")
        print(f"   - Embedding sample: {embeddings[0][:5]}...")
        
        # Check for embedding collapse
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"   - Similarity between test texts: {similarity:.4f}")
        
        if similarity > 0.99:
            print("   ⚠️  WARNING: Very high similarity - possible embedding collapse!")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # 2. Test dataset loading
    print("\n2. Testing dataset loading...")
    try:
        examples_dataset = load_dataset("xlangai/BRIGHT", "examples", split="biology", trust_remote_code=True)
        documents_dataset = load_dataset("xlangai/BRIGHT", "documents", split="biology", trust_remote_code=True)
        
        print(f"✓ Loaded {len(examples_dataset)} examples")
        print(f"✓ Loaded {len(documents_dataset)} documents")
        
        # Examine first example
        first_example = examples_dataset[0]
        print(f"   - First example query: {first_example['query'][:100]}...")
        print(f"   - Gold document IDs: {first_example['gold_ids']}")
        print(f"   - Number of gold documents: {len(first_example['gold_ids'])}")
        
        # Check if gold documents exist in documents dataset
        doc_ids = [doc['id'] for doc in documents_dataset]
        gold_ids = first_example['gold_ids']
        found_gold = [gid for gid in gold_ids if gid in doc_ids]
        print(f"   - Gold documents found in corpus: {len(found_gold)}/{len(gold_ids)}")
        
        if len(found_gold) == 0:
            print("   ❌ CRITICAL: No gold documents found in corpus!")
        else:
            print("   ✓ Gold documents found in corpus")
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # 3. Test single example evaluation
    print("\n3. Testing single example evaluation...")
    try:
        # Take first example
        example = examples_dataset[0]
        query = example['query']
        gold_ids = example['gold_ids']
        
        print(f"   - Query: {query[:100]}...")
        print(f"   - Gold IDs: {gold_ids}")
        
        # Get a small sample of documents
        sample_docs = documents_dataset.select(range(min(100, len(documents_dataset))))
        doc_lookup = {doc['id']: doc['content'] for doc in sample_docs}
        
        # Check if any gold documents are in our sample
        sample_gold = [gid for gid in gold_ids if gid in doc_lookup]
        print(f"   - Gold documents in sample: {len(sample_gold)}/{len(gold_ids)}")
        
        if len(sample_gold) == 0:
            print("   ⚠️  No gold documents in sample, using first few documents")
            candidate_docs = list(doc_lookup.items())[:50]
        else:
            print("   ✓ Gold documents found in sample")
            candidate_docs = list(doc_lookup.items())[:50]
        
        # Encode query and documents
        print("   - Encoding query and documents...")
        query_embedding = evaluator.model.encode([query], batch_size=1)[0]
        
        candidate_contents = [content for _, content in candidate_docs]
        doc_embeddings = evaluator.model.encode(candidate_contents, batch_size=1)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:10]
        top_scores = similarities[top_indices]
        top_ids = [candidate_docs[idx][0] for idx in top_indices]
        
        print("   - Top 10 results:")
        for i, (doc_id, score) in enumerate(zip(top_ids, top_scores)):
            is_gold = "✓" if doc_id in gold_ids else "✗"
            print(f"     {i+1}. {is_gold} Score: {score:.4f} | ID: {doc_id}")
        
        # Check gold documents
        gold_in_top = [doc_id for doc_id in gold_ids if doc_id in top_ids]
        recall = len(gold_in_top) / len(gold_ids) if gold_ids else 0
        print(f"   - Gold documents in top-10: {len(gold_in_top)}/{len(gold_ids)}")
        print(f"   - Recall@10: {recall:.4f}")
        
        # Check similarity distribution
        print(f"   - Similarity scores - Min: {similarities.min():.4f}, Max: {similarities.max():.4f}, Mean: {similarities.mean():.4f}")
        
        if similarities.max() - similarities.min() < 0.01:
            print("   ⚠️  WARNING: Very small similarity range - possible embedding collapse!")
        
    except Exception as e:
        print(f"❌ Single example evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Test with different model
    print("\n4. Testing with fallback model...")
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use a known working model
        fallback_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        
        # Test the same example
        query_embedding_fb = fallback_model.encode([query], batch_size=1)[0]
        doc_embeddings_fb = fallback_model.encode(candidate_contents, batch_size=1)
        
        similarities_fb = cosine_similarity([query_embedding_fb], doc_embeddings_fb)[0]
        top_indices_fb = np.argsort(similarities_fb)[::-1][:10]
        top_scores_fb = similarities_fb[top_indices_fb]
        top_ids_fb = [candidate_docs[idx][0] for idx in top_indices_fb]
        
        print("   - Fallback model top 10 results:")
        for i, (doc_id, score) in enumerate(zip(top_ids_fb, top_scores_fb)):
            is_gold = "✓" if doc_id in gold_ids else "✗"
            print(f"     {i+1}. {is_gold} Score: {score:.4f} | ID: {doc_id}")
        
        gold_in_top_fb = [doc_id for doc_id in gold_ids if doc_id in top_ids_fb]
        recall_fb = len(gold_in_top_fb) / len(gold_ids) if gold_ids else 0
        print(f"   - Fallback model Recall@10: {recall_fb:.4f}")
        
        if recall_fb > recall:
            print("   ✓ Fallback model performs better - original model has issues")
        else:
            print("   ⚠️  Both models perform poorly - might be dataset issue")
            
    except Exception as e:
        print(f"❌ Fallback model test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    diagnose_model() 