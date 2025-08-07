#!/usr/bin/env python3
"""
Quick Test Script for BRIGHT Benchmark
Fast evaluation with minimal samples for testing
"""

from bright_evaluator import BrightEvaluator
from config import AVAILABLE_DOMAINS

def quick_test(domain="biology", num_samples=5, retriever_model=None, reranker_model=None):
    """
    Run a quick test evaluation
    Args:
        domain: Domain to test (default: biology)
        num_samples: Number of samples (default: 5)
        retriever_model: Retriever model name (optional)
        reranker_model: Reranker model name (optional)
    """
    print("🚀 QUICK TEST MODE")
    print("=" * 40)
    print(f"Domain: {domain}")
    print(f"Samples: {num_samples}")
    print(f"Pool size: 50 (reduced for speed)")
    print(f"Top-k: 10")
    if retriever_model:
        print(f"Retriever model: {retriever_model}")
    if reranker_model:
        print(f"Reranker model: {reranker_model}")
    print()
    try:
        # Initialize evaluator
        evaluator = BrightEvaluator(retriever_model_name=retriever_model, reranker_model_name=reranker_model)
        # Run quick evaluation
        results = evaluator.run_evaluation(
            domain=domain,
            config="examples",
            num_samples=num_samples,
            candidate_pool_size=50,  # Much smaller pool for speed
            top_k=10,
            output_dir="quick_test_results"
        )
        return results
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for quick test"""
    import sys
    # Get domain from command line or use default
    domain = sys.argv[1] if len(sys.argv) > 1 else "biology"
    if domain not in AVAILABLE_DOMAINS:
        print(f"❌ Invalid domain: {domain}")
        print(f"Available domains: {', '.join(AVAILABLE_DOMAINS)}")
        return
    # Get number of samples from command line or use default
    try:
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    except ValueError:
        print("❌ Invalid number of samples")
        return
    # Get retriever and reranker models from command line if provided
    retriever_model = sys.argv[3] if len(sys.argv) > 3 else None
    reranker_model = sys.argv[4] if len(sys.argv) > 4 else None
    # Run quick test
    quick_test(domain, num_samples, retriever_model, reranker_model)

if __name__ == "__main__":
    main() 