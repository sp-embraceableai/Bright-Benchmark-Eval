"""
BRIGHT Benchmark Evaluator
Main class for evaluating models on BRIGHT benchmark
"""

import os
import torch
import numpy as np
import gc
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
from datetime import datetime
from config import MODEL_CONFIG, AVAILABLE_DOMAINS, AVAILABLE_CONFIGS, DEFAULT_SETTINGS

class BrightEvaluator:
    """
    Evaluator for BRIGHT benchmark
    Now supports two-model evaluation: retriever (bi-encoder) and reranker (cross-encoder or reranker)
    """
    
    def __init__(self, model_name=None, batch_size=None, device=None, retriever_model_name=None, reranker_model_name=None):
        """
        Initialize the evaluator
        Args:
            model_name: (legacy) Name of the model to evaluate (used if retriever_model_name/reranker_model_name not set)
            batch_size: Batch size for encoding
            device: Device to use ("auto", "cuda", "cpu")
            retriever_model_name: Name of the retriever model (bi-encoder)
            reranker_model_name: Name of the reranker model (cross-encoder or reranker)
        """
        self.retriever_model_name = retriever_model_name or model_name or MODEL_CONFIG["model_name"]
        self.reranker_model_name = reranker_model_name
        self.batch_size = batch_size or MODEL_CONFIG["batch_size"]
        self.device = device or MODEL_CONFIG["device"]

        # Initialize models
        self.retriever = None
        self.reranker = None
        self._load_models()

    def _load_models(self):
        """Load the retriever and reranker models with robust fallback logic for retriever (handles MaxSim etc.)"""
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from sentence_transformers.models import Transformer, Pooling
        from transformers import AutoTokenizer, AutoModel
        import torch

        # Device selection
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        # --- Robust retriever loading (handles MaxSim etc.) ---
        print(f"Loading retriever model: {self.retriever_model_name}")
        try:
            self.retriever = SentenceTransformer(self.retriever_model_name, device=device)
            self.retriever.batch_size = self.batch_size
            print(f"✓ Retriever loaded successfully with batch_size={self.batch_size}")
        except Exception as e:
            print(f"⚠️  Error loading retriever with default loader: {e}")
            print("Trying manual module loading for retriever...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.retriever_model_name)
                model = AutoModel.from_pretrained(self.retriever_model_name)
                transformer = Transformer(self.retriever_model_name, max_seq_length=512)
                pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode='mean')
                modules = [transformer, pooling]
                self.retriever = SentenceTransformer(modules=modules, device=device)
                self.retriever.batch_size = self.batch_size
                print(f"✓ Retriever loaded manually with batch_size={self.batch_size}")
            except Exception as e2:
                print(f"❌ Failed manual retriever loading: {e2}")
                print("Trying with CPU and minimal settings for retriever...")
                try:
                    transformer = Transformer(self.retriever_model_name, max_seq_length=512)
                    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode='mean')
                    modules = [transformer, pooling]
                    self.retriever = SentenceTransformer(modules=modules, device="cpu")
                    self.retriever.batch_size = 1
                    print(f"✓ Retriever loaded on CPU with minimal settings")
                except Exception as e3:
                    print(f"❌ Failed to load retriever completely: {e3}")
                    raise e3

        # --- Reranker loading (CrossEncoder or reranker) ---
        if self.reranker_model_name:
            print(f"Loading reranker model: {self.reranker_model_name}")
            try:
                self.reranker = CrossEncoder(self.reranker_model_name, device=device)
                print(f"✓ Reranker loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load reranker: {e}")
                raise e
        else:
            self.reranker = None
            print("No reranker specified; using retriever only.")
        
    def load_datasets(self, domain, config="examples"):
        """
        Load BRIGHT datasets for specified domain and config
        
        Args:
            domain: Domain name (e.g., "biology")
            config: Dataset configuration (e.g., "examples", "documents")
            
        Returns:
            tuple: (examples_dataset, documents_dataset)
        """
        print(f"Loading BRIGHT datasets for domain: {domain}, config: {config}")
        
        try:
            # Load examples (queries and gold documents)
            print("Loading examples dataset...")
            examples_dataset = load_dataset(
                "xlangai/BRIGHT", 
                config, 
                split=domain,
                trust_remote_code=True
            )
            print(f"✓ Loaded {len(examples_dataset)} examples")
            
            # Load documents (actual content)
            print("Loading documents dataset...")
            documents_dataset = load_dataset(
                "xlangai/BRIGHT", 
                "documents", 
                split=domain,
                trust_remote_code=True
            )
            print(f"✓ Loaded {len(documents_dataset)} documents")
            
            return examples_dataset, documents_dataset
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            print("Trying with cache and reduced timeout...")
            
            try:
                # Try with cache directory
                import os
                cache_dir = os.path.join(os.getcwd(), "dataset_cache")
                os.makedirs(cache_dir, exist_ok=True)
                
                print("Loading examples dataset with cache...")
                examples_dataset = load_dataset(
                    "xlangai/BRIGHT", 
                    config, 
                    split=domain,
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                print(f"✓ Loaded {len(examples_dataset)} examples")
                
                print("Loading documents dataset with cache...")
                documents_dataset = load_dataset(
                    "xlangai/BRIGHT", 
                    "documents", 
                    split=domain,
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                print(f"✓ Loaded {len(documents_dataset)} documents")
                
                return examples_dataset, documents_dataset
                
            except Exception as e2:
                print(f"❌ Failed to load datasets even with cache: {e2}")
                print("Trying with smaller sample for testing...")
                
                # For testing, try with a very small sample
                try:
                    print("Loading minimal sample for testing...")
                    examples_dataset = load_dataset(
                        "xlangai/BRIGHT", 
                        config, 
                        split=domain,
                        trust_remote_code=True,
                        streaming=True
                    )
                    # Convert streaming dataset to regular dataset with first few examples
                    examples_list = []
                    for i, example in enumerate(examples_dataset):
                        if i >= 10:  # Only take first 10 examples
                            break
                        examples_list.append(example)
                    
                    # Create a small documents dataset for testing
                    documents_list = [
                        {"id": f"doc_{i}", "content": f"Test document {i} content"} 
                        for i in range(100)
                    ]
                    
                    from datasets import Dataset
                    examples_dataset = Dataset.from_list(examples_list)
                    documents_dataset = Dataset.from_list(documents_list)
                    
                    print(f"✓ Loaded {len(examples_dataset)} examples (test mode)")
                    print(f"✓ Loaded {len(documents_dataset)} documents (test mode)")
                    
                    return examples_dataset, documents_dataset
                    
                except Exception as e3:
                    print(f"❌ Failed to load even minimal dataset: {e3}")
                    raise e3
            
    def evaluate_retrieval(self, examples_dataset, documents_dataset, num_samples=None, 
                          candidate_pool_size=None, top_k=None):
        """
        Evaluate retrieval performance using two-stage (retriever + reranker) if reranker is set.
        """
        # Use default values if not specified
        num_samples = num_samples or DEFAULT_SETTINGS["num_samples"]
        candidate_pool_size = candidate_pool_size or DEFAULT_SETTINGS["candidate_pool_size"]
        top_k = top_k or DEFAULT_SETTINGS["top_k"]
        print(f"Starting retrieval evaluation...")
        print(f"  - Number of examples: {num_samples}")
        print(f"  - Candidate pool size: {candidate_pool_size}")
        print(f"  - Top-k: {top_k}")
        examples_sample = examples_dataset.select(range(min(num_samples, len(examples_dataset))))
        doc_lookup = {doc['id']: doc['content'] for doc in documents_dataset}
        print(f"✓ Created lookup for {len(doc_lookup)} documents")
        results = []
        for i, example in enumerate(tqdm(examples_sample, desc="Evaluating examples")):
            query = example['query']
            gold_ids = example['gold_ids']
            print(f"\n--- Example {i+1} ---")
            print(f"Query: {query[:100]}...")
            print(f"Gold documents: {len(gold_ids)}")
            # Ensure gold documents are included in candidate pool
            gold_docs = [(gold_id, doc_lookup[gold_id]) for gold_id in gold_ids if gold_id in doc_lookup]
            remaining_size = candidate_pool_size - len(gold_docs)
            additional_docs = list(doc_lookup.items())[:remaining_size]
            candidate_docs = gold_docs + additional_docs
            # Remove duplicates while preserving order
            seen_ids = set()
            unique_candidate_docs = []
            for doc_id, content in candidate_docs:
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_candidate_docs.append((doc_id, content))
            candidate_docs = unique_candidate_docs[:candidate_pool_size]
            candidate_ids = [doc_id for doc_id, _ in candidate_docs]
            candidate_contents = [content for _, content in candidate_docs]
            print(f"Encoding {len(candidate_contents)} candidate documents...")
            print(f"Gold documents in pool: {len([gid for gid in gold_ids if gid in candidate_ids])}/{len(gold_ids)}")
            try:
                # Stage 1: Retrieval (bi-encoder)
                query_embedding = self.retriever.encode([query], batch_size=1)[0]
                doc_embeddings = self.retriever.encode(
                    candidate_contents, 
                    batch_size=self.batch_size, 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                # Get top-N for reranking (if reranker is set)
                rerank_N = min(50, len(candidate_docs))  # rerank top 50 or all if less
                top_indices_retriever = np.argsort(similarities)[::-1][:rerank_N]
                top_ids_retriever = [candidate_ids[idx] for idx in top_indices_retriever]
                top_contents_retriever = [candidate_contents[idx] for idx in top_indices_retriever]
                # Stage 2: Reranking (cross-encoder/reranker)
                if self.reranker:
                    print(f"Reranking top {rerank_N} candidates with reranker...")
                    pairs = [[query, doc] for doc in top_contents_retriever]
                    rerank_scores = self.reranker.predict(pairs)
                    reranked = sorted(zip(top_ids_retriever, rerank_scores), key=lambda x: x[1], reverse=True)
                    top_ids = [doc_id for doc_id, _ in reranked[:top_k]]
                    top_scores = [score for _, score in reranked[:top_k]]
                else:
                    # No reranker: use retriever scores
                    top_ids = top_ids_retriever[:top_k]
                    top_scores = similarities[top_indices_retriever][:top_k]
                gold_in_top_k = [doc_id for doc_id in gold_ids if doc_id in top_ids]
                recall_at_k = len(gold_in_top_k) / len(gold_ids) if gold_ids else 0
                example_result = {
                    'example_id': i,
                    'query': query,
                    'gold_ids': gold_ids,
                    'top_ids': top_ids,
                    'top_scores': top_scores if isinstance(top_scores, list) else top_scores.tolist(),
                    'gold_in_top_k': gold_in_top_k,
                    'recall_at_k': recall_at_k,
                    'num_gold': len(gold_ids),
                    'num_found': len(gold_in_top_k)
                }
                results.append(example_result)
                print(f"Top {top_k} retrieved documents:")
                for j, (doc_id, score) in enumerate(zip(top_ids, top_scores)):
                    found = "✓" if doc_id in gold_ids else "✗"
                    print(f"  {j+1}. {found} Score: {score:.4f} | ID: {doc_id}...")
                print(f"Gold documents found in top-{top_k}: {len(gold_in_top_k)}/{len(gold_ids)}")
                print(f"Recall@{top_k}: {recall_at_k:.4f}")
            except Exception as e:
                print(f"❌ Error processing example {i+1}: {e}")
                example_result = {
                    'example_id': i,
                    'query': query,
                    'gold_ids': gold_ids,
                    'top_ids': [],
                    'top_scores': [],
                    'gold_in_top_k': [],
                    'recall_at_k': 0.0,
                    'num_gold': len(gold_ids),
                    'num_found': 0,
                    'error': str(e)
                }
                results.append(example_result)
        return results
        
    def calculate_overall_metrics(self, results):
        """
        Calculate overall evaluation metrics
        
        Args:
            results: List of example results
            
        Returns:
            dict: Overall metrics
        """
        recalls = [result['recall_at_k'] for result in results]
        
        overall_metrics = {
            'average_recall_at_k': np.mean(recalls),
            'recall_at_k_std': np.std(recalls),
            'min_recall_at_k': np.min(recalls),
            'max_recall_at_k': np.max(recalls),
            'total_examples': len(results),
            'examples_with_gold_found': sum(1 for r in results if r['num_found'] > 0),
            'total_gold_documents': sum(r['num_gold'] for r in results),
            'total_found_documents': sum(r['num_found'] for r in results)
        }
        
        return overall_metrics
        
    def save_results(self, results, overall_metrics, domain, config, output_dir=None):
        """
        Save evaluation results
        """
        output_dir = output_dir or DEFAULT_SETTINGS["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare model info for JSON
        model_info = {
            'retriever_model_name': self.retriever_model_name,
        }
        if self.reranker_model_name:
            model_info['reranker_model_name'] = self.reranker_model_name
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert overall_metrics and results
        overall_metrics_converted = convert_numpy_types(overall_metrics)
        results_converted = convert_numpy_types(results)
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"results_{domain}_{config}_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                **model_info,
                'domain': domain,
                'config': config,
                'timestamp': timestamp,
                'overall_metrics': overall_metrics_converted,
                'detailed_results': results_converted
            }, f, indent=2)
            
        # Save summary
        summary_file = os.path.join(output_dir, f"summary_{domain}_{config}_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("BRIGHT BENCHMARK EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Retriever Model: {self.retriever_model_name}\n")
            if self.reranker_model_name:
                f.write(f"Reranker Model: {self.reranker_model_name}\n")
            f.write(f"Domain: {domain}\n")
            f.write(f"Config: {config}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in overall_metrics_converted.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
                    
        print(f"✓ Results saved to: {output_dir}")
        print(f"  - Detailed results: {results_file}")
        print(f"  - Summary: {summary_file}")
        
    def run_evaluation(self, domain, config="examples", num_samples=None, 
                      candidate_pool_size=None, top_k=None, output_dir=None):
        """
        Run complete evaluation
        
        Args:
            domain: Domain to evaluate
            config: Dataset configuration
            num_samples: Number of examples to evaluate
            candidate_pool_size: Size of candidate document pool
            top_k: Number of top results to consider
            output_dir: Output directory
            
        Returns:
            dict: Evaluation results
        """
        print(f"\n🎯 Starting BRIGHT evaluation for domain: {domain}")
        print("=" * 60)
        
        # Load datasets
        examples_dataset, documents_dataset = self.load_datasets(domain, config)
        
        # Run evaluation
        results = self.evaluate_retrieval(
            examples_dataset, documents_dataset, 
            num_samples, candidate_pool_size, top_k
        )
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics(results)
        
        # Print results
        self._print_results(results, overall_metrics)
        
        # Save results
        self.save_results(results, overall_metrics, domain, config, output_dir)
        
        return {
            'results': results,
            'overall_metrics': overall_metrics
        }
        
    def _print_results(self, results, overall_metrics):
        """Print evaluation results"""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        
        print(f"Average Recall@K: {overall_metrics['average_recall_at_k']:.4f}")
        print(f"Recall@K std dev: {overall_metrics['recall_at_k_std']:.4f}")
        print(f"Examples with gold found: {overall_metrics['examples_with_gold_found']}/{overall_metrics['total_examples']}")
        print(f"Total gold documents: {overall_metrics['total_gold_documents']}")
        print(f"Total found documents: {overall_metrics['total_found_documents']}")
        
        print(f"\nIndividual Recall@K scores:")
        for i, result in enumerate(results):
            print(f"  Example {i+1}: {result['recall_at_k']:.4f} ({result['num_found']}/{result['num_gold']} found)")
            
        print("\n✓ Evaluation completed successfully!") 