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
    """
    
    def __init__(self, model_name=None, batch_size=None, device=None):
        """
        Initialize the evaluator
        
        Args:
            model_name: Name of the model to evaluate
            batch_size: Batch size for encoding
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.batch_size = batch_size or MODEL_CONFIG["batch_size"]
        self.device = device or MODEL_CONFIG["device"]
        
        # Initialize model
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model"""
        print(f"Loading model: {self.model_name}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Determine device
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
            
        # Load model
        self.model = SentenceTransformer(self.model_name, device=device)
        self.model.batch_size = self.batch_size
        
        print(f"✓ Model loaded successfully with batch_size={self.batch_size}")
        print(f"✓ Using device: {device}")
        
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
            examples_dataset = load_dataset("xlangai/BRIGHT", config, split=domain)
            
            # Load documents (actual content)
            documents_dataset = load_dataset("xlangai/BRIGHT", "documents", split=domain)
            
            print(f"✓ Loaded {len(examples_dataset)} examples")
            print(f"✓ Loaded {len(documents_dataset)} documents")
            
            return examples_dataset, documents_dataset
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            raise
            
    def evaluate_retrieval(self, examples_dataset, documents_dataset, num_samples=None, 
                          candidate_pool_size=None, top_k=None):
        """
        Evaluate retrieval performance
        
        Args:
            examples_dataset: Dataset with queries and gold documents
            documents_dataset: Dataset with document content
            num_samples: Number of examples to evaluate
            candidate_pool_size: Size of candidate document pool
            top_k: Number of top results to consider
            
        Returns:
            dict: Evaluation results
        """
        # Use default values if not specified
        num_samples = num_samples or DEFAULT_SETTINGS["num_samples"]
        candidate_pool_size = candidate_pool_size or DEFAULT_SETTINGS["candidate_pool_size"]
        top_k = top_k or DEFAULT_SETTINGS["top_k"]
        
        print(f"Starting retrieval evaluation...")
        print(f"  - Number of examples: {num_samples}")
        print(f"  - Candidate pool size: {candidate_pool_size}")
        print(f"  - Top-k: {top_k}")
        
        # Take sample of examples
        examples_sample = examples_dataset.select(range(min(num_samples, len(examples_dataset))))
        
        # Create document lookup
        print("Creating document lookup...")
        doc_lookup = {doc['id']: doc['content'] for doc in documents_dataset}
        print(f"✓ Created lookup for {len(doc_lookup)} documents")
        
        # Evaluation results
        results = []
        
        for i, example in enumerate(tqdm(examples_sample, desc="Evaluating examples")):
            query = example['query']
            gold_ids = example['gold_ids']
            
            # Get candidate documents
            candidate_docs = list(doc_lookup.items())[:candidate_pool_size]
            candidate_ids = [doc_id for doc_id, _ in candidate_docs]
            candidate_contents = [content for _, content in candidate_docs]
            
            # Encode query
            query_embedding = self.model.encode([query], batch_size=1)[0]
            
            # Encode candidate documents
            doc_embeddings = self.model.encode(candidate_contents, batch_size=self.batch_size, 
                                             show_progress_bar=False)
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_scores = similarities[top_indices]
            top_ids = [candidate_ids[idx] for idx in top_indices]
            
            # Check which gold documents are in top-k
            gold_in_top_k = [doc_id for doc_id in gold_ids if doc_id in top_ids]
            
            # Calculate metrics
            recall_at_k = len(gold_in_top_k) / len(gold_ids) if gold_ids else 0
            
            # Store results
            example_result = {
                'example_id': i,
                'query': query,
                'gold_ids': gold_ids,
                'top_ids': top_ids,
                'top_scores': top_scores.tolist(),
                'gold_in_top_k': gold_in_top_k,
                'recall_at_k': recall_at_k,
                'num_gold': len(gold_ids),
                'num_found': len(gold_in_top_k)
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
        
        Args:
            results: List of example results
            overall_metrics: Overall evaluation metrics
            domain: Domain name
            config: Dataset configuration
            output_dir: Output directory
        """
        output_dir = output_dir or DEFAULT_SETTINGS["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"results_{domain}_{config}_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'domain': domain,
                'config': config,
                'timestamp': timestamp,
                'overall_metrics': overall_metrics,
                'detailed_results': results
            }, f, indent=2)
            
        # Save summary
        summary_file = os.path.join(output_dir, f"summary_{domain}_{config}_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("BRIGHT BENCHMARK EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Domain: {domain}\n")
            f.write(f"Config: {config}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in overall_metrics.items():
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