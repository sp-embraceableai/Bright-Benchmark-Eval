import mteb
from sentence_transformers import SentenceTransformer
import os
import torch
import sentence_transformers

def evaluate_on_bright():
    """
    Evaluate the model on the BRIGHT benchmark using MTEB with latest sentence-transformers
    """
    # Model configuration
    model_name = "sp-embraceable/Colbert-Reranker-FT-2000steps"
    
    print(f"Loading model: {model_name}")
    print(f"Using sentence-transformers version: {sentence_transformers.__version__}")
    
    try:
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load the model using SentenceTransformers with latest API
        model = SentenceTransformer(model_name, device=device)
        print("Model loaded successfully!")
        
        # Print model info
        print(f"Model max sequence length: {model.max_seq_length}")
        print(f"Model dimension: {model.get_sentence_embedding_dimension()}")
        
        # Select the BRIGHT tasks (using correct task names)
        print("Getting BRIGHT tasks...")
        bright_tasks = ["BrightRetrieval", "BrightLongRetrieval"]
        tasks = mteb.get_tasks(tasks=bright_tasks)
        print(f"Found {len(tasks)} tasks: {bright_tasks}")
        
        # Create evaluation instance
        evaluation = mteb.MTEB(tasks=tasks)
        
        # Set the output path
        output_path = "./output"
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Starting evaluation on BRIGHT benchmark...")
        print(f"Results will be saved to: {output_path}")
        
        # Run the evaluation with latest API
        results = evaluation.run(model, output_folder=output_path)
        
        print("Evaluation completed successfully!")
        print(f"Results saved to: {output_path}")
        
        # Print summary of results
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for task_name, task_results in results.items():
            print(f"\nTask: {task_name}")
            if hasattr(task_results, 'main_score'):
                print(f"Main Score: {task_results.main_score:.4f}")
            if hasattr(task_results, 'scores'):
                for metric, score in task_results.scores.items():
                    print(f"  {metric}: {score:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_on_bright() 