import mteb
from sentence_transformers import SentenceTransformer
import torch
import sentence_transformers

def check_mteb_tasks():
    """
    Check what tasks are available in MTEB and show version information
    """
    print("Checking available tasks in MTEB...")
    print("="*50)
    
    # Show version information
    print(f"MTEB version: {mteb.__version__}")
    print(f"Sentence-transformers version: {sentence_transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        # Get all available tasks
        all_tasks = mteb.get_tasks()
        task_names = [task.__class__.__name__ for task in all_tasks]
        
        print(f"Total number of available tasks: {len(task_names)}")
        print("\nAll available tasks:")
        print("-" * 30)
        
        # Print all task names in a numbered list
        for i, task_name in enumerate(sorted(task_names), 1):
            print(f"{i:3d}. {task_name}")
        
        # Check for BRIGHT specifically
        print("\n" + "="*50)
        print("SEARCHING FOR BRIGHT TASK")
        print("="*50)
        
        if "BRIGHT" in task_names:
            print("✓ BRIGHT task found!")
        else:
            print("✗ BRIGHT task NOT found")
            
            # Look for similar tasks
            print("\nLooking for similar tasks...")
            similar_tasks = []
            for task_name in task_names:
                if any(keyword in task_name.upper() for keyword in ['BRIGHT', 'RERANK', 'RANKING', 'RETRIEVAL', 'SEARCH']):
                    similar_tasks.append(task_name)
            
            if similar_tasks:
                print(f"Found {len(similar_tasks)} potentially relevant tasks:")
                for task in similar_tasks:
                    print(f"  - {task}")
            else:
                print("No similar tasks found")
        
        # Check for benchmarks
        print("\n" + "="*50)
        print("CHECKING AVAILABLE BENCHMARKS")
        print("="*50)
        
        try:
            benchmarks = mteb.get_benchmarks()
            print(f"Available benchmarks: {list(benchmarks.keys())}")
        except Exception as e:
            print(f"Error getting benchmarks: {e}")
        
        return task_names
        
    except Exception as e:
        print(f"Error checking tasks: {e}")
        return []

if __name__ == "__main__":
    check_mteb_tasks() 