#!/usr/bin/env python3
"""
BRIGHT Benchmark Evaluation - Main Entry Point
A comprehensive evaluation framework for the BRIGHT benchmark
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from run_evaluation import main as run_evaluation_main
from quick_test import quick_test
from config import AVAILABLE_DOMAINS

def main():
    """Main entry point for BRIGHT benchmark evaluation"""
    
    # Check if any arguments provided
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        print("🎯 BRIGHT Benchmark Evaluation")
        print("=" * 50)
        print("No arguments provided. Starting interactive mode...")
        print()
        run_evaluation_main()
        return
    
    # Check for quick test command
    if sys.argv[1] == "quick":
        if len(sys.argv) >= 3:
            domain = sys.argv[2]
            num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            retriever_model = sys.argv[4] if len(sys.argv) > 4 else None
            reranker_model = sys.argv[5] if len(sys.argv) > 5 else None
        else:
            domain = "biology"
            num_samples = 5
            retriever_model = None
            reranker_model = None
            
        if domain not in AVAILABLE_DOMAINS:
            print(f"❌ Invalid domain: {domain}")
            print(f"Available domains: {', '.join(AVAILABLE_DOMAINS)}")
            return
            
        print(f"🚀 Quick test: {domain} domain with {num_samples} samples")
        quick_test(domain, num_samples, retriever_model, reranker_model)
        return
    
    # Check for help command
    if sys.argv[1] in ["-h", "--help", "help"]:
        print_help()
        return
    
    # Check for list command
    if sys.argv[1] in ["-l", "--list", "list"]:
        print_available_options()
        return
    
    # Otherwise, pass arguments to run_evaluation
    run_evaluation_main()

def print_help():
    """Print help information"""
    print("🎯 BRIGHT Benchmark Evaluation")
    print("=" * 50)
    print()
    print("USAGE:")
    print("  python main.py                    # Interactive mode")
    print("  python main.py quick [domain] [samples] [retriever_model] [reranker_model]  # Quick test with two models")
    print("  python main.py --domain biology --samples 10  # Command line mode")
    print("  python main.py --list             # List available options")
    print("  python main.py --help             # Show this help")
    print()
    print("EXAMPLES:")
    print("  # Interactive mode (recommended)")
    print("  python main.py")
    print()
    print("  # Quick test with 5 samples")
    print("  python main.py quick biology 5")
    print()
    print("  # Quick test with two models")
    print("  python main.py quick biology 5 'retriever-model-name' 'reranker-model-name'")
    print()
    print("  # Full evaluation")
    print("  python main.py --domain biology --config examples --samples 20")
    print()
    print("  # List available options")
    print("  python main.py --list")
    print()
    print("For detailed command line options, run:")
    print("  python main.py --help")

def print_available_options():
    """Print available domains and configurations"""
    from config import AVAILABLE_CONFIGS
    
    print("Available domains:")
    for i, domain in enumerate(AVAILABLE_DOMAINS, 1):
        print(f"  {i:2d}. {domain}")
    print()
    
    print("Available configurations:")
    for i, config in enumerate(AVAILABLE_CONFIGS, 1):
        print(f"  {i:2d}. {config}")
    print()

if __name__ == "__main__":
    main() 