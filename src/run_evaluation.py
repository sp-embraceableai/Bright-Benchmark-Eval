#!/usr/bin/env python3
"""
BRIGHT Benchmark Evaluation Runner
Main script for running evaluations with user-friendly interface
"""

import argparse
import sys
from bright_evaluator import BrightEvaluator
from config import AVAILABLE_DOMAINS, AVAILABLE_CONFIGS, DEFAULT_SETTINGS

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🎯 BRIGHT BENCHMARK EVALUATION")
    print("=" * 60)
    print("Evaluate your model on the BRIGHT benchmark")
    print("Choose domain, configuration, and evaluation parameters")
    print()

def print_available_options():
    """Print available domains and configurations"""
    print("Available domains:")
    for i, domain in enumerate(AVAILABLE_DOMAINS, 1):
        print(f"  {i:2d}. {domain}")
    print()
    
    print("Available configurations:")
    for i, config in enumerate(AVAILABLE_CONFIGS, 1):
        print(f"  {i:2d}. {config}")
    print()

def get_user_choice(options, prompt, default=None):
    """Get user choice from a list of options"""
    while True:
        try:
            choice = input(prompt).strip()
            if not choice and default is not None:
                return default
            
            # Try as number first
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            
            # Try as string
            if choice in options:
                return choice
                
            print(f"❌ Invalid choice. Please select from: {', '.join(options)}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def get_numeric_input(prompt, default=None, min_val=None, max_val=None):
    """Get numeric input from user"""
    while True:
        try:
            choice = input(prompt).strip()
            if not choice and default is not None:
                return default
            
            value = int(choice)
            
            if min_val is not None and value < min_val:
                print(f"❌ Value must be at least {min_val}")
                continue
                
            if max_val is not None and value > max_val:
                print(f"❌ Value must be at most {max_val}")
                continue
                
            return value
            
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def interactive_mode():
    """Run evaluation in interactive mode"""
    print_banner()
    print_available_options()
    
    # Get domain
    domain = get_user_choice(
        AVAILABLE_DOMAINS, 
        f"Select domain (1-{len(AVAILABLE_DOMAINS)}) [default: {DEFAULT_SETTINGS['domain']}]: ",
        DEFAULT_SETTINGS['domain']
    )
    
    # Get configuration
    config = get_user_choice(
        AVAILABLE_CONFIGS,
        f"Select configuration (1-{len(AVAILABLE_CONFIGS)}) [default: {DEFAULT_SETTINGS['config']}]: ",
        DEFAULT_SETTINGS['config']
    )
    
    # Get number of samples
    num_samples = get_numeric_input(
        f"Number of samples to evaluate [default: {DEFAULT_SETTINGS['num_samples']}]: ",
        DEFAULT_SETTINGS['num_samples'],
        min_val=1
    )
    
    # Get candidate pool size
    candidate_pool_size = get_numeric_input(
        f"Candidate pool size [default: {DEFAULT_SETTINGS['candidate_pool_size']}]: ",
        DEFAULT_SETTINGS['candidate_pool_size'],
        min_val=1
    )
    
    # Get top-k
    top_k = get_numeric_input(
        f"Top-k for evaluation [default: {DEFAULT_SETTINGS['top_k']}]: ",
        DEFAULT_SETTINGS['top_k'],
        min_val=1
    )
    
    # Get output directory
    output_dir = input(f"Output directory [default: {DEFAULT_SETTINGS['output_dir']}]: ").strip()
    if not output_dir:
        output_dir = DEFAULT_SETTINGS['output_dir']
    
    # Confirm settings
    print("\n" + "=" * 50)
    print("EVALUATION SETTINGS")
    print("=" * 50)
    print(f"Domain: {domain}")
    print(f"Configuration: {config}")
    print(f"Number of samples: {num_samples}")
    print(f"Candidate pool size: {candidate_pool_size}")
    print(f"Top-k: {top_k}")
    print(f"Output directory: {output_dir}")
    print()
    
    confirm = input("Proceed with evaluation? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Evaluation cancelled")
        return
    
    # Run evaluation
    run_evaluation(domain, config, num_samples, candidate_pool_size, top_k, output_dir)

def run_evaluation(domain, config, num_samples, candidate_pool_size, top_k, output_dir):
    """Run the evaluation with specified parameters"""
    try:
        # Initialize evaluator
        evaluator = BrightEvaluator()
        
        # Run evaluation
        results = evaluator.run_evaluation(
            domain=domain,
            config=config,
            num_samples=num_samples,
            candidate_pool_size=candidate_pool_size,
            top_k=top_k,
            output_dir=output_dir
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="BRIGHT Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py
  
  # Command line mode
  python main.py --domain biology --config examples --samples 10
  
  # Quick test
  python main.py --domain biology --samples 5 --pool-size 500
        """
    )
    
    parser.add_argument(
        "--domain", 
        choices=AVAILABLE_DOMAINS,
        default=DEFAULT_SETTINGS['domain'],
        help="Domain to evaluate"
    )
    
    parser.add_argument(
        "--config",
        choices=AVAILABLE_CONFIGS,
        default=DEFAULT_SETTINGS['config'],
        help="Dataset configuration"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SETTINGS['num_samples'],
        help="Number of samples to evaluate"
    )
    
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_SETTINGS['candidate_pool_size'],
        help="Size of candidate document pool"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_SETTINGS['top_k'],
        help="Top-k for evaluation"
    )
    
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_SETTINGS['output_dir'],
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List available domains and configurations"
    )
    
    args = parser.parse_args()
    
    # List options if requested
    if args.list_options:
        print_banner()
        print_available_options()
        return
    
    # Check if no arguments provided (interactive mode)
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        # Command line mode
        print_banner()
        print("Command line mode selected")
        print(f"Domain: {args.domain}")
        print(f"Configuration: {args.config}")
        print(f"Samples: {args.samples}")
        print(f"Pool size: {args.pool_size}")
        print(f"Top-k: {args.top_k}")
        print(f"Output dir: {args.output_dir}")
        print()
        
        run_evaluation(
            args.domain,
            args.config,
            args.samples,
            args.pool_size,
            args.top_k,
            args.output_dir
        )

if __name__ == "__main__":
    main() 