# BRIGHT Benchmark Evaluation

A comprehensive evaluation framework for the BRIGHT benchmark using MTEB and sentence-transformers.

## Quick Start

### Interactive Mode (Recommended)
```bash
python main.py
```

### Quick Test Mode
```bash
# Quick test with 5 samples
python main.py quick biology 5

# Quick test with custom samples
python main.py quick economics 3
```

### Command Line Mode
```bash
# Full evaluation
python main.py --domain biology --config examples --samples 10

# List available options
python main.py --list
```

## Project Structure

```
Bright-benchmark/
├── main.py                    # Main entry point
├── src/                       # Source code package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── bright_evaluator.py   # Main evaluator class
│   ├── run_evaluation.py     # Interactive/CLI evaluation runner
│   └── quick_test.py         # Quick test script
├── requirements.txt           # Python dependencies
├── evaluation_results/        # Output directory for results
└── README.md                 # This file
```

## Features

- **Single Entry Point**: Use `main.py` for all operations
- **Interactive Mode**: User-friendly interface to choose evaluation parameters
- **Quick Test Mode**: Fast evaluation with minimal samples
- **Command Line Interface**: Scriptable evaluation with command line arguments
- **Multiple Domains**: Support for all 12 BRIGHT domains
- **Multiple Configurations**: Support for different dataset configurations
- **Customizable Parameters**: Choose number of samples, pool size, top-k, etc.
- **Comprehensive Results**: Detailed metrics and saved results
- **Memory Management**: Optimized for GPU memory usage

## Available Domains

1. biology
2. earth_science
3. economics
4. psychology
5. robotics
6. stackoverflow
7. sustainable_living
8. pony
9. leetcode
10. aops
11. theoremqa_theorems
12. theoremqa_questions

## Available Configurations

1. documents
2. examples
3. Gemini-1.0_reason
4. claude-3-opus_reason
5. gpt4_reason
6. grit_reason
7. llama3-70b_reason
8. long_documents

## Configuration

Edit `src/config.py` to customize:

- Model settings (name, batch size, device)
- Default evaluation parameters
- Available domains and configurations

## Evaluation Metrics

- **Recall@K**: Percentage of gold documents found in top-K results
- **Precision@K**: Precision of retrieved documents
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

## Usage Examples

### Interactive Mode
```bash
python main.py
```
Follow the prompts to select:
- Domain (e.g., biology, economics)
- Configuration (e.g., examples, documents)
- Number of samples (e.g., 10, 50, 100)
- Candidate pool size (e.g., 1000, 5000)
- Top-k for evaluation (e.g., 10, 20)

### Quick Test Mode
```bash
# Test biology domain with 5 samples
python main.py quick biology 5

# Test economics domain with 3 samples
python main.py quick economics 3

# Test psychology domain with 10 samples
python main.py quick psychology 10
```

### Command Line Mode
```bash
# Full evaluation
python main.py --domain biology --config examples --samples 20 --pool-size 2000 --top-k 15

# Minimal evaluation
python main.py --domain psychology --samples 5 --pool-size 500

# List available options
python main.py --list

# Show help
python main.py --help
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- sentence-transformers==4.1.0
- mteb>=1.38.0
- torch>=2.0.0
- datasets
- scikit-learn
- tqdm

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in `src/config.py`
- Reduce `candidate_pool_size`
- Use CPU instead of GPU: set `device="cpu"`

### Model Loading Issues
- Check model name in `src/config.py`
- Ensure model is available on Hugging Face Hub
- Verify sentence-transformers version compatibility

### Dataset Loading Issues
- Check internet connection
- Verify domain and configuration names
- Try different configurations if one fails

## Results

Results are saved in the `evaluation_results/` directory:

- `results_*.json`: Detailed results with all metrics
- `summary_*.txt`: Human-readable summary
- Timestamped files for multiple runs

## Development

### Project Structure
- `main.py`: Single entry point for all operations
- `src/`: Source code package
  - `config.py`: Configuration management
  - `bright_evaluator.py`: Core evaluation logic
  - `run_evaluation.py`: Interactive and CLI interfaces
  - `quick_test.py`: Quick testing functionality

### Adding New Features
1. Add new functionality to appropriate module in `src/`
2. Update `src/__init__.py` if adding new exports
3. Update `main.py` if adding new command line options
4. Update this README with new usage examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- BRIGHT benchmark: https://huggingface.co/datasets/xlangai/BRIGHT
- MTEB framework: https://github.com/embeddings-benchmark/mteb
- Sentence Transformers: https://github.com/UKPLab/sentence-transformers
