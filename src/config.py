"""
Configuration file for BRIGHT benchmark evaluation
"""

# Model configuration
MODEL_CONFIG = {
    "model_name": "sp-embraceable/Colbert-Reranker-FT-2000steps",
    "batch_size": 4,
    "device": "auto"  # "auto", "cuda", "cpu"
}

# Available BRIGHT domains
AVAILABLE_DOMAINS = [
    "biology",
    "earth_science", 
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions"
]

# Available BRIGHT configurations
AVAILABLE_CONFIGS = [
    "documents",
    "examples", 
    "Gemini-1.0_reason",
    "claude-3-opus_reason",
    "gpt4_reason",
    "grit_reason",
    "llama3-70b_reason",
    "long_documents"
]

# Default evaluation settings
DEFAULT_SETTINGS = {
    "domain": "biology",
    "config": "examples",
    "num_samples": 10,
    "candidate_pool_size": 1000,
    "top_k": 10,
    "output_dir": "evaluation_results"
}

# Evaluation metrics
METRICS = [
    "recall_at_k",
    "precision_at_k", 
    "ndcg_at_k",
    "mrr"
] 