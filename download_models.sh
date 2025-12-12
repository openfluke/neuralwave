#!/bin/bash

# Script to download ~100 small language models (under ~1.5B parameters)
# Using huggingface-cli
#
# Usage: ./download_models.sh [max_models]
# Example: ./download_models.sh 10 (Downloads only the first 10)

MAX_ABSOLUTE_LIMIT=${1:-150} # Default to trying all, or user can set limit

echo "🚀 Starting download of small open-source models..."
echo "📂 Target directory: ~/.cache/huggingface/hub"

counter=0

# Function to download a model
# Returns 0 on success, 1 on failure
download_model() {
    if [ $counter -ge "$MAX_ABSOLUTE_LIMIT" ]; then
        return 0
    fi

    MODEL_ID=$1
    ((counter++))
    
    echo ""
    echo "[$counter] ⬇️  Downloading $MODEL_ID..."
    
    # Exclude heavy formats to save time/space (prefer safetensors)
    huggingface-cli download "$MODEL_ID" --exclude "*.pth" "*.bin" "*.msgpack" "*.h5" "*.ot"
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Success: $MODEL_ID"
    else
        echo "   ❌ Failed: $MODEL_ID"
    fi
}

# --- Qwen Family (Apache 2.0 / Tongyi Qianwen) ---
download_model "Qwen/Qwen2.5-0.5B"
download_model "Qwen/Qwen2.5-1.5B"
download_model "Qwen/Qwen2.5-Coder-0.5B"
download_model "Qwen/Qwen2.5-Coder-1.5B"
download_model "Qwen/Qwen2.5-Math-1.5B"
download_model "Qwen/Qwen2-0.5B"
download_model "Qwen/Qwen2-1.5B"
download_model "Qwen/Qwen1.5-0.5B"
download_model "Qwen/Qwen1.5-1.8B"
download_model "Qwen/Qwen1.5-MoE-A2.7B" # Maybe too big? Excluding to keep strict <2B
download_model "Qwen/Qwen-1.8B"

# --- SmolLM Family (Apache 2.0) ---
download_model "HuggingFaceTB/SmolLM2-135M"
download_model "HuggingFaceTB/SmolLM2-360M"
download_model "HuggingFaceTB/SmolLM2-1.7B"
download_model "HuggingFaceTB/SmolLM-135M"
download_model "HuggingFaceTB/SmolLM-360M"
download_model "HuggingFaceTB/SmolLM-1.7B"

# --- TinyLlama & Variants (Apache 2.0) ---
download_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
download_model "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
download_model "TinyLlama/TinyLlama-1.1B-Python-v0.1"
download_model "TinyLlama/TinyLlama-1.1B-step-50K-105b"

# --- Pythia Family (Apache 2.0) ---
# Deduped and V0 versions
download_model "EleutherAI/pythia-14m"
download_model "EleutherAI/pythia-31m"
download_model "EleutherAI/pythia-70m"
download_model "EleutherAI/pythia-160m"
download_model "EleutherAI/pythia-410m"
download_model "EleutherAI/pythia-1b"
download_model "EleutherAI/pythia-1.4b"
download_model "EleutherAI/pythia-14m-deduped"
download_model "EleutherAI/pythia-31m-deduped"
download_model "EleutherAI/pythia-70m-deduped"
download_model "EleutherAI/pythia-160m-deduped"
download_model "EleutherAI/pythia-410m-deduped"
download_model "EleutherAI/pythia-1b-deduped"

# --- Cerebras Family (Apache 2.0) ---
download_model "cerebras/Cerebras-GPT-111M"
download_model "cerebras/Cerebras-GPT-256M"
download_model "cerebras/Cerebras-GPT-590M"
download_model "cerebras/Cerebras-GPT-1.3B"

# --- OPT (Open Pre-trained Transformer) Family (MIT) ---
download_model "facebook/opt-125m"
download_model "facebook/opt-350m"
download_model "facebook/opt-1.3b"

# --- Bloom Family (BigScience OpenRAIL-M) ---
download_model "bigscience/bloom-560m"
download_model "bigscience/bloom-1b1"
download_model "bigscience/bloom-1b7"
download_model "bigscience/bloomz-560m"
download_model "bigscience/bloomz-1b1"
download_model "bigscience/bloomz-1b7"

# --- GPT-Neo & NeoX (MIT/Apache 2.0) ---
download_model "EleutherAI/gpt-neo-125M"
download_model "EleutherAI/gpt-neo-1.3B"

# --- TinyStories (Various) ---
# Extremely small models trained on synthetic data
download_model "roneneldan/TinyStories-1M"
download_model "roneneldan/TinyStories-3M"
download_model "roneneldan/TinyStories-8M"
download_model "roneneldan/TinyStories-28M"
download_model "roneneldan/TinyStories-33M"
download_model "roneneldan/TinyStories-Instruct-1M"
download_model "roneneldan/TinyStories-Instruct-3M"
download_model "roneneldan/TinyStories-Instruct-8M"
download_model "roneneldan/TinyStories-Instruct-28M"
download_model "roneneldan/TinyStories-Instruct-33M"

# --- Microsoft Phi Family (MIT) ---
download_model "microsoft/phi-1"
download_model "microsoft/phi-1_5"
download_model "microsoft/phi-2" # ~2.7B but very popular, might be bordering small

# --- StableLM (Apache 2.0) ---
download_model "stabilityai/stablelm-zephyr-3b" # A bit larger
download_model "stabilityai/stablelm-3b-4e1t"    # A bit larger
download_model "stabilityai/stablelm-2-1_6b"
download_model "stabilityai/stablelm-2-zephyr-1_6b"

# --- Sheared LLaMA (Apache 2.0) ---
# Pruned versions of Llama
download_model "princeton-nlp/Sheared-LLaMA-1.3B"
download_model "princeton-nlp/Sheared-LLaMA-2.7B" # Borderline

# --- MobiLlama (Apache 2.0) ---
download_model "MBZUAI/MobiLlama-0.5B"
download_model "MBZUAI/MobiLlama-1B"
download_model "MBZUAI/MobiLlama-0.5B-Chat"
download_model "MBZUAI/MobiLlama-1B-Chat"

# --- OpenELM (Apple - Apple Sample Code License - Careful here, usually open enough for research) ---
download_model "apple/OpenELM-270M"
download_model "apple/OpenELM-450M"
download_model "apple/OpenELM-1_1B"
download_model "apple/OpenELM-270M-Instruct"
download_model "apple/OpenELM-450M-Instruct"
download_model "apple/OpenELM-1_1B-Instruct"

# --- H2O Danube (Apache 2.0) ---
download_model "h2oai/h2o-danube-1.8b-base"
download_model "h2oai/h2o-danube-1.8b-chat"
download_model "h2oai/h2o-danube2-1.8b-base"
download_model "h2oai/h2o-danube2-1.8b-chat"
download_model "h2oai/h2o-danube3-500m-base"
download_model "h2oai/h2o-danube3-500m-chat"

# --- OLMo (Apache 2.0) ---
download_model "allenai/OLMo-1B-0724"
download_model "allenai/OLMo-1B"

# --- DeepSeek - Coder (MIT) ---
download_model "deepseek-ai/deepseek-coder-1.3b-base"
download_model "deepseek-ai/deepseek-coder-1.3b-instruct"

# --- Other Interesting Small Models ---
download_model "Locutusque/TinyMistral-248M"
download_model "GeneZC/MiniChat-1.5-3B" # Maybe too big
download_model "ibm/granite-3b-code-base" # Maybe too big
download_model "google/gemma-2b"
download_model "google/gemma-2b-it"
download_model "google/recurrentgemma-2b"
download_model "google/recurrentgemma-2b-it"
download_model "SofiTesfay2010/HRM-LLM"

echo "----------------------------------------------------------------"
echo "🎉 Process finished. Attempted $counter model family downloads."
echo "Run ./neuralwave to experiment!"
