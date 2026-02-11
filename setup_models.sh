#!/bin/bash

# Ensure Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama could not be found. Please install it with 'brew install ollama' first."
    exit 1
fi

echo "Pulling models for Intelligence Arbitrage experiment..."

# Core Models
echo "Pulling Llama 3.2 3B..."
ollama pull llama3.2:3b

echo "Pulling Llama 3.3 8B Instruct (Q4_K_M)..."
ollama pull llama3.3:8b-instruct-q4_K_M

echo "Pulling Qwen 2.5 7B Instruct (Q4_K_M)..."
ollama pull qwen2.5:7b-instruct-q4_K_M

echo "Pulling Mistral 7B Instruct v0.3 (Q4_K_M)..."
ollama pull mistral:7b-instruct-v0.3-q4_K_M

echo "Pulling Phi-4 14B..."
ollama pull phi4:latest

echo "All models pulled successfully!"
