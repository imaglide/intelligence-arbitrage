import dspy
import requests
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import Config

def check_ollama():
    print("Checking Ollama connection...")
    try:
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = [m['name'] for m in response.json()['models']]
            print(f"✅ Ollama is up. Found models: {models}")
            return models
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Could not connect to Ollama: {e}")
        return []

def test_dspy_inference(model_name):
    print(f"Testing DSPy inference with {model_name}...")
    try:
        lm = dspy.LM(
            f"ollama_chat/{model_name}",
            api_base=Config.OLLAMA_BASE_URL,
            api_key=""
        )
        dspy.configure(lm=lm)
        
        response = lm("What is 2 + 2? Return only the number.")
        print(f"Response: {response}")
        
        if "4" in response[0]:
            print("✅ DSPy inference successful!")
            return True
        else:
            print(f"⚠️ Unexpected response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ DSPy inference failed: {e}")
        return False

if __name__ == "__main__":
    models = check_ollama()
    if not models:
        sys.exit(1)
        
    # Try to find a model we recognize, or just use the first one
    test_model = None
    for required_model in Config.MODELS.values():
        if required_model in models:
            test_model = required_model
            break
            
    if not test_model and models:
        test_model = models[0]
        print(f"⚠️ specific experiment models not found, testing with available model: {test_model}")
        
    if test_model:
        success = test_dspy_inference(test_model)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("❌ No models found to test.")
        sys.exit(1)
