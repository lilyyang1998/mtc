import os
import requests
import logging
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
print("Loading environment variables...")
load_dotenv()

# API configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHERAI_API_KEY")

print("Checking API keys...")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found!")
if not TOGETHER_API_KEY:
    logger.warning("Together API key not found!")

# Initialize API client
print("Initializing OpenAI client...")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Load default prompt from secret file
def load_default_prompt():
    try:
        # Try to load from /etc/secrets first (production)
        prompt_path = "/etc/secrets/default_prompt.txt"
        if not os.path.exists(prompt_path):
            # Fallback to local secrets directory (development)
            prompt_path = "secrets/default_prompt.txt"
        
        with open(prompt_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error loading default prompt: {str(e)}")
        return "You are a kind and compassionate language model."

DEFAULT_PROMPT = load_default_prompt()

# Model configurations
OPENAI_MODELS = {
    "GPT-4.1": "gpt-4.1",
    "GPT-4o": "gpt-4o",
    "GPT-o1": "o1"
}

TOGETHER_MODELS = {
    "Llama-3.3-70B-Instruct-Turbo": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3"
}

# Available models
AVAILABLE_MODELS = list(OPENAI_MODELS.keys()) + list(TOGETHER_MODELS.keys())

def call_openai_api(text, prompt, model_name):
    """Call OpenAI API for text simplification"""
    try:
        print(f"\nProcessing OpenAI API request with model: {model_name}")
        print(f"Input text length: {len(text)} characters")
        logger.info(f"Starting OpenAI API call with model {model_name}")
        
        start_time = datetime.now()
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Text: {text}"}
        ]
        
        response = openai_client.chat.completions.create(
            model=OPENAI_MODELS[model_name],
            messages=messages,
            temperature=1
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"OpenAI API request completed in {processing_time:.2f} seconds")
        logger.info(f"OpenAI API request completed successfully in {processing_time:.2f} seconds")
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"OpenAI API Error: {str(e)}"
        print(f"Error: {error_msg}")
        logger.error(error_msg)
        return f"Error with OpenAI API: {str(e)}"

def clean_deepseek_output(response):
    """Clean DeepSeek output by removing thinking process"""
    try:
        print("Cleaning DeepSeek output...")
        if "<think>" in response and "</think>" in response:
            print("Found thinking tags, removing them...")
            actual_content = response.split("</think>")[-1].strip()
            return actual_content
        return response
    except Exception as e:
        error_msg = f"Error cleaning DeepSeek output: {str(e)}"
        print(f"Error: {error_msg}")
        logger.error(error_msg)
        return response

def call_together_api(text, prompt, model_name):
    """Call Together.ai API for text simplification"""
    try:
        print(f"\nProcessing Together API request with model: {model_name}")
        print(f"Input text length: {len(text)} characters")
        logger.info(f"Starting Together API call with model {model_name}")
        
        start_time = datetime.now()
        url = "https://api.together.xyz/v1/chat/completions"
        
        payload = {
            "model": TOGETHER_MODELS[model_name],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Please simplify this text: {text}"}
            ],
            "temperature": 1
        }
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {TOGETHER_API_KEY}"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"Together API request completed in {processing_time:.2f} seconds")
        logger.info(f"Together API request completed successfully in {processing_time:.2f} seconds")
        
        response_content = response_json["choices"][0]["message"]["content"]
        
        if model_name == "DeepSeek-R1":
            response_content = clean_deepseek_output(response_content)
            
        return response_content
    except Exception as e:
        error_msg = f"Together.ai API Error: {str(e)}"
        print(f"Error: {error_msg}")
        logger.error(error_msg)
        return f"Error with Together.ai API: {str(e)}"

def simplify_text(text, prompt, model):
    """Main function for text simplification"""
    print(f"\nStarting text simplification with model: {model}")
    logger.info(f"Starting text simplification process with model: {model}")
    
    if not text:
        print("Error: No input text provided")
        logger.warning("Attempt to simplify empty text")
        return "Please enter text to simplify"
    
    print(f"Text length: {len(text)} characters")
    print(f"Prompt length: {len(prompt)} characters")
    
    if model in OPENAI_MODELS:
        print(f"Using OpenAI model: {model}")
        return call_openai_api(text, prompt, model)
    elif model in TOGETHER_MODELS:
        print(f"Using Together model: {model}")
        return call_together_api(text, prompt, model)
    else:
        error_msg = f"Invalid model selected: {model}"
        print(f"Error: {error_msg}")
        logger.error(error_msg)
        return "Invalid model selected"

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS, default_prompt=DEFAULT_PROMPT)

@app.route('/simplify', methods=['POST'])
def simplify():
    try:
        data = request.json
        text = data.get('text', '')
        prompt = data.get('prompt', DEFAULT_PROMPT)
        model = data.get('model', AVAILABLE_MODELS[0])
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        simplified_text = simplify_text(text, prompt, model)
        return jsonify({'result': simplified_text})
    except Exception as e:
        logger.error(f"Error in simplify endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting MTC Text Simplification Tool")
    print("="*50)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))