"""
TFQ (True/False Question) Dataset Construction
This script generates True/False questions based on images and their descriptions.
"""

from openai import OpenAI
import base64
import requests
import json
import os
import re
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm


# ======================== Configuration ========================

# Image Description Prompt
PROMPT_IMAGE_DESCRIPTION = '''
Please provide a description of the image in a paragraph. You should consider the role, text, color, and layout of the image. Focus on the text and important elements in the image. Try to be concise while ensuring the correctness of the description.
'''

# TFQ Generation Prompt
PROMPT_TFQ_GENERATION = """
# Role
You are a researcher who is familiar with Internet culture and memes, and is good at digging out and analyzing the deep meaning of Internet memes.

## Attention
You are responsible for generating 5-10 judgment questions based on pictures and picture descriptions. Each judgment question consists of arguments and evidence, examines the key content of the picture and is related to the metaphor of the picture, and gives the correct answer.

## Skills
You have image analysis ability, metaphor understanding ability, educational test design ability and logical thinking ability. You can combine the key content of the picture with the metaphor and design judgment questions that both test knowledge and trigger thinking.

## Workflow:
1. Carefully analyze the picture and picture description to extract the key content and metaphorical meaning.
2. Design judgment questions based on the extracted information to ensure that the questions are related to the key content and metaphor of the picture.
3. Give the correct answer to each judgment question and ensure that the answer is accurate.

## Constraints
- True or False questions must be related to the key content and metaphor of the picture, ensuring that the questions are clear and accurate
- The difficulty of the questions is layered, wrong questions are highly confusing, and correct questions should have clear basis
- The format is unified as "[T/F]: question"

## OutputFormat:
A list of true or false questions, with the correct answer attached to each question.

## Example:
- Input: Picture description: "A painting depicting a lonely deer in the forest, a metaphor for the harmonious coexistence of loneliness and nature."
- Output:
1. [T]: The deer in the picture symbolizes loneliness, because the deer stands alone in the forest, with no other animals around.
2. [F]: The painting expresses the deer's fear, because the deer's eyes reveal fear.
3. [F]: The metaphor of this painting is the conflict between humans and nature, because the deer seems to be running away from some threat.
4. [T]: The fawn in the forest symbolizes the loneliness of human beings, as it faces the vast natural environment alone.
5. [F]: The position of the fawn in the forest suggests its isolation from nature, as it is surrounded by trees and appears isolated and helpless.

## Solve:
Image description: {}
"""

# API Configuration
API_KEY = "xxx"  # Replace with your actual API key
PROXY_API_URL = "xxx"  # Replace with your actual proxy API URL
MODEL_NAME = "gpt-4.1"

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds


# ======================== Utility Functions ========================

def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def retry_on_failure(func, max_retries: int = MAX_RETRIES, retry_delay: int = RETRY_DELAY):
    """
    Retry decorator for API calls.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Result from the function if successful
        
    Raises:
        Exception: If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            print(f"Error calling {func.__name__}: {e}. Retry attempt {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"{func.__name__} call failed after {max_retries} attempts, skipping this item.")
                raise e


def parse_true_false_questions(text: str) -> List[Dict[str, str]]:
    """
    Parse true/false questions from text response.
    
    Args:
        text: Raw text containing questions in format "[T/F]: question"
        
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    lines = text.strip().split('\n')
    questions_data = []
    
    for line in lines:
        match = re.match(r'\d+\.\s*\[([TF])\]:\s*(.*)', line)
        if match:
            questions_data.append({
                'question': match.group(2),
                'answer': match.group(1)
            })
    
    return questions_data


def extract_test_number(url: str) -> int:
    """
    Extract test number from URL for sorting.
    
    Args:
        url: URL string containing "test-X" pattern
        
    Returns:
        Test number as integer, or infinity if not found
    """
    match = re.search(r'test-(\d+)', url)
    return int(match.group(1)) if match else float('inf')


# ======================== API Functions ========================

def get_image_description(prompt: str, image_path: str) -> str:
    """
    Get image description from GPT model.
    
    Args:
        prompt: Prompt for image description
        image_path: Path to the image file
        
    Returns:
        Image description text
    """
    base64_image = encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    data = {
        'model': MODEL_NAME,
        'messages': messages,
        'temperature': 0.7,
        'top_p': 0.9
    }
    
    response = requests.post(PROXY_API_URL, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]


def generate_questions(image_path: str, prompt: str, image_implication: str) -> List[Dict[str, str]]:
    """
    Generate true/false questions based on image and description.
    
    Args:
        image_path: Path to the image file
        prompt: Prompt template for question generation
        image_implication: Image metaphorical meaning/implication
        
    Returns:
        List of generated questions with answers
    """
    base64_image = encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.format(image_implication)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    data = {
        'model': MODEL_NAME,
        'messages': messages,
        'temperature': 0.5,
        'top_p': 0.9
    }
    
    response = requests.post(PROXY_API_URL, headers=headers, json=data)
    response.raise_for_status()
    
    raw_content = response.json()["choices"][0]["message"]["content"]
    return parse_true_false_questions(raw_content)


# ======================== Main Processing Functions ========================

def load_existing_results(result_file: str) -> tuple[List[Dict], set]:
    """
    Load existing results from file if it exists.
    
    Args:
        result_file: Path to the results JSON file
        
    Returns:
        Tuple of (results list, set of processed URLs)
    """
    results = []
    processed_urls = set()
    
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            for item in results:
                processed_urls.add(item['url'])
        print(f"Loaded {len(results)} existing results, will skip {len(processed_urls)} processed URLs.")
    
    return results, processed_urls


def save_results(results: List[Dict], result_file: str):
    """
    Save results to JSON file.
    
    Args:
        results: List of result dictionaries
        result_file: Path to save the results
    """
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def process_single_item(item: Dict, idx: int) -> Optional[Dict]:
    """
    Process a single data item to generate TFQ.
    
    Args:
        item: Data item containing image path and metadata
        idx: Index of the item
        
    Returns:
        Result dictionary or None if processing fails
    """
    url = item['local_path']
    
    try:
        # Get image description with retry logic
        print(f"\nProcessing image {idx}: {url}")
        image_dep = retry_on_failure(
            lambda: get_image_description(PROMPT_IMAGE_DESCRIPTION, url)
        )
        print(f"Image description: {image_dep[:100]}...")
        print('-' * 50)
        
        # Get image implication from metadata
        image_implication = item['meta_data']['explanation']
        print(f"Image implication: {image_implication[:100]}...")
        print('-' * 50)
        
        # Generate questions with retry logic
        questions = retry_on_failure(
            lambda: generate_questions(url, PROMPT_TFQ_GENERATION, image_implication)
        )
        print(f"Generated {len(questions)} questions")
        
        # Organize result
        result = {
            'id': idx,
            'url': url,
            'image_dep': image_dep,
            'image_implication': image_implication,
            'True_false_questions': {
                'questions': [q['question'] for q in questions],
                'answers': [q['answer'] for q in questions]
            }
        }
        
        return result
        
    except Exception as e:
        print(f'Error processing image {url}: {e}')
        return None


def sort_and_reindex_results(results: List[Dict]) -> List[Dict]:
    """
    Sort results by test number and reindex IDs.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Sorted and reindexed results
    """
    # Sort by test number extracted from URL
    results.sort(key=lambda x: extract_test_number(x['url']))
    
    # Regenerate IDs
    for new_id, item in enumerate(results):
        item['id'] = new_id
    
    return results


def build_tfq_dataset(
    input_file: str,
    output_file: str,
    save_progress: bool = True
):
    """
    Main function to build TFQ dataset.
    
    Args:
        input_file: Path to input JSON file containing image data
        output_file: Path to output JSON file for results
        save_progress: Whether to save progress after each item
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load existing results if any
    results, processed_urls = load_existing_results(output_file)
    
    # Filter unprocessed items
    unprocessed_data = [
        item for item in data
        if item.get('local_path') and item.get('local_path') not in processed_urls
    ]
    
    print(f"Total data: {len(data)}, Unprocessed data: {len(unprocessed_data)}")
    
    # Process each item
    for idx, item in enumerate(tqdm(unprocessed_data, desc="Processing images")):
        result = process_single_item(item, len(results) + idx)
        
        if result is not None:
            results.append(result)
            
            # Save progress after each successful processing
            if save_progress:
                save_results(results, output_file)
    
    # Final sorting and saving
    print("\nAll processing complete. Sorting and saving final results...")
    results = sort_and_reindex_results(results)
    save_results(results, output_file)
    
    print(f"Processing complete! Total {len(results)} results sorted and saved to {output_file}")


# ======================== Main Entry Point ========================

def main():
    """Main entry point for the script."""
    # Configuration
    INPUT_FILE = 'dataset/II-Bench.json'
    OUTPUT_FILE = 'dataset/train/TFQ_II-Bench.json'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Build dataset
    build_tfq_dataset(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
