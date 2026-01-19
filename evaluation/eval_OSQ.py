# OSQ Evaluation Code
import json
import re
import os
import base64
import requests
from tqdm import tqdm

# OSQ Scoring Prompt
prompt_OSQ = '''
# Role
You are an impartial judge who is familiar with Internet culture and memes, and is good at digging out and analyzing the deep meaning of Internet memes.

## Attention
You are responsible for evaluating the quality of the answer provided by the model for Internet culture and memes. Your evaluation should refer to the human answer and image, and score based on the Evaluation Standard.

## Evaluation Standard:
- [1 point]: Failed to capture important elements in the image (such as text, important entities), and did not point out the emotion, domain, and rhetorical devices. Only stayed on the surface information description, lacking depth and creativity, with a huge gap from the standard answer.
  
- [2 points]: Able to partially capture important elements in the image, but the identification of emotion, domain, and rhetorical devices is vague. The surface information description is relatively complete, but there is a clear deficiency in digging into the deep metaphor, with a noticeable gap from the standard answer.
  
- [3 points]: Able to capture important elements in the image relatively well, and preliminarily point out the emotion, domain, and rhetorical devices. The surface information description is relatively accurate, with some relevance to the metaphor expressed, but there is still room for improvement in depth and creativity. Overall close to the standard answer.
  
- [4 points]: Able to accurately capture important elements in the image, and clearly point out the emotion, domain, and rhetorical devices. The surface information description is detailed and accurate, with relatively deep mining of the metaphor, showing certain creativity and depth. Overall consistent with the standard answer, but slightly insufficient in some details or depth.
  
- [5 points]: Accurately capture important elements in the image, and profoundly point out the emotion, domain, and rhetorical devices. The surface information description is comprehensive and precise, with unique insights into the metaphor, able to cleverly blend image elements with the metaphorical mood, showing extremely high creativity and depth. Highly consistent with the standard answer, showing a deep grasp of metaphor creation and cultural understanding.

## Standrad Answer:
human answer: {}

## Constraints
- Avoid any position biases and be as objective as possible
- Do not allow the length of the descriptions to influence your evaluation
- Output your final directly by strictly following this format: "[ratings]"

## Example:
Input: model answer   
Output: [x points]

## Solve:
model answer: {}
'''

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def GPT_OSQ_EN(url, prompt, explanation, image_implication):
    """Score English answer using GPT-4"""
    base64_image = encode_image(url)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt.format(explanation, image_implication)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
        ]}
    ]
    api_key = "xxx"  # Replace with your actual API key
    proxy_api_url = 'xxx'  # Replace with your actual proxy API URL
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    data = {'model': 'gpt-4o-2024-11-20', 'messages': messages, 'temperature': 0, 'top_p': 0.9}
    response = requests.post(proxy_api_url, headers=headers, json=data)
    score = response.json()["choices"][0]["message"]["content"]
    return score

def GPT_OSQ_ZH(url, prompt, metaphorical_meaning, explanation, image_implication):
    """Score Chinese answer using GPT-4"""
    base64_image = encode_image(url)
    human_answer = metaphorical_meaning + ';' + explanation
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt.format(human_answer, image_implication)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
        ]}
    ]
    api_key = "xxx"  # Replace with your actual API key
    proxy_api_url = 'xxx'  # Replace with your actual proxy API URL
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    data = {'model': 'gpt-4o-2024-11-20', 'messages': messages, 'temperature': 0, 'top_p': 0.9}
    response = requests.post(proxy_api_url, headers=headers, json=data)
    score = response.json()["choices"][0]["message"]["content"]
    return score

def run_evaluation(data_path, output_path, language='en'):
    """
    Run OSQ evaluation
    
    Args:
    data_path: Path to input data JSON file
    output_path: Path to output result JSON file
    language: Language type, 'en' or 'zh'
    """
    # Read main dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if result file exists, if so read existing data
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Create set of processed URLs for quick lookup
        processed_urls = {result['url'] for result in results}
        print(f"Processed {len(processed_urls)} images")
    else:
        results = []
        processed_urls = set()
        print("No existing result file found, starting from scratch")
    
    # Iterate through all data with progress bar
    for idx, item in enumerate(tqdm(data, desc="Processing images")):
        try:
            url = item['Image_URL']
            
            # Check if this URL has been processed
            if url in processed_urls:
                print(f"Skipping processed image: {url}")
                continue
            
            explanation = item['Ground_Truth']
            print(explanation)
            image_implication = item['Model_Output']
            
            # Select scoring function based on language
            if language == 'zh':
                metaphorical_meaning = item.get('Metaphorical_Meaning', '')
                score = GPT_OSQ_ZH(url, prompt_OSQ, metaphorical_meaning, explanation, image_implication)
            else:
                score = GPT_OSQ_EN(url, prompt_OSQ, explanation, image_implication)
            
            print(score)
            print("------")
            
            # Store results
            result = {
                'id': idx,
                'url': url,
                'human_answer': explanation,
                'model_answer': image_implication,
                'score': score
            }
            
            # Add result to list
            results.append(result)
            processed_urls.add(url)  # Add URL to processed set
            
            # Write updated results to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
        except Exception as e:
            # Handle exceptions
            print(f'Error processing image {url}: {e}')
            continue
    
    return results

def calculate_average_score(results_path, output_txt_path):
    """
    Calculate average score and save results
    
    Args:
    results_path: Path to scoring result JSON file
    output_txt_path: Path to output text file
    """
    # Read JSON file
    with open(results_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    count = 0
    total_score = 0
    
    # Iterate through each instance, accumulate score and count
    for instance in data:
        count += 1
        score_str = instance.get('score', '[]')
        score_match = re.search(r'\d+', score_str)
        score = int(score_match.group()) if score_match else 0
        total_score += score
    
    # Calculate overall average score
    average_score = total_score / count if count > 0 else 0
    print(f'Overall average score: {average_score:.2f}, total {count} instances')
    
    summary = f"Overall average score: {average_score:.2f}, total {count} instances"
    
    # Save results to file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    return average_score, count

def main():
    """Main function"""
    # ⭐⭐⭐ Set paths ⭐⭐⭐
    os.chdir('evaluation')
    
    # Input/Output paths
    input_path = 'evaluation/OSQ/eval_result.json'
    output_score_path = 'experiment/en/OSQ/eval_result_score.json'
    output_summary_path = 'results/en/OSQ/answer_eval_result_score.txt'
    
    # Language setting: 'en' or 'zh'
    language = 'en'
    
    # Run evaluation
    print("Starting OSQ evaluation...")
    results = run_evaluation(input_path, output_score_path, language)
    
    # Calculate and save average score
    print("\nCalculating average score...")
    average_score, count = calculate_average_score(output_score_path, output_summary_path)
    
    print(f"\nEvaluation completed!")
    print(f"Overall average score: {average_score:.2f}")
    print(f"Number of evaluated instances: {count}")
    print(f"Scoring results saved to: {output_score_path}")
    print(f"Summary saved to: {output_summary_path}")

if __name__ == "__main__":
    main()
