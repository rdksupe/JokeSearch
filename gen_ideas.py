import uuid
from openai import OpenAI
import os
import json
import sys
import re
from utils.config import get_api_base_url, get_openai_key, DEFAULT_MODEL

def openai_llm_call(prompt_content: str, purpose: str, json_format: str) -> dict:
    """
    Makes a call to the OpenAI API and parses the JSON response.
    
    Args:
        prompt_content: The prompt to send to the API
        purpose: The purpose of this call (for logging)
        json_format: Expected JSON format description
        
    Returns:
        Parsed JSON response
    """
    print(f"\n--- OpenAI LLM Call ({purpose}) ---")
    
    # Get API configuration from environment
    api_base_url = get_api_base_url()
    api_key = get_openai_key()
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in configuration.")
        sys.exit(1)
    
    try:
        # Initialize client with API details
        client = OpenAI(base_url=api_base_url, api_key=api_key)
        
        # Prepare system prompt that explicitly asks for JSON
        system_prompt = (
            f"You are a JSON generation assistant. Return a JSON object with this structure: {json_format}. "
            f"Do not include explanations or markdown formatting, just the pure JSON object."
        )
        
        # Make the API call
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
        )
        
        # Get raw response
        raw_content = response.choices[0].message.content
        print(f"Raw response (first 100 chars): {raw_content[:100]}...")
        
        # Clean up response and extract JSON
        clean_json = extract_valid_json(raw_content)
        
        try:
            return json.loads(clean_json)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from response: {clean_json}")
            print(f"JSON error: {e}")
            return fallback_json_extraction(raw_content, purpose)
    
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def extract_valid_json(text):
    """
    Extract valid JSON from text that might include code blocks, explanations, etc.
    """
    # First try to extract from code blocks if present
    if "```json" in text:
        parts = text.split("```json", 1)
        if len(parts) > 1:
            # Get content after ```json marker
            json_part = parts[1]
            # Find the closing ```
            if "```" in json_part:
                # Extract content between ```json and the next ```
                json_text = json_part.split("```", 1)[0].strip()
                return json_text
    
    # If there's a code block without language specification
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            # Get content after first ``` marker
            json_part = parts[1]
            # Find the closing ```
            if "```" in json_part:
                # Extract content between first ``` and the next ```
                json_text = json_part.split("```", 1)[0].strip()
                return json_text
    
    # If no code blocks or extraction failed, try to find JSON between { and }
    # This looks for the outermost matching braces in the text
    text = text.strip()
    start_idx = text.find('{')
    if start_idx != -1:
        # Find matching closing brace by counting opening and closing braces
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON - return it
                    return text[start_idx:i+1]
    
    # If all else fails, return the original text
    return text

def fallback_json_extraction(text, purpose):
    """
    Last resort extraction of JSON when parsing fails.
    Attempts to build a valid JSON structure based on the purpose.
    """
    print(f"Attempting fallback JSON extraction for {purpose}...")
    
    # For first-order or second-order observations
    if purpose in ["first_order_observations", "second_order_observations"]:
        # Try to extract a list of items
        items = re.findall(r'"([^"]*)"', text)
        if not items:
            # Try to find numbered or bullet items
            items = re.findall(r'(?:\d+\.|\*)\s*(.*?)(?=(?:\d+\.|\*)|$)', text)
        
        if items:
            return {"observations": [item.strip() for item in items]}
        else:
            print("Fallback extraction failed. Using default observations.")
            return {"observations": ["Observation 1", "Observation 2", "Observation 3"]}
    
    # For joke ideas
    elif purpose == "formulate_joke_ideas":
        # Try to extract concepts
        concepts = []
        pattern = r'(?:"concept"\s*:\s*"([^"]*)"|(?:\d+\.|\*)\s*(.*?)(?=(?:\d+\.|\*)|$))'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Each match is a tuple with groups from the pattern
            # Take the non-empty group
            concept = next((m for m in match if m), "").strip()
            if concept:
                concepts.append({"concept": concept})
        
        if concepts:
            return {"ideas": concepts}
        else:
            print("Fallback extraction failed. Using default ideas.")
            return {"ideas": [{"concept": "Default joke idea 1"}, {"concept": "Default joke idea 2"}]}
    
    # Default case
    print("Unknown purpose for fallback extraction. Using empty object.")
    return {}

def generate_first_order_observations(theme: str) -> list[str]:
    """
    Generates initial broad 'observations' or 'humor angles' related to the theme.
    """
    prompt_content = (
        f"For the theme '{theme}', generate 3-5 diverse, high-level 'observations' or 'humor angles'. "
        f"These should be broad starting points for jokes. Focus on common frustrations, ironies, or absurdities related to the theme."
    )
    json_format = '{"observations": ["observation1", "observation2", "observation3", ...]}'
    
    response = openai_llm_call(prompt_content, "first_order_observations", json_format)
    observations = response.get("observations", [])
    
    print(f"Generated First-Order Observations: {observations}")
    return observations


def generate_second_order_observations(first_order_observations: list[str], theme: str) -> list[str]:
    """
    Generates second-order observations by combining or elaborating on first-order ones.
    """
    if not first_order_observations:
        return []
    
    prompt_content = (
        f"Given the theme '{theme}' and the following first-order humor observations: {first_order_observations}. "
        f"Derive 2-3 new, more specific, or nuanced humor angles or observations by building upon or combining these."
    )
    json_format = '{"observations": ["specific_angle1", "specific_angle2", ...]}'
    
    response = openai_llm_call(prompt_content, "second_order_observations", json_format)
    observations = response.get("observations", [])
    
    print(f"Generated Second-Order Observations: {observations}")
    return observations


def formulate_joke_ideas(all_observations: list[str], theme: str) -> list[dict]:
    """
    Formulates specific joke ideas or concepts based on the generated observations.
    Each idea will have a unique ID added client-side.
    """
    if not all_observations:
        return []
    
    prompt_content = (
        f"Based on the theme '{theme}' and the following humor observations: {all_observations}. "
        f"Formulate 3-5 diverse and specific joke ideas or concepts. Each idea should represent a distinct conceptual direction for a joke."
    )
    json_format = '{"ideas": [{"concept": "joke concept 1"}, {"concept": "joke concept 2"}, ...]}'
    
    response = openai_llm_call(prompt_content, "formulate_joke_ideas", json_format)
    
    # Add unique IDs to each idea
    joke_ideas = []
    for idea in response.get("ideas", []):
        if isinstance(idea, dict) and "concept" in idea:
            joke_ideas.append({
                "id": str(uuid.uuid4()),
                "concept": idea["concept"]
            })
    
    print(f"Formulated Joke Ideas: {joke_ideas}")
    return joke_ideas


if __name__ == '__main__':
    from utils.config import initialize_config
    
    if not initialize_config():
        print("Failed to initialize configuration. Please check your .env file.")
        sys.exit(1)
    
    sample_theme = "Modern Technology Woes"
    
    print("\n--- Testing gen_ideas.py with OpenAI Integration ---")
    first_obs = generate_first_order_observations(sample_theme)
    second_obs = generate_second_order_observations(first_obs, sample_theme)
    
    all_obs = first_obs + second_obs
    ideas = formulate_joke_ideas(all_obs, sample_theme)
    print(f"\nGenerated {len(ideas)} joke ideas.")
