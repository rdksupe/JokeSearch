import uuid
from openai import OpenAI
import os
import json
import sys

def openai_llm_call(prompt_content: str, purpose: str, json_format: str) -> dict:
    """
    Makes a call to the OpenAI API and parses the JSON response.
    Simplified version that exits on failure.
    
    Args:
        prompt_content: The prompt to send to the API
        purpose: The purpose of this call (for logging)
        json_format: Expected JSON format description
        
    Returns:
        Parsed JSON response
    """
    print(f"\n--- OpenAI LLM Call ({purpose}) ---")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please set it as an environment variable.")
        sys.exit(1)
    
    try:
        # Initialize client with local API
        client = OpenAI(base_url="http://localhost:1234/v1/")
        
        # Prepare system prompt that explicitly asks for JSON
        system_prompt = (
            f"You are a JSON generation assistant. Return a JSON object with this structure: {json_format}. "
            f"Do not include explanations or markdown formatting, just the pure JSON object."
        )
        
        # Make the API call
        response = client.chat.completions.create(
            model="gemma-3-4b-it-qat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
        )
        
        # Get raw response
        raw_content = response.choices[0].message.content
        print(f"Raw response (first 100 chars): {raw_content[:100]}...")
        
        # Clean up response if it's in a code block
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON from response: {raw_content}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    
    sample_theme = "Modern Technology Woes"
    
    print("\n--- Testing gen_ideas.py with OpenAI Integration ---")
    first_obs = generate_first_order_observations(sample_theme)
    second_obs = generate_second_order_observations(first_obs, sample_theme)
    
    all_obs = first_obs + second_obs
    ideas = formulate_joke_ideas(all_obs, sample_theme)
    print(f"\nGenerated {len(ideas)} joke ideas.")
