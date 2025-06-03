import uuid
from openai import OpenAI
import os
import json

def openai_llm_call(prompt_content: str, purpose: str, json_format: str) -> any:
    """
    Makes a call to the OpenAI API and parses the JSON response.
    """
    print(f"\n--- OpenAI LLM Call ({purpose}) ---")
    client = OpenAI(base_url="")  
    
    if not  os.getenv("OPENAI_API_KEY"):
        assert("Error: OPENAI_API_KEY not found. Please set it as an environment variable.")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a JSON generation assistant. Return ONLY valid JSON that strictly follows this format: {json_format}. No explanations, no markdown, just pure JSON."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        raw_response_content = response.choices[0].message.content
        
        # Parse JSON
        parsed_response = json.loads(raw_response_content)
        return parsed_response

    except client.APIError as e:
        print(f"OpenAI API Error ({purpose}): {e}")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error ({purpose}): Failed to parse LLM response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during LLM call ({purpose}): {e}")
    



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
    
    sample_theme = "Modern Technology Woes"
    
    print("\n--- Testing gen_ideas.py with OpenAI Integration ---")
    first_obs = generate_first_order_observations(sample_theme)
    second_obs = generate_second_order_observations(first_obs, sample_theme) if first_obs else []
    
    all_obs = first_obs + second_obs
    if all_obs:
        ideas = formulate_joke_ideas(all_obs, sample_theme)
        print(f"\nGenerated {len(ideas)} joke ideas.")
    else:
        print("\nNo observations were generated.")
