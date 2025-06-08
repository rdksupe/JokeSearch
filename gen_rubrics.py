# gen_rubrics.py
import uuid
import os
import json
import sys
import re
from openai import OpenAI
from openai import APIError, BadRequestError  # Import specific exceptions
from utils.config import get_api_base_url, get_openai_key, DEFAULT_MODEL

def _extract_and_clean_json(raw_text):
    """Extract and clean JSON from text, handling code blocks and invalid characters"""
    # First, extract JSON content if it's in a code block
    if "```json" in raw_text:
        parts = raw_text.split("```json", 1)
        if len(parts) > 1 and "```" in parts[1]:
            raw_text = parts[1].split("```", 1)[0].strip()
    elif "```" in raw_text:
        parts = raw_text.split("```", 1)
        if len(parts) > 1 and "```" in parts[1]:
            raw_text = parts[1].split("```", 1)[0].strip()
    
    # Clean control characters that might cause JSON parsing to fail
    # Replace various problematic characters with appropriate replacements
    cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', raw_text)
    
    # Replace Unicode quotes with straight quotes
    cleaned_text = cleaned_text.replace('"', '"').replace('"', '"')
    cleaned_text = cleaned_text.replace("'", "'").replace("'", "'")
    
    # Also handle escaped characters like \n \t etc.
    cleaned_text = cleaned_text.replace('\\n', ' ').replace('\\t', ' ')
    
    return cleaned_text

def _openai_llm_call(prompt_content: str, purpose: str, expected_format_description: str) -> any:
    """
    Makes a call to the OpenAI API and parses the response.
    """
    print(f"\n--- OpenAI LLM Call ({purpose}) ---")
    print(f"System Instruction: {expected_format_description}")
    print(f"User Prompt (first 200 chars): {prompt_content[:200]}...")

    # Get API configuration from environment
    api_base_url = get_api_base_url()
    api_key = get_openai_key()
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in configuration.")
        return _fallback_placeholder_response(purpose)

    try:
        # Initialize client with API configuration
        client = OpenAI(base_url=api_base_url, api_key=api_key)
        
        response = client.chat.completions.create(
            model=DEFAULT_MODEL, 
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Your response should be a pure JSON object with this structure: {expected_format_description}. No markdown, no explanations, just the JSON object."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
        )
        
        raw_response_content = response.choices[0].message.content
        print(f"Raw LLM Response (first 100 chars): {raw_response_content[:100]}...")
        
        # Extract and clean JSON from response before parsing
        try:
            cleaned_json = _extract_and_clean_json(raw_response_content)
            parsed_response = json.loads(cleaned_json)
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error ({purpose}): {e}")
            print("Attempting alternate JSON extraction...")
            
            # Try to extract JSON between curly braces
            match = re.search(r'\{.+\}', raw_response_content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)
                    cleaned_json = _extract_and_clean_json(json_str)
                    return json.loads(cleaned_json)
                except json.JSONDecodeError:
                    pass
            
            # Fall back to placeholder if all parsing fails
            print("Failed to extract valid JSON")
            return _fallback_placeholder_response(purpose)

    except APIError as e:  # Using the imported APIError
        print(f"OpenAI API Error ({purpose}): {e}")
    except BadRequestError as e:  # Using the imported BadRequestError
        print(f"OpenAI Bad Request Error ({purpose}): {e}")
    except Exception as e:
        print(f"An unexpected error occurred during LLM call ({purpose}): {e}")
        import traceback
        traceback.print_exc()
    
    return _fallback_placeholder_response(purpose)

def _fallback_placeholder_response(purpose: str, idea_id: str = None) -> any:
    """Provides a fallback response if the LLM call fails."""
    print(f"Warning: LLM call failed for '{purpose}'. Using fallback placeholder response.")
    if purpose == "generate_rubric":
        return {
            "id": str(uuid.uuid4()), "idea_id": idea_id or "fallback_idea_id", 
            "type": "Fallback Observational",
            "structure": "Fallback: Setup, Punchline.",
            "key_elements": ["Fallback element 1", "Fallback element 2"],
            "tone": "Fallback Neutral"
        }
    elif purpose == "critique_and_refine_rubric":
        return {
            "id": str(uuid.uuid4()), "idea_id": idea_id or "fallback_idea_id",
            "type": "Fallback Character-based",
            "structure": "Fallback: Dialogue between A and B.",
            "key_elements": ["Fallback character trait", "Fallback witty exchange"],
            "tone": "Fallback Quirky",
            "critique_of_original": "Fallback: Original was okay, this offers a different angle."
        }
    return f"Fallback placeholder response for {purpose}"

def generate_rubric_for_idea(joke_idea: dict, theme: str, num_rubrics: int = 3) -> list:
    """
    Generates multiple detailed 'rubrics' for a given joke idea.
    
    Stage 3: Rubric Generation (Planning the Joke Structure)
    Step 4.1: For each distinct joke idea, create multiple detailed "rubrics" that
    act as specific plans or guidelines for constructing the joke.
    
    Args:
        joke_idea: Dictionary containing joke idea details
        theme: The theme for the joke
        num_rubrics: Number of different rubrics to generate (default: 3)
        
    Returns:
        List of rubric dictionaries
    """
    if not joke_idea or 'concept' not in joke_idea or 'id' not in joke_idea:
        print("Error: Invalid joke_idea provided to generate_rubric_for_idea.")
        return [_fallback_placeholder_response("generate_rubric", "error_idea_id")]

    rubrics = []
    
    for i in range(num_rubrics):
        prompt_content = (
            f"For the joke theme '{theme}' and the specific joke idea: '{joke_idea['concept']}', "
            f"create a detailed rubric for constructing a joke. This rubric will guide the final joke writing. "
            f"The rubric should define: "
            f"1. 'type' (e.g., Observational, Pun, Character-based, Story, Setup-Punchline, etc.), "
            f"2. 'structure' (a brief description of how the joke should be built), "
            f"3. 'key_elements' (a Python list of 2-4 essential components or details to include), "
            f"4. 'tone' (e.g., sarcastic, absurd, dry, witty, dark)."
            f"\n\nThis is rubric {i+1} of {num_rubrics}, so make it distinct from other potential rubrics for the same joke idea."
        )
        expected_format = (
            "A Python dictionary with keys: "
            "'type' (string), 'structure' (string), 'key_elements' (list of strings), and 'tone' (string). "
            "Example: {'type': 'Observational', 'structure': 'Setup, Punchline', 'key_elements': ['Element A', 'Element B'], 'tone': 'Sarcastic'}"
        )
        
        llm_generated_rubric_parts = _openai_llm_call(prompt_content, f"generate_rubric_{i+1}", expected_format)
        
        # Add id and idea_id client-side
        if isinstance(llm_generated_rubric_parts, dict) and all(k in llm_generated_rubric_parts for k in ['type', 'structure', 'key_elements', 'tone']):
            rubric = {
                "id": str(uuid.uuid4()),
                "idea_id": joke_idea['id'],
                **llm_generated_rubric_parts
            }
            print(f"Generated Rubric #{i+1} for Idea ID '{joke_idea['id']}': {rubric}")
            rubrics.append(rubric)
        else:
            print(f"Warning: LLM response for generate_rubric was not in the expected format. Got: {llm_generated_rubric_parts}")
            fallback = _fallback_placeholder_response("generate_rubric", joke_idea['id'])
            fallback["id"] = str(uuid.uuid4())  # Ensure unique ID even for fallbacks
            rubrics.append(fallback)
    
    return rubrics


def critique_and_refine_rubrics(original_rubrics: list, joke_idea: dict, theme: str, num_critiques_per_rubric: int = 2) -> list:
    """
    Critiques existing rubrics and proposes alternatives or refined rubrics to enhance diversity.
    
    Stage 4: Rubric Critique and Diversification (Enhancing Plan Variety)
    Step 5.1: For each rubric, apply a critique step to systematically increase 
    the diversity of joke plans by prompting the LLM to propose alternatives.
    
    Args:
        original_rubrics: List of rubric dictionaries to critique
        joke_idea: Dictionary containing joke idea details
        theme: The theme for the joke
        num_critiques_per_rubric: Number of critiques to generate per original rubric (default: 2)
        
    Returns:
        List of refined/alternative rubric dictionaries
    """
    if not original_rubrics or not joke_idea or 'concept' not in joke_idea or 'id' not in joke_idea:
        print("Error: Invalid inputs to critique_and_refine_rubrics.")
        return [_fallback_placeholder_response("critique_and_refine_rubric", "error_idea_id")]

    refined_rubrics = []
    
    for i, original_rubric in enumerate(original_rubrics):
        print(f"\nCritiquing rubric {i+1}/{len(original_rubrics)} for idea '{joke_idea['concept']}'")
        
        for j in range(num_critiques_per_rubric):
            prompt_content = (
                f"For the joke theme '{theme}' and joke idea '{joke_idea['concept']}', "
                f"the following rubric was initially generated: {original_rubric}. "
                f"This rubric is flawed or could be improved. Please critique it and propose "
                f"an alternative or refined rubric for the same joke idea to enhance creativity or humor. "
                f"Specifically focus on creating a significantly different approach than the original rubric. "
                f"\n\nThis is critique {j+1} of {num_critiques_per_rubric} for this rubric, "
                f"so ensure it differs from other potential critiques. "
                f"The new/refined rubric part should contain: 'type', 'structure', 'key_elements' (a list of strings), and 'tone'. "
                f"Also include a 'critique_of_original' (string) field explaining how the original rubric could be improved."
            )
            expected_format = (
                "A Python dictionary with keys: "
                "'type' (string), 'structure' (string), 'key_elements' (list of strings), 'tone' (string), "
                "and 'critique_of_original' (string). "
                "Example: {'type': 'Character-based', ..., 'critique_of_original': 'The first rubric was too generic...'}"
            )
            
            llm_generated_refined_parts = _openai_llm_call(prompt_content, f"critique_rubric_{i+1}_{j+1}", expected_format)

            if isinstance(llm_generated_refined_parts, dict) and all(k in llm_generated_refined_parts for k in ['type', 'structure', 'key_elements', 'tone', 'critique_of_original']):
                refined_rubric = {
                    "id": str(uuid.uuid4()),
                    "idea_id": joke_idea['id'],
                    "original_rubric_id": original_rubric.get("id", "unknown"),
                    **llm_generated_refined_parts
                }
                print(f"Refined Rubric #{j+1} for Original Rubric ID '{original_rubric.get('id')}': {refined_rubric}")
                refined_rubrics.append(refined_rubric)
            else:
                print(f"Warning: LLM response for critique_and_refine_rubric was not in the expected format. Got: {llm_generated_refined_parts}")
                fallback = _fallback_placeholder_response("critique_and_refine_rubric", joke_idea['id'])
                fallback["id"] = str(uuid.uuid4())  # Ensure unique ID even for fallbacks
                fallback["original_rubric_id"] = original_rubric.get("id", "unknown")
                refined_rubrics.append(fallback)
    
    return refined_rubrics


if __name__ == '__main__':
    from utils.config import initialize_config
    
    if not initialize_config():
        print("Failed to initialize configuration. Please check your .env file.")
        sys.exit(1)

    sample_theme_rubric = "Social Media Addiction"
    sample_joke_idea_rubric = {
        "id": "idea_sm_123", 
        "concept": "A person trying to have a 'digital detox' weekend but their smart home devices keep tempting them back online."
    }
    
    print("\n--- Testing gen_rubrics.py with OpenAI Integration ---")
    if sample_joke_idea_rubric.get("id"):
        # Stage 3: Generate multiple rubrics for the joke idea
        print("\n=== STAGE 3: RUBRIC GENERATION ===")
        initial_rubrics = generate_rubric_for_idea(sample_joke_idea_rubric, sample_theme_rubric, num_rubrics=2)
        
        if initial_rubrics and len(initial_rubrics) > 0:
            print(f"\nSuccessfully generated {len(initial_rubrics)} rubrics for idea: '{sample_joke_idea_rubric['concept']}'")
            
            # Stage 4: Critique and diversify each rubric
            print("\n=== STAGE 4: RUBRIC CRITIQUE AND DIVERSIFICATION ===")
            critiqued_rubrics = critique_and_refine_rubrics(
                initial_rubrics, 
                sample_joke_idea_rubric, 
                sample_theme_rubric, 
                num_critiques_per_rubric=1
            )
            
            if critiqued_rubrics and len(critiqued_rubrics) > 0:
                print(f"\nSuccessfully generated {len(critiqued_rubrics)} critiqued/alternative rubrics")
                
                # Combine all rubrics (original + critiqued)
                all_rubrics = initial_rubrics + critiqued_rubrics
                print(f"\nTotal rubrics available: {len(all_rubrics)}")
            else:
                print("\nFailed to generate critiqued/alternative rubrics or got fallbacks.")
        else:
            print("\nFailed to generate initial rubrics or got fallbacks.")
    else:
        print("\nSample joke idea is missing an ID, cannot test rubric generation.")
        
    print("\n--- End of gen_rubrics.py Test ---")
