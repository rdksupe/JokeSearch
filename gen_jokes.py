# gen_jokes.py

import uuid
import os
import json
import re
from openai import OpenAI
from utils.config import get_api_base_url, get_openai_key, DEFAULT_MODEL

# --- OpenAI API Configuration ---
# Ensure your OpenAI API key is set as an environment variable: OPENAI_API_KEY

def _extract_json_from_text(text):
    """Extract JSON from text that might contain other content."""
    if not text:
        return "{}"
        
    # Try to find JSON within markdown code blocks
    if "```" in text:
        pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
    
    # Try to find JSON between curly braces
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if (start_idx != -1 and end_idx != -1):
            return text[start_idx:end_idx+1].strip()
    except:
        pass
    
    # Return the original text as a last resort
    return text.strip()

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
                {"role": "system", "content": f"You are a helpful assistant. Your response should be a JSON string that can be parsed into the following Python structure: {expected_format_description}. Do not include any explanatory text outside of the JSON string itself."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
        )
        
        raw_response_content = response.choices[0].message.content
        print(f"Raw LLM Response (first 200 chars): {raw_response_content[:200]}...")

        # Extract and clean JSON from response
        json_str = _extract_json_from_text(raw_response_content)
        
        try:
            parsed_response = json.loads(json_str)
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error ({purpose}): Failed to parse extracted JSON. Error: {e}")
            print(f"Extracted JSON string (first 200 chars): {json_str[:200]}...")
            
            # If we're generating a joke and parsing fails, try to extract text directly
            if purpose.startswith("generate_joke"):
                # Try to extract joke directly from text
                text_match = re.search(r"text[\"']?\s*:\s*[\"']([^\"']+)[\"']", json_str)
                explanation_match = re.search(r"explanation[\"']?\s*:\s*[\"']([^\"']+)[\"']", json_str)
                
                if text_match:
                    return {
                        "text": text_match.group(1),
                        "explanation": explanation_match.group(1) if explanation_match else "Explanation not available"
                    }
            
            raise

    except openai.APIError as e:
        print(f"OpenAI API Error ({purpose}): {e}")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error ({purpose}): Failed to parse LLM response. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during LLM call ({purpose}): {e}")
        import traceback
        traceback.print_exc()
    
    return _fallback_placeholder_response(purpose)


def _fallback_placeholder_response(purpose: str) -> any:
    """Provides a fallback response if the LLM call fails."""
    print(f"Warning: LLM call failed for '{purpose}'. Using fallback placeholder response.")
    if purpose.startswith("generate_joke"):
        return {
            "id": str(uuid.uuid4()),
            "text": f"Fallback joke about {purpose.split('_')[-1] if '_' in purpose else 'something'}: Why did the AI cross the road? Because it was trying to get to the other data center.",
            "explanation": "This is a fallback joke due to API failure."
        }
    return f"Fallback placeholder response for {purpose}"


def generate_joke_from_rubric(rubric: dict, joke_idea: dict, theme: str) -> dict:
    """
    Stage 5: Joke Generation (Implementing the Plan)
    Step 6.1: Generate a final joke by strongly conditioning on both the specific joke idea and the detailed rubric.
    
    Args:
        rubric: Dictionary containing the joke rubric details
        joke_idea: Dictionary containing the original joke idea
        theme: The theme for the joke
        
    Returns:
        Dictionary containing the generated joke and metadata
    """
    if not rubric or not joke_idea:
        print("Error: Invalid inputs to generate_joke_from_rubric.")
        return _fallback_placeholder_response("generate_joke")
    
    # Extract rubric elements for better prompt construction
    joke_type = rubric.get("type", "Unknown")
    joke_structure = rubric.get("structure", "Setup, Punchline")
    key_elements = rubric.get("key_elements", ["humor", "surprise"])
    tone = rubric.get("tone", "Neutral")
    
    # Make sure key_elements is actually a list
    if not isinstance(key_elements, list):
        key_elements = [str(key_elements)]
    
    prompt_content = (
        f"You are a professional comedy writer. Create a joke based on the following specifications:\n\n"
        f"Theme: '{theme}'\n\n"
        f"Joke Idea: '{joke_idea['concept']}'\n\n"
        f"Joke Rubric:\n"
        f"- Type: {joke_type}\n"
        f"- Structure: {joke_structure}\n"
        f"- Key Elements to Include: {key_elements}\n"
        f"- Tone: {tone}\n\n"
        f"Write a complete joke that strictly follows this rubric. Then provide a brief explanation of how your joke implements "
        f"the rubric and the original idea. Keep the joke concise, entertaining, and aligned with the specified structure and tone.\n\n"
        f"Format your response as a JSON object with 'text' and 'explanation' fields."
    )
    
    expected_format = (
        "A Python dictionary with keys: "
        "'text' (string containing the joke), "
        "'explanation' (string explaining how the joke implements the rubric and original idea)"
    )
    
    try:
        llm_generated_joke = _openai_llm_call(prompt_content, f"generate_joke_{rubric.get('id', '')[:8]}", expected_format)
        
        if isinstance(llm_generated_joke, dict) and "text" in llm_generated_joke:
            joke = {
                "id": str(uuid.uuid4()),
                "theme": theme,
                "idea_id": joke_idea.get("id", "unknown"),
                "rubric_id": rubric.get("id", "unknown"),
                "text": llm_generated_joke["text"],
                "explanation": llm_generated_joke.get("explanation", "No explanation provided"),
                "metadata": {
                    "joke_type": joke_type,
                    "tone": tone,
                    "structure": joke_structure
                }
            }
            print(f"\nGenerated Joke for Idea: '{joke_idea.get('concept', '')}': {joke['text']}")
            return joke
        else:
            print(f"Warning: LLM response for joke generation was not in the expected format. Got: {llm_generated_joke}")
            
            # Try to handle simple text response
            if isinstance(llm_generated_joke, str) and len(llm_generated_joke) > 10:
                return {
                    "id": str(uuid.uuid4()),
                    "theme": theme,
                    "idea_id": joke_idea.get("id", "unknown"),
                    "rubric_id": rubric.get("id", "unknown"),
                    "text": llm_generated_joke,
                    "explanation": "No structured explanation available",
                    "metadata": {
                        "joke_type": joke_type,
                        "tone": tone,
                        "structure": joke_structure
                    }
                }
            
            fallback = _fallback_placeholder_response("generate_joke")
            fallback["id"] = str(uuid.uuid4())
            fallback["idea_id"] = joke_idea.get("id", "unknown")
            fallback["rubric_id"] = rubric.get("id", "unknown")
            return fallback
            
    except Exception as e:
        print(f"Error in generate_joke_from_rubric: {e}")
        import traceback
        traceback.print_exc()
        
        fallback = _fallback_placeholder_response("generate_joke")
        fallback["id"] = str(uuid.uuid4())
        fallback["idea_id"] = joke_idea.get("id", "unknown")
        fallback["rubric_id"] = rubric.get("id", "unknown")
        return fallback


if __name__ == '__main__':
    from utils.config import initialize_config
    
    if not initialize_config():
        print("Failed to initialize configuration. Please check your .env file.")
        sys.exit(1)
    
    sample_theme = "Technology Frustrations"
    sample_joke_idea = {
        "id": "idea123",
        "concept": "People getting more attached to their devices than to other humans"
    }
    sample_rubric = {
        "id": "rubric456",
        "idea_id": "idea123",
        "type": "Observational",
        "structure": "Setup (relatable situation), Punchline (ironic twist)",
        "key_elements": ["smartphone addiction", "social awkwardness", "irony"],
        "tone": "Satirical"
    }
    
    print("\n--- Testing gen_jokes.py with OpenAI Integration ---")
    joke = generate_joke_from_rubric(sample_rubric, sample_joke_idea, sample_theme)
    
    if joke and "text" in joke:
        print("\nJoke Generation Test Successful!")
        print(f"Joke: {joke['text']}")
        print(f"Explanation: {joke['explanation']}")
    else:
        print("\nJoke Generation Test Failed!")
    
    print("\n--- End of gen_jokes.py Test ---")
