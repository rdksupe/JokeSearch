#!/usr/bin/env python3
"""
Baseline Joke Generator

This script provides a simple, direct approach to joke generation using OpenAI API,
without the multi-stage planning framework used in the main application.
It can be used as a baseline for comparison with the more sophisticated approach.

Usage:
  python baseline_joke_gen.py "Your joke prompt here"
  python baseline_joke_gen.py --interactive
  python baseline_joke_gen.py --enhanced "Your prompt here" --num-jokes 3
  python baseline_joke_gen.py --enhanced "Your prompt here" --save-raw --output jokes.json
"""

import sys
import uuid
import json
import argparse
import os
import re
from openai import OpenAI

def _extract_json_from_text(text):
    """
    Extract JSON from text that might contain other content or nested code blocks.
    Handles cases with nested JSON code blocks or malformed responses.
    """
    if not text:
        return "{}"
    
    # First, try to find the most complete JSON object in the text
    # This pattern looks for either a JSON code block or a raw JSON object
    json_candidates = []
    
    # Look for JSON in markdown code blocks (handling nested code blocks)
    json_block_pattern = r"```(?:json)?\s*([\s\S]+?)```"
    json_blocks = re.findall(json_block_pattern, text)
    for block in json_blocks:
        # Check if this block contains a valid JSON object
        try:
            # Find opening { and closing } that form a complete object
            start_idx = block.find('{')
            if start_idx != -1:
                # Count brackets to find matching closing }
                brace_count = 0
                for i in range(start_idx, len(block)):
                    if block[i] == '{':
                        brace_count += 1
                    elif block[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            json_candidates.append(block[start_idx:i+1])
        except Exception:
            pass
    
    # If no valid JSON objects found in code blocks, try to extract JSON directly
    if not json_candidates:
        try:
            start_idx = text.find('{')
            if start_idx != -1:
                # Count brackets to find matching closing }
                brace_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            json_candidates.append(text[start_idx:i+1])
        except Exception:
            pass
    
    # Try each candidate until we find one that parses correctly
    for candidate in json_candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    
    # If we get here, no valid JSON was found - return the original text
    return text.strip()

def _parse_jokes_from_response(raw_content):
    """
    Parse jokes from the API response, handling various formats and complex JSON structures.
    """
    jokes_data = []
    
    # First try: direct JSON parsing with improved extraction
    try:
        # Extract JSON content from potentially complex response
        json_str = _extract_json_from_text(raw_content)
        data = json.loads(json_str)
        
        # Handle different response structures
        if isinstance(data, dict):
            if "jokes" in data and isinstance(data["jokes"], list):
                jokes_data = data["jokes"]
            elif "text" in data:
                # Single joke directly in the response
                jokes_data = [data]
        elif isinstance(data, list):
            jokes_data = data
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed, trying alternative extraction... Error: {e}")
    
    # If no jokes found yet, try regex-based approaches
    if not jokes_data:
        # Look for standard joke objects with text fields
        try:
            json_text_pattern = r"\"text\":\s*\"([^\"]+)\""
            text_matches = re.findall(json_text_pattern, raw_content)
            
            json_type_pattern = r"\"type\":\s*\"([^\"]+)\""
            type_matches = re.findall(json_type_pattern, raw_content)
            
            # If we found text fields, create joke objects
            if text_matches:
                for i, text in enumerate(text_matches):
                    joke_type = type_matches[i] if i < len(type_matches) else "General"
                    jokes_data.append({
                        "text": text,
                        "type": joke_type
                    })
        except Exception as e:
            print(f"JSON field extraction failed: {e}")
    
    # If still no jokes, try to extract numbered jokes
    if not jokes_data:
        try:
            # Look for numbered jokes
            joke_pattern = r"(?:^|\n)(?:Joke\s*\d+:|#\d+:?)\s*(.*?)(?=(?:\n(?:Joke\s*\d+:|#\d+:))|$)"
            matches = re.findall(joke_pattern, raw_content, re.DOTALL)
            
            if matches:
                for joke_text in matches:
                    jokes_data.append({"text": joke_text.strip(), "type": "General"})
        except Exception as e:
            print(f"Numbered joke extraction failed: {e}")
    
    # Last resort: treat the whole response as one joke
    if not jokes_data and raw_content.strip():
        # Remove any markdown or code block markers
        cleaned_content = re.sub(r'```(?:json)?\s*', '', raw_content)
        cleaned_content = re.sub(r'```\s*', '', cleaned_content)
        
        jokes_data = [{"text": cleaned_content.strip(), "type": "General"}]
    
    return jokes_data

def generate_joke(prompt: str, num_jokes: int = 1, model: str = "gemma-3-4b-it-qat", enhanced: bool = False, save_raw: bool = False) -> tuple:
    """
    Generate jokes directly from a prompt without using the multi-stage framework.
    
    Args:
        prompt: The joke prompt or theme
        num_jokes: Number of jokes to generate
        model: The OpenAI model to use
        enhanced: Whether to use an enhanced prompt for fairer comparison
        save_raw: Whether to return the raw LLM response
        
    Returns:
        tuple: (list of joke dictionaries, raw response if save_raw=True else None)
    """
    if not prompt:
        print("Error: Empty prompt provided")
        return [], None
    
    print(f"\nGenerating {num_jokes} joke{'' if num_jokes == 1 else 's'} for prompt: '{prompt}'")
    print(f"Using {'enhanced' if enhanced else 'basic'} prompting")
    
    try:
        # Use localhost endpoint for OpenAI API
        client = OpenAI(base_url="http://localhost:1234/v1/")
        
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            return [], None
        
        if enhanced:
            # Enhanced prompt that includes some elements from the multi-stage approach
            # but still in a single step, now with a clear JSON example
            system_prompt = (
                "You are a professional comedy writer with expertise in joke construction. "
                "Create original, well-crafted jokes based on the given theme. "
                "Consider different joke types (observational, character-based, absurdist, etc.) "
                "and tones (witty, sarcastic, lighthearted, etc.) to create diverse jokes. "
                "Format your output as well-structured JSON."
            )
            
            user_prompt = (
                f"Theme: {prompt}\n\n"
                f"Please generate {num_jokes} distinct, high-quality joke(s) about this theme. For each joke:\n"
                f"1. Think about a specific angle or observation related to the theme\n"
                f"2. Consider what joke structure would work best (setup-punchline, misdirection, character-based, etc.)\n"
                f"3. Include key elements that make the joke work (irony, absurdity, wordplay, etc.)\n"
                f"4. Use an appropriate tone for maximum comedic effect\n"
                f"5. Write a complete, polished joke\n\n"
                f"Format your response as a JSON object using EXACTLY this structure:\n\n"
                f"{{\n"
                f"  \"jokes\": [\n"
                f"    {{\n"
                f"      \"text\": \"The actual joke goes here with setup and punchline.\",\n"
                f"      \"type\": \"The style of joke (Observational, Wordplay, etc.)\",\n"
                f"      \"approach\": \"Brief explanation of the comedic technique used\",\n"
                f"      \"tone\": \"The emotional tone (Sarcastic, Absurd, etc.)\"\n"
                f"    }},\n"
                f"    {{\n"
                f"      \"text\": \"A second joke if multiple were requested.\",\n"
                f"      \"type\": \"Another joke style\",\n"
                f"      \"approach\": \"Another comedic technique\",\n"
                f"      \"tone\": \"Another tone\"\n"
                f"    }}\n"
                f"  ]\n"
                f"}}\n"
                f"```\n\n"
                f"Make sure your response contains only this JSON object and nothing else."
            )
        else:
            # Basic prompt - simpler but still with clear JSON example
            system_prompt = (
                "You are a professional comedy writer. Create funny, original jokes based on the given prompt. "
                "Each joke should be concise, clever, and entertaining. "
                "Return your output as JSON."
            )
            
            user_prompt = (
                f"Write {num_jokes} funny joke(s) about: {prompt}.\n\n"
                f"Format your response as a JSON object using EXACTLY this structure:\n\n"
                f"{{\n"
                f"  \"jokes\": [\n"
                f"    {{\n"
                f"      \"text\": \"The full joke goes here.\",\n"
                f"      \"type\": \"The type of joke (Pun, One-liner, etc.)\"\n"
                f"    }},\n"
                f"    {{\n"
                f"      \"text\": \"Another joke if more than one is requested.\",\n"
                f"      \"type\": \"Another joke type\"\n"
                f"    }}\n"
                f"  ]\n"
                f"}}\n"
                f"```\n\n"
                f"Return ONLY the JSON. Include exactly {num_jokes} joke(s) in your response."
            )
        
        # Call API with appropriate prompt
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
        )
        
        raw_response_content = response.choices[0].message.content
        print(f"Raw LLM Response (first 200 chars): {raw_response_content[:200]}...")
        
        # Parse jokes from the response
        parsed_jokes = _parse_jokes_from_response(raw_response_content)
        
        jokes = []
        for i, joke_data in enumerate(parsed_jokes[:num_jokes]):  # Limit to requested number
            if isinstance(joke_data, dict) and "text" in joke_data:
                joke = {
                    "id": str(uuid.uuid4()),
                    "prompt": prompt,
                    "text": joke_data["text"],
                    "type": joke_data.get("type", "General"),
                    "tone": joke_data.get("tone", "Standard"),
                    "approach": joke_data.get("approach", "Direct humor"),
                    "model": model,
                    "method": "enhanced_baseline" if enhanced else "basic_baseline"
                }
                jokes.append(joke)
                print(f"\nJoke {i+1}:\n{joke['text']}")
                print(f"Type: {joke['type']}, Tone: {joke['tone']}")
        
        return jokes, raw_response_content if save_raw else None
    
    except Exception as e:
        print(f"Error generating jokes: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], None


def interactive_mode(enhanced: bool = False, save_raw: bool = False, output_file: str = None):
    """Run the joke generator in interactive mode"""
    print("===== Baseline Joke Generator (Interactive Mode) =====")
    print(f"Using {'enhanced' if enhanced else 'basic'} prompting")
    print(f"{'Saving' if save_raw else 'Not saving'} raw LLM responses")
    print("Type 'exit' or 'quit' to end the session\n")
    
    all_results = {"sessions": []}
    
    while True:
        prompt = input("\nEnter a joke prompt/theme: ")
        if prompt.lower() in ['exit', 'quit']:
            print("Exiting interactive mode.")
            break
            
        try:
            num_jokes = int(input("How many jokes would you like? [1-5]: "))
            num_jokes = max(1, min(5, num_jokes))  # Limit between 1-5
        except ValueError:
            num_jokes = 1
            print("Invalid number, generating 1 joke.")
        
        jokes, raw_response = generate_joke(prompt, num_jokes, enhanced=enhanced, save_raw=save_raw)
        
        # Add to results if we're keeping track
        if output_file:
            session_result = {
                "prompt": prompt,
                "jokes": jokes
            }
            if save_raw and raw_response:
                session_result["raw_response"] = raw_response
                
            all_results["sessions"].append(session_result)
        
        print("\n" + "-"*50)
    
    # Save results if an output file was specified
    if output_file and all_results["sessions"]:
        try:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSession results saved to {output_file}")
        except Exception as e:
            print(f"Error saving session results: {e}")


def main():
    """Main function to handle CLI arguments"""
    parser = argparse.ArgumentParser(description="Generate jokes directly from prompts")
    parser.add_argument("prompt", nargs="?", default=None, help="The joke prompt or theme")
    parser.add_argument("-n", "--num-jokes", type=int, default=1, help="Number of jokes to generate")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-e", "--enhanced", action="store_true", help="Use enhanced prompting for fairer comparison")
    parser.add_argument("-m", "--model", default="gemma-3-4b-it-qat", help="Model to use")
    parser.add_argument("-o", "--output", help="Output JSON file to save jokes")
    parser.add_argument("-r", "--save-raw", action="store_true", help="Save raw LLM responses in output file")
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
    
    if args.interactive:
        interactive_mode(enhanced=args.enhanced, save_raw=args.save_raw, output_file=args.output)
    elif args.prompt:
        jokes, raw_response = generate_joke(args.prompt, args.num_jokes, args.model, 
                                           enhanced=args.enhanced, save_raw=args.save_raw)
        
        if jokes and args.output:
            try:
                output_data = {
                    "jokes": jokes, 
                    "config": {
                        "prompt": args.prompt,
                        "model": args.model,
                        "enhanced_prompting": args.enhanced
                    }
                }
                
                # Add raw response if requested
                if args.save_raw and raw_response:
                    output_data["raw_response"] = raw_response
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nJokes saved to {args.output}")
            except Exception as e:
                print(f"Error saving jokes: {e}")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  Basic prompt:    python baseline_joke_gen.py \"Smartphones\"")
        print("  Enhanced prompt: python baseline_joke_gen.py \"Smartphones\" --enhanced")
        print("  Save raw:        python baseline_joke_gen.py \"Smartphones\" --save-raw -o output.json")


if __name__ == "__main__":
    main()
