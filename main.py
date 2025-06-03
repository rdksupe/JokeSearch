#!/usr/bin/env python3
"""
LossFunk Assignment - Main Entry Point

This script orchestrates the joke generation process using the LLM planning framework:
1. Stage 1: Theme Understanding
2. Stage 2: Idea Generation
3. Stage 3: Rubric Generation (Planning)
4. Stage 4: Rubric Critique and Diversification
5. Stage 5: Joke Generation
6. Stage 6: Joke Refinement and Selection

Usage:
  python main.py [--theme "Theme Name"] [--ideas 5] [--rubrics-per-idea 2] [--critiques-per-rubric 1]

Example:
  python main.py --theme "Smartphones" --ideas 3 --rubrics-per-idea 3 --critiques-per-rubric 2
"""

import os
import sys
import json
import argparse
from pathlib import Path

try:
    from gen_ideas import generate_first_order_observations, generate_second_order_observations, formulate_joke_ideas
    from gen_rubrics import generate_rubric_for_idea, critique_and_refine_rubrics
    from gen_jokes import generate_joke_from_rubric
except ImportError:
    print("Error: Required modules not found. Make sure you're running from the project root.")
    sys.exit(1)

# Default configuration
DEFAULT_CONFIG = {
    "theme": "Penguins",
    "num_ideas": 3,               # Number of joke ideas to generate
    "rubrics_per_idea": 2,        # Number of rubrics per joke idea
    "critiques_per_rubric": 1,    # Number of critiques/alternatives per rubric
    "output_file": "results.json",
    "sample_display_limit": 3,    # Number of jokes to display in the summary
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate jokes using a multi-stage LLM planning approach")
    parser.add_argument("--theme", type=str, default=DEFAULT_CONFIG["theme"],
                        help=f"Theme for joke generation (default: {DEFAULT_CONFIG['theme']})")
    parser.add_argument("--ideas", type=int, default=DEFAULT_CONFIG["num_ideas"],
                        help=f"Number of joke ideas to generate (default: {DEFAULT_CONFIG['num_ideas']})")
    parser.add_argument("--rubrics-per-idea", type=int, default=DEFAULT_CONFIG["rubrics_per_idea"],
                        help=f"Number of rubrics per joke idea (default: {DEFAULT_CONFIG['rubrics_per_idea']})")
    parser.add_argument("--critiques-per-rubric", type=int, default=DEFAULT_CONFIG["critiques_per_rubric"],
                        help=f"Number of critiques per rubric (default: {DEFAULT_CONFIG['critiques_per_rubric']})")
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG["output_file"],
                        help=f"Output JSON file (default: {DEFAULT_CONFIG['output_file']})")
    parser.add_argument("--sample", type=int, default=DEFAULT_CONFIG["sample_display_limit"],
                        help=f"Number of sample jokes to display (default: {DEFAULT_CONFIG['sample_display_limit']})")
    
    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_args()

    # Configuration
    theme = args.theme
    num_ideas = max(1, min(10, args.ideas))  # Limit between 1-10
    rubrics_per_idea = max(1, min(5, args.rubrics_per_idea))  # Limit between 1-5
    critiques_per_rubric = max(0, min(3, args.critiques_per_rubric))  # Limit between 0-3
    output_file = args.output
    sample_display_limit = args.sample
    
    print("LossFunk Assignment - Joke Generation Pipeline\n")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Continuing with limited functionality...\n")
    
    # Print model and configuration information
    print("Using model: gemma-3-4b-it-qat")
    print("API endpoint: http://localhost:1234/v1/")
    print("\nConfiguration:")
    print(f"- Theme: '{theme}'")
    print(f"- Target joke ideas: {num_ideas}")
    print(f"- Rubrics per idea: {rubrics_per_idea}")
    print(f"- Critiques per rubric: {critiques_per_rubric}")
    print(f"- Total expected jokes: {num_ideas * rubrics_per_idea * (1 + critiques_per_rubric)}")
    
    # Stage 1: Theme Selection (using provided theme)
    print(f"\nSelected Theme: '{theme}'\n")
    
    # Stage 2: Idea Generation
    print("=== STAGE 2: IDEA GENERATION ===")
    
    # Stage 2.1: Generate first-order observations
    print("\nGenerating first-order observations...")
    first_order_obs = generate_first_order_observations(theme)
    if not first_order_obs:
        print("Failed to generate first-order observations. Exiting.")
        return
    
    # Stage 2.2: Generate second-order observations
    print("\nGenerating second-order observations...")
    second_order_obs = generate_second_order_observations(first_order_obs, theme)
    
    # Stage 2.3: Formulate joke ideas from observations
    all_observations = first_order_obs + second_order_obs
    print(f"\nCombined Observations: {len(all_observations)} total")
    print("\nFormulating joke ideas...")
    joke_ideas = formulate_joke_ideas(all_observations, theme)
    
    if not joke_ideas:
        print("Failed to generate joke ideas. Exiting.")
        return
    
    # Limit to requested number of ideas
    if len(joke_ideas) > num_ideas:
        print(f"Limiting to {num_ideas} joke ideas (from {len(joke_ideas)} generated)")
        joke_ideas = joke_ideas[:num_ideas]
    
    print(f"\nFinal Joke Ideas ({len(joke_ideas)}):")
    for i, idea in enumerate(joke_ideas):
        print(f"  {i+1}. {idea['concept']} (ID: {idea['id'][:8]}...)")
    
    # Store results for all stages
    all_rubrics = []
    all_jokes = []
    
    # Process each joke idea through stages 3, 4, and 5
    for joke_idx, joke_idea in enumerate(joke_ideas):
        print(f"\n=== Processing Joke Idea {joke_idx+1}/{len(joke_ideas)}: {joke_idea['concept']} ===")
        
        # Stage 3: Generate multiple rubrics for each joke idea
        print(f"\nStage 3: Generating {rubrics_per_idea} Rubrics...")
        initial_rubrics = generate_rubric_for_idea(joke_idea, theme, num_rubrics=rubrics_per_idea)
        print(f"Generated {len(initial_rubrics)} initial rubrics")
        
        # Stage 4: Critique and diversify the rubrics
        print(f"\nStage 4: Critiquing and Diversifying Rubrics ({critiques_per_rubric} critiques per rubric)...")
        critiqued_rubrics = []
        if critiques_per_rubric > 0:
            critiqued_rubrics = critique_and_refine_rubrics(
                initial_rubrics, 
                joke_idea, 
                theme, 
                num_critiques_per_rubric=critiques_per_rubric
            )
            print(f"Generated {len(critiqued_rubrics)} critiqued/alternative rubrics")
        
        # Combine all rubrics for this joke idea
        joke_rubrics = initial_rubrics + critiqued_rubrics
        all_rubrics.extend(joke_rubrics)
        
        print(f"\nTotal rubrics for this joke idea: {len(joke_rubrics)}")
        
        # Stage 5: Generate jokes based on each rubric
        print("\nStage 5: Generating Jokes...")
        idea_jokes = []
        for rubric_idx, rubric in enumerate(joke_rubrics):
            print(f"\nGenerating joke {rubric_idx+1}/{len(joke_rubrics)} for rubric type: {rubric.get('type', 'Unknown')}, tone: {rubric.get('tone', 'Unknown')}")
            joke = generate_joke_from_rubric(rubric, joke_idea, theme)
            if joke and "text" in joke:
                idea_jokes.append(joke)
                print(f"  ✓ Generated joke: {joke['text'][:100]}...")
            else:
                print(f"  ✗ Failed to generate joke for rubric ID: {rubric.get('id', 'unknown')}")
        
        all_jokes.extend(idea_jokes)
        print(f"Generated {len(idea_jokes)} jokes for idea: '{joke_idea['concept']}'")
    
    # Summary of results
    print(f"\n=== Summary ===")
    print(f"Theme: '{theme}'")
    print(f"Observations: {len(all_observations)} ({len(first_order_obs)} first-order, {len(second_order_obs)} second-order)")
    print(f"Joke Ideas: {len(joke_ideas)}")
    print(f"Total Rubrics: {len(all_rubrics)}")
    print(f"Total Jokes: {len(all_jokes)}")
    
    # Display some sample jokes
    if all_jokes:
        print(f"\n=== Sample of Generated Jokes ({min(sample_display_limit, len(all_jokes))}) ===")
        display_jokes = all_jokes[:sample_display_limit]
        for i, joke in enumerate(display_jokes):
            print(f"\nJoke {i+1}:")
            print(f"  {joke['text']}")
            print(f"  (Based on rubric type: {joke.get('metadata', {}).get('joke_type', 'Unknown')}, " + 
                  f"tone: {joke.get('metadata', {}).get('tone', 'Unknown')})")
    
    # Save results to a file
    try:
        results = {
            "theme": theme,
            "observations": {
                "first_order": first_order_obs,
                "second_order": second_order_obs
            },
            "joke_ideas": joke_ideas,
            "rubrics": all_rubrics,
            "jokes": all_jokes,
            "config": {
                "num_ideas": num_ideas,
                "rubrics_per_idea": rubrics_per_idea,
                "critiques_per_rubric": critiques_per_rubric
            }
        }
        
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path.absolute()}")
    except Exception as e:
        print(f"\nFailed to save results: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
