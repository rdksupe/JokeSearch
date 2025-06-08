#!/usr/bin/env python3
"""
LossFunk Assignment - Main Entry Point

This script orchestrates the joke generation process using the LLM planning framework:
1. Stage 1: Theme Understanding
2. Stage 2: Idea Generation
3. Stage 3: Rubric Generation (Planning)
4. Stage 4: Rubric Critique and Diversification
5. Stage 5: Joke Generation
6. Stage 6: Joke Evaluation and Selection

Usage:
  python main.py [--theme "Theme Name"] [--ideas 3] [--rubrics 2] [--critiques 1]
  python main.py --run-all

Example:
  python main.py --theme "Smartphones" --ideas 3 --rubrics 2 --critiques 1
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from tabulate import tabulate
from colorama import Fore, Style, init
init(autoreset=True)  # Initialize colorama

try:
    from utils.config import (
        initialize_config, DEFAULT_THEME, DEFAULT_NUM_IDEAS,
        DEFAULT_RUBRICS_PER_IDEA, DEFAULT_CRITIQUES_PER_RUBRIC, DEFAULT_OUTPUT_FILE,
        BASELINE_OUTPUT_FILE
    )
    from gen_ideas import generate_first_order_observations, generate_second_order_observations, formulate_joke_ideas
    from gen_rubrics import generate_rubric_for_idea, critique_and_refine_rubrics
    from gen_jokes import generate_joke_from_rubric
    from baseline_joke_gen import generate_joke
    from joke_judge import JokeJudge
except ImportError as e:
    print(f"Error: Failed to import required modules: {e}")
    print("Make sure you're running from the project root and requirements are installed.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate jokes using a multi-stage LLM planning approach")
    parser.add_argument("--theme", type=str, default=DEFAULT_THEME,
                        help=f"Theme for joke generation (default: {DEFAULT_THEME})")
    parser.add_argument("--ideas", type=int, default=DEFAULT_NUM_IDEAS,
                        help=f"Number of joke ideas to generate (default: {DEFAULT_NUM_IDEAS})")
    parser.add_argument("--rubrics", type=int, dest="rubrics_per_idea", default=DEFAULT_RUBRICS_PER_IDEA,
                        help=f"Number of rubrics per joke idea (default: {DEFAULT_RUBRICS_PER_IDEA})")
    parser.add_argument("--critiques", type=int, dest="critiques_per_rubric", default=DEFAULT_CRITIQUES_PER_RUBRIC,
                        help=f"Number of critiques per rubric (default: {DEFAULT_CRITIQUES_PER_RUBRIC})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f"Output JSON file (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--baseline", type=str, default=BASELINE_OUTPUT_FILE,
                        help=f"Baseline output JSON file (default: {BASELINE_OUTPUT_FILE})")
    parser.add_argument("--run-all", action="store_true", help="Run the entire pipeline including baseline and evaluation")
    parser.add_argument("--no-baseline", action="store_true", help="Skip generating baseline jokes")
    parser.add_argument("--no-judge", action="store_true", help="Skip judging the jokes")
    
    return parser.parse_args()

def generate_multistage_jokes(theme, num_ideas, rubrics_per_idea, critiques_per_rubric, output_file):
    """Run the multi-stage joke generation pipeline"""
    print(f"\n{Fore.CYAN}========== MULTI-STAGE JOKE GENERATION PIPELINE =========={Style.RESET_ALL}")
    print(f"Theme: '{theme}'")
    
    results = {
        "theme": theme,
        "config": {
            "num_ideas": num_ideas,
            "rubrics_per_idea": rubrics_per_idea,
            "critiques_per_rubric": critiques_per_rubric
        }
    }
    
    # STAGE 1: Theme Selection
    print(f"\n{Fore.GREEN}=== STAGE 1: THEME UNDERSTANDING ==={Style.RESET_ALL}")
    
    # STAGE 2: Idea Generation
    print(f"\n{Fore.GREEN}=== STAGE 2: IDEA GENERATION ==={Style.RESET_ALL}")
    
    # Generate first-order observations
    print("Generating first-order observations...")
    first_order_obs = generate_first_order_observations(theme)
    if not first_order_obs:
        print(f"{Fore.RED}Failed to generate first-order observations. Exiting.{Style.RESET_ALL}")
        return None
    
    # Generate second-order observations
    print("Generating second-order observations...")
    second_order_obs = generate_second_order_observations(first_order_obs, theme)
    
    # Formulate joke ideas
    all_observations = first_order_obs + second_order_obs
    print(f"Combined Observations: {len(all_observations)} total")
    print("Formulating joke ideas...")
    joke_ideas = formulate_joke_ideas(all_observations, theme)
    
    if not joke_ideas:
        print(f"{Fore.RED}Failed to generate joke ideas. Exiting.{Style.RESET_ALL}")
        return None
    
    # Limit to requested number of ideas
    if len(joke_ideas) > num_ideas:
        print(f"Limiting to {num_ideas} joke ideas (from {len(joke_ideas)} generated)")
        joke_ideas = joke_ideas[:num_ideas]
    
    print(f"\nFinal Joke Ideas ({len(joke_ideas)}):")
    for i, idea in enumerate(joke_ideas):
        print(f"  {i+1}. {idea['concept']}")
    
    # Store results
    results["observations"] = {
        "first_order": first_order_obs,
        "second_order": second_order_obs
    }
    results["joke_ideas"] = joke_ideas
    
    # Initialize storage for rubrics and jokes
    all_rubrics = []
    all_jokes = []
    
    # Process each joke idea with progress bar
    print(f"\n{Fore.GREEN}=== STAGE 3-5: GENERATING RUBRICS AND JOKES ==={Style.RESET_ALL}")
    
    # Calculate total steps for progress bar
    total_steps = len(joke_ideas) * (1 + rubrics_per_idea * (1 + critiques_per_rubric))
    progress_bar = tqdm(total=total_steps, desc="Processing joke ideas", unit="step")
    
    for joke_idx, joke_idea in enumerate(joke_ideas):
        progress_bar.set_description(f"Processing idea {joke_idx+1}/{len(joke_ideas)}")
        
        # STAGE 3: Generate rubrics
        initial_rubrics = generate_rubric_for_idea(joke_idea, theme, num_rubrics=rubrics_per_idea)
        progress_bar.update(1)  # Update for idea processing
        
        # STAGE 4: Critique and diversify
        critiqued_rubrics = []
        if critiques_per_rubric > 0:
            critiqued_rubrics = critique_and_refine_rubrics(
                initial_rubrics, 
                joke_idea, 
                theme, 
                num_critiques_per_rubric=critiques_per_rubric
            )
        
        # Combine rubrics
        joke_rubrics = initial_rubrics + critiqued_rubrics
        all_rubrics.extend(joke_rubrics)
        
        # STAGE 5: Generate jokes from rubrics
        idea_jokes = []
        for rubric in joke_rubrics:
            joke = generate_joke_from_rubric(rubric, joke_idea, theme)
            if joke and "text" in joke:
                idea_jokes.append(joke)
            progress_bar.update(1)  # Update for each rubric-joke combo
        
        all_jokes.extend(idea_jokes)
    
    progress_bar.close()
    
    # Store results
    results["rubrics"] = all_rubrics
    results["jokes"] = all_jokes
    
    # Summary
    print(f"\n{Fore.CYAN}=== Summary ==={Style.RESET_ALL}")
    print(f"Theme: '{theme}'")
    print(f"Observations: {len(all_observations)} ({len(first_order_obs)} first-order, {len(second_order_obs)} second-order)")
    print(f"Joke Ideas: {len(joke_ideas)}")
    print(f"Total Rubrics: {len(all_rubrics)}")
    print(f"Total Jokes: {len(all_jokes)}")
    
    # Save results
    try:
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path.absolute()}")
    except Exception as e:
        print(f"\n{Fore.RED}Failed to save results: {e}{Style.RESET_ALL}")
    
    return results

def generate_baseline_jokes(theme, num_jokes, output_file):
    """Generate baseline jokes"""
    print(f"\n{Fore.CYAN}========== BASELINE JOKE GENERATION =========={Style.RESET_ALL}")
    print(f"Generating {num_jokes} baseline jokes for theme: '{theme}'")
    
    jokes, raw_response = generate_joke(
        theme, 
        num_jokes=num_jokes, 
        enhanced=True, 
        save_raw=True
    )
    
    if not jokes:
        print(f"{Fore.RED}Failed to generate baseline jokes.{Style.RESET_ALL}")
        return None
        
    # Save jokes
    try:
        output_data = {
            "jokes": jokes,
            "config": {
                "prompt": theme,
                "model": "gemma-3-4b-it-qat",
                "enhanced_prompting": True
            }
        }
        
        if raw_response:
            output_data["raw_response"] = raw_response
            
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nBaseline jokes saved to {Path(output_file).absolute()}")
        
        return output_data
    except Exception as e:
        print(f"{Fore.RED}Error saving baseline jokes: {e}{Style.RESET_ALL}")
        return None

def evaluate_jokes(multistage_file, baseline_file):
    """Evaluate jokes using the Judge"""
    print(f"\n{Fore.CYAN}========== JOKE EVALUATION =========={Style.RESET_ALL}")
    
    judge = JokeJudge()
    
    try:
        multistage_jokes = judge.load_multistage_jokes(multistage_file, filter_fallbacks=True)
        baseline_jokes = judge.load_baseline_jokes(baseline_file, filter_fallbacks=True)
        
        if not multistage_jokes:
            print(f"{Fore.RED}No multi-stage jokes found to evaluate.{Style.RESET_ALL}")
            return None
            
        if not baseline_jokes:
            print(f"{Fore.RED}No baseline jokes found to evaluate.{Style.RESET_ALL}")
            return None
            
        # Limit to sample size for comparison (match the smaller set)
        sample_size = min(len(multistage_jokes), len(baseline_jokes))
        if sample_size > 5:
            sample_size = 5  # Cap at 5 jokes per method for reasonable evaluation time
            
        print(f"Evaluating {sample_size} jokes from each method...")
        
        multistage_sample = multistage_jokes[:sample_size]
        baseline_sample = baseline_jokes[:sample_size]
        all_jokes = multistage_sample + baseline_sample
        
        # Judge jokes with progress bar
        judgments = []
        progress_bar = tqdm(total=len(all_jokes), desc="Evaluating jokes", unit="joke")
        
        for joke in all_jokes:
            judgment = judge.judge_joke(joke)
            judgments.append(judgment)
            progress_bar.update(1)
            
        progress_bar.close()
        
        # Calculate statistics
        parameter_stats, overall_stats = judge.calculate_statistics(judgments)
        
        # Save judgments
        output_file = "joke_judgments.json"
        with open(output_file, 'w') as f:
            json.dump({"judgments": judgments}, f, indent=2)
        print(f"Judgments saved to {Path(output_file).absolute()}")
        
        # Print comparison
        judge.print_comparison(parameter_stats, overall_stats)
        
        return {
            "judgments": judgments,
            "parameter_stats": parameter_stats,
            "overall_stats": overall_stats
        }
        
    except Exception as e:
        print(f"{Fore.RED}Error during joke evaluation: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return None

def display_top_jokes(judgment_results, multistage_file, baseline_file):
    """Display the top jokes based on evaluation results"""
    if not judgment_results or "judgments" not in judgment_results:
        print(f"{Fore.RED}No judgment results available.{Style.RESET_ALL}")
        return
        
    print(f"\n{Fore.CYAN}========== TOP JOKES =========={Style.RESET_ALL}")
    
    # Load jokes from files
    try:
        with open(multistage_file, 'r') as f:
            multistage_data = json.load(f)
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
            
        # Get multi-stage jokes with their IDs
        multistage_jokes = {joke.get("id", "unknown"): joke for joke in multistage_data.get("jokes", [])}
        
        # Get baseline jokes with their IDs
        baseline_jokes = {joke.get("id", "unknown"): joke for joke in baseline_data.get("jokes", [])}
        
        # Combine all jokes
        all_jokes = {**multistage_jokes, **baseline_jokes}
        
        # Get judgments with scores
        judgments = sorted(
            judgment_results["judgments"],
            key=lambda j: j.get("overall", 0),
            reverse=True
        )
        
        # Display top jokes
        top_count = min(5, len(judgments))
        
        print(f"\n{Fore.YELLOW}Top {top_count} Jokes:{Style.RESET_ALL}")
        
        table_data = []
        for i, judgment in enumerate(judgments[:top_count]):
            joke_id = judgment.get("joke_id", "unknown")
            joke = all_jokes.get(joke_id, {})
            method = judgment.get("method", "unknown")
            score = judgment.get("overall", 0)
            
            joke_text = joke.get("text", "Text not found")
            if len(joke_text) > 80:
                joke_text = joke_text[:77] + "..."
                
            # Add color based on method
            method_display = f"{Fore.BLUE}{method}{Style.RESET_ALL}" if method == "multi-stage" else f"{Fore.GREEN}{method}{Style.RESET_ALL}"
            
            table_data.append([
                i + 1,
                method_display,
                score,
                joke_text
            ])
        
        # Print table
        print(tabulate(
            table_data,
            headers=["Rank", "Method", "Score", "Joke"],
            tablefmt="grid"
        ))
        
    except Exception as e:
        print(f"{Fore.RED}Error displaying top jokes: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution function"""
    # Check configuration
    if not initialize_config():
        print(f"{Fore.RED}Failed to initialize configuration. Please check your .env file.{Style.RESET_ALL}")
        return
    
    # Parse arguments
    args = parse_args()
    
    # Configuration
    theme = args.theme
    num_ideas = max(1, min(10, args.ideas))  # Limit between 1-10
    rubrics_per_idea = max(1, min(5, args.rubrics_per_idea))  # Limit between 1-5
    critiques_per_rubric = max(0, min(3, args.critiques_per_rubric))  # Limit between 0-3
    output_file = args.output
    baseline_file = args.baseline
    
    print(f"\n{Fore.MAGENTA}========== JOKE GENERATION PIPELINE =========={Style.RESET_ALL}")
    print(f"Configuration:")
    print(f"- Theme: '{theme}'")
    print(f"- Target joke ideas: {num_ideas}")
    print(f"- Rubrics per idea: {rubrics_per_idea}")
    print(f"- Critiques per rubric: {critiques_per_rubric}")
    print(f"- Total expected jokes: {num_ideas * rubrics_per_idea * (1 + critiques_per_rubric)}")
    
    # Generate multi-stage jokes
    multistage_results = generate_multistage_jokes(
        theme, 
        num_ideas, 
        rubrics_per_idea, 
        critiques_per_rubric, 
        output_file
    )
    
    # Generate baseline jokes if not skipped
    baseline_results = None
    if not args.no_baseline:
        # For baseline, generate a similar number of jokes as the multi-stage approach
        baseline_num_jokes = min(10, num_ideas * rubrics_per_idea)
        baseline_results = generate_baseline_jokes(theme, baseline_num_jokes, baseline_file)
    else:
        print(f"\n{Fore.YELLOW}Skipping baseline joke generation.{Style.RESET_ALL}")
    
    # Evaluate jokes if not skipped
    judgment_results = None
    if not args.no_judge and (args.run_all or multistage_results and baseline_results):
        judgment_results = evaluate_jokes(output_file, baseline_file)
        
        # Display top jokes
        if judgment_results:
            display_top_jokes(judgment_results, output_file, baseline_file)
    else:
        print(f"\n{Fore.YELLOW}Skipping joke evaluation.{Style.RESET_ALL}")
    
    print(f"\n{Fore.MAGENTA}========== PIPELINE COMPLETE =========={Style.RESET_ALL}")

if __name__ == "__main__":
    main()
