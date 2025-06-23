import os
import json
import re
import openai
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Set up OpenAI API key
# Make sure to set your API key as an environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

def load_problems(file_path):
    """Load problems from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_answer(completion):
    """Extract numerical answer from a completion."""
    # Look for patterns like "The answer is 42" or "= 42" or "answer: 42"
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
        r"(?:=|:)\s*(-?\d+(?:\.\d+)?)\s*$",
        r"(?:therefore|so|thus|hence),?\s+(?:the\s+)?(?:answer|result)\s+(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
        r"(?:therefore|so|thus|hence),?\s+(?:we\s+get|we\s+have)\s+(?:the\s+)?(?:answer|result)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)",
        r"(?:profit|total|sum|difference|product|quotient|result)\s+(?:is|=|:)\s*(-?\d+(?:\.\d+)?)"
    ]
    
    # Convert to lowercase and clean up the text
    text = completion.lower().strip()
    
    # Try the specific patterns first
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Get the last group that matched (some patterns have multiple capture groups)
            for i in range(1, len(match.groups()) + 1):
                if match.group(i):
                    return float(match.group(i))
    
    # Look for a line that starts with "Answer:" or similar
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(("answer:", "the answer:", "final answer:")):
            numbers = re.findall(r"(-?\d+(?:\.\d+)?)", line)
            if numbers:
                return float(numbers[0])
    
    # Look for the last number in the text as a fallback
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if numbers:
        return float(numbers[-1])
    
    return None

def get_single_completion(problem, temperature=0):
    """Get a single completion for a problem."""
    prompt = f"{problem}\n\nLet's think step-by-step."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=500
    )
    
    return response.choices[0].message['content']

def get_multiple_completions(problem, n=10, temperature=1.1):
    """Get multiple completions for a problem."""
    prompt = f"{problem}\n\nLet's think step-by-step."
    
    completions = []
    for _ in range(n):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        completions.append(response.choices[0].message['content'])
    
    return completions

def majority_vote(answers):
    """Return the majority vote from a list of answers."""
    # Round answers to 2 decimal places to handle floating point precision issues
    rounded_answers = [round(ans, 2) for ans in answers]
    counter = Counter(rounded_answers)
    return counter.most_common(1)[0][0] if counter else None

def evaluate_problems():
    """Evaluate problems using both methods and compare results."""
    problems = load_problems("problems.json")
    
    deterministic_results = []
    majority_vote_results = []
    
    for i, problem in enumerate(problems):
        print(f"Processing problem {i+1}/{len(problems)}")
        
        # Method 1: Single deterministic run
        completion = get_single_completion(problem["question"], temperature=0)
        deterministic_answer = extract_answer(completion)
        deterministic_correct = abs(deterministic_answer - problem["answer"]) < 0.01 if deterministic_answer is not None else False
        deterministic_results.append(deterministic_correct)
        
        print(f"  Deterministic answer: {deterministic_answer}, Correct: {deterministic_correct}")
        
        # Method 2: Majority vote
        completions = get_multiple_completions(problem["question"], n=10, temperature=1.1)
        answers = []
        
        for completion in completions:
            answer = extract_answer(completion)
            if answer is not None:
                answers.append(answer)
        
        majority_answer = majority_vote(answers) if answers else None
        majority_correct = abs(majority_answer - problem["answer"]) < 0.01 if majority_answer is not None else False
        majority_vote_results.append(majority_correct)
        
        print(f"  Majority vote answer: {majority_answer}, Correct: {majority_correct}")
        print(f"  All answers: {answers}")
        
    # Calculate accuracy
    deterministic_accuracy = sum(deterministic_results) / len(deterministic_results)
    majority_accuracy = sum(majority_vote_results) / len(majority_vote_results)
    
    print(f"\nDeterministic accuracy: {deterministic_accuracy:.2f}")
    print(f"Majority vote accuracy: {majority_accuracy:.2f}")
    
    # Create visualization
    create_accuracy_plot(deterministic_accuracy, majority_accuracy)
    
    return deterministic_accuracy, majority_accuracy

def create_accuracy_plot(deterministic_accuracy, majority_accuracy):
    """Create a bar plot comparing the accuracies."""
    methods = ['Single run\n(temp=0)', 'Majority vote\n(temp=1.1)']
    accuracies = [deterministic_accuracy, majority_accuracy]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, accuracies, color=['#3498db', '#e74c3c'])
    
    plt.title('Accuracy Comparison: Single Run vs. Majority Vote', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('accuracy.png')
    print("Accuracy plot saved to 'accuracy.png'")

if __name__ == "__main__":
    evaluate_problems() 