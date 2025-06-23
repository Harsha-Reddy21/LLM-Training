# "Thinking Mode" Sampler

This project explores chain-of-thought reasoning with self-consistency by comparing two approaches to solving arithmetic word problems:

1. Single deterministic run (temperature=0)
2. Majority vote from multiple completions (temperature=1.1)

## Setup

1. Install the required packages:
```
pip install openai matplotlib numpy
```

2. Set your OpenAI API key as an environment variable:
```
# On Windows
set OPENAI_API_KEY=your_api_key_here

# On macOS/Linux
export OPENAI_API_KEY=your_api_key_here
```

## Running the Script

1. Make sure you have the following files in place:
   - `self_consistency.py` - The main script
   - `problems.json` - Contains 10 GRE-style arithmetic word problems

2. Run the script:
```
python self_consistency.py
```

3. The script will:
   - Process each problem using both methods
   - Print the results for each problem
   - Generate an accuracy comparison plot saved as `accuracy.png`

## How It Works

1. For each problem, the script:
   - Generates a single completion with temperature=0
   - Generates 10 completions with temperature=1.1, each using "Let's think step-by-step"
   - Extracts numerical answers from each completion
   - Performs a majority vote over the 10 answers

2. The script compares the accuracy of both methods and visualizes the results.

## Results

The results are visualized in the `accuracy.png` file, showing the comparison between the two approaches. 