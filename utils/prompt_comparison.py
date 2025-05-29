# Import necessary libraries
import requests
import json
import os
# Define a question to experiment with
question = "What foods should be avoided by patients with gout?"

# Example for one-shot and few-shot prompting
example_q = "What are the symptoms of gout?"
example_a = "Gout symptoms include sudden severe pain, swelling, redness, and tenderness in joints, often the big toe."

# Examples for few-shot prompting
examples = [
    ("What are the symptoms of gout?",
     "Gout symptoms include sudden severe pain, swelling, redness, and tenderness in joints, often the big toe."),
    ("How is gout diagnosed?",
     "Gout is diagnosed through physical examination, medical history, blood tests for uric acid levels, and joint fluid analysis to look for urate crystals.")
]

# Create prompting templates
# Zero-shot template (just the question)
zero_shot_template = "Question: {question}\nAnswer:"

# One-shot template (one example + the question)
one_shot_template = """Question: {example_q}
Answer: {example_a}

Question: {question}
Answer:"""

# Few-shot template (multiple examples + the question)
few_shot_template = """Question: {examples[0][0]}
Answer: {examples[0][1]}

Question: {examples[1][0]}
Answer: {examples[1][1]}

Question: {question}
Answer:"""

# Format the templates with your question and examples
zero_shot_prompt = zero_shot_template.format(question=question)
one_shot_prompt = one_shot_template.format(example_q=example_q, example_a=example_a, question=question)
# For few-shot, you'll need to format it with the examples list
few_shot_prompt = few_shot_template.format(examples=examples, question=question)

print("Zero-shot prompt:")
print(zero_shot_prompt)
print("\nOne-shot prompt:")
print(one_shot_prompt)
print("\nFew-shot prompt:")
print(few_shot_prompt)

api_key = os.getenv("HUGGINGFACE_API_KEY")
model_name = "google/flan-t5-base"
def get_llm_response(prompt, model_name="google/flan-t5-base", api_key=None):
    """Get a response from the LLM based on the prompt"""
    # Implement the get_llm_response function
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100}
    }
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    output = response.json()
    if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
    elif isinstance(output, list):
            return output[0]  # Fallback if format is simpler
    else:
        return f"Unexpected response format: {output}"

# Test your get_llm_response function with different prompts
print("Zero-shot response:")
print(get_llm_response(zero_shot_prompt, api_key=api_key))

print("\nOne-shot response:")
print(get_llm_response(one_shot_prompt, api_key=api_key))

print("\nFew-shot response:")
print(get_llm_response(few_shot_prompt, api_key=api_key))

# List of healthcare questions to test
questions = [
    "What foods should be avoided by patients with gout?",
    "What medications are commonly prescribed for gout?",
    "How can gout flares be prevented?",
    "Is gout related to diet?",
    "Can gout be cured permanently?"
]

# Compare the different prompting strategies on these questions
# For each question:
# - Create prompts using each strategy
# - Get responses from the LLM
# - Store the results
def build_prompt(question, strategy):
    if strategy == "zero_shot":
        return f"Question: {question}\nAnswer:"
    elif strategy == "one_shot":
        return (f"Question: {examples[0][0]}\nAnswer: {examples[0][1]}\n\n"
                f"Question: {question}\nAnswer:")
    elif strategy == "few_shot":
        return (f"Question: {examples[0][0]}\nAnswer: {examples[0][1]}\n\n"
                f"Question: {examples[1][0]}\nAnswer: {examples[1][1]}\n\n"
                f"Question: {question}\nAnswer:")
    
def score_response(response, keywords):
    """Score a response based on the presence of expected keywords"""
    response = response.lower()
    found_keywords = 0
    for keyword in keywords:
        if keyword.lower() in response:
            found_keywords += 1
    return found_keywords / len(keywords) if keywords else 0

# Expected keywords for each question
expected_keywords = {
    "What foods should be avoided by patients with gout?": 
        ["purine", "red meat", "seafood", "alcohol", "beer", "organ meats"],
    "What medications are commonly prescribed for gout?": 
        ["nsaids", "colchicine", "allopurinol", "febuxostat", "probenecid", "corticosteroids"],
    "How can gout flares be prevented?": 
        ["medication", "diet", "weight", "alcohol", "water", "exercise"],
    "Is gout related to diet?": 
        ["yes", "purine", "food", "alcohol", "seafood", "meat"],
    "Can gout be cured permanently?": 
        ["manage", "treatment", "lifestyle", "medication", "chronic"]
}

# Score the responses and calculate average scores for each strategy
def compare_prompting_strategies(questions, model_name, api_key):
    results = []
    strategies = ["zero_shot", "one_shot", "few_shot"]
    strategy_scores = {s: 0 for s in strategies}

    for question in questions:
        response_set = {}
        score_set = {}

        for strategy in strategies:
            prompt = build_prompt(question, strategy)
            try:
                response = get_llm_response(prompt, model_name=model_name, api_key=api_key)
            except Exception as e:
                response = f"Error: {e}"

            score = score_response(response, expected_keywords.get(question, []))
            response_set[strategy] = response
            score_set[strategy] = score
            strategy_scores[strategy] += score

        results.append({
            "question": question,
            "responses": response_set,
            "scores": score_set
        })

    averages = {s: strategy_scores[s] / len(questions) for s in strategies}
    best_strategy = max(averages, key=averages.get)
    return results, averages, best_strategy
# Determine which strategy performs best overall

# Save your results to results/part_3/prompting_results.txt
# The file should include:
# - Raw responses for each question and strategy
# - Scores for each question and strategy
# - Average scores for each strategy
# - The best performing strategy

# Example format:
"""
# Prompt Engineering Results

## Question: What foods should be avoided by patients with gout?

### Zero-shot response:
[response text]

### One-shot response:
[response text]

### Few-shot response:
[response text]

--------------------------------------------------

## Scores

```
question,zero_shot,one_shot,few_shot
what_foods_should,0.67,0.83,0.83
what_medications_are,0.50,0.67,0.83
how_can_gout,0.33,0.50,0.67
is_gout_related,0.80,0.80,1.00
can_gout_be,0.40,0.60,0.80

average,0.54,0.68,0.83
best_method,few_shot
```
"""
def save_results(results, averages, best_strategy, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Prompt Engineering Results\n\n")

        for result in results:
            f.write(f"## Question: {result['question']}\n\n")
            for strategy in ["zero_shot", "one_shot", "few_shot"]:
                f.write(f"### {strategy.replace('_', ' ').title()} response:\n")
                f.write(f"{result['responses'][strategy]}\n\n")
            f.write("-" * 50 + "\n\n")

        # Scores summary
        f.write("## Scores\n\n```csv\n")
        f.write("question,zero_shot,one_shot,few_shot\n")
        for result in results:
            label = result['question'].split("?")[0].lower().replace(" ", "_").strip(",")
            zs = result['scores']['zero_shot']
            os_ = result['scores']['one_shot']
            fs = result['scores']['few_shot']
            f.write(f"{label},{zs:.2f},{os_:.2f},{fs:.2f}\n")
        f.write(f"\naverage,{averages['zero_shot']:.2f},{averages['one_shot']:.2f},{averages['few_shot']:.2f}")
        f.write(f"\nbest_method,{best_strategy}\n```")
    
if __name__ == "__main__":
    output_path = "results/part_3/prompting_results.txt"
    results, avg_scores, best = compare_prompting_strategies(questions, model_name, api_key)
    save_results(results, avg_scores, best, output_path)
    print(f"\nResults saved to: {output_path}")