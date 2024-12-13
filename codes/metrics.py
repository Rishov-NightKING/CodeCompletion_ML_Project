import pandas as pd
import sys
from pathlib import Path
file_path="/scratch/nwc8gr/CodeCompletion_ML_Project/results/codellama-7b-Python-hf/zero-shot/block_completion_processed_few_shot_output.csv"
df = pd.read_csv(file_path)

from codebleu import calc_codebleu



def exact_match(df):
    exact_matches = 0
    total_rows = len(df)
    for index, row in df.iterrows():
        eval_prompt = row['eval_prompt']
        ground_truth = row['ground_truth']
        model_output = row['model_output']

        if ground_truth.strip() == model_output.strip():
            exact_matches += 1

    accuracy = (exact_matches / total_rows) * 100
    return accuracy

def match_if_contains(df):
    exact_matches = 0
    total_rows = len(df)
    for index, row in df.iterrows():
        eval_prompt = row['eval_prompt']
        ground_truth = row['ground_truth']
        model_output = row['model_output']

        if ground_truth.strip() in model_output.strip():
            exact_matches += 1

    accuracy = (exact_matches / total_rows) * 100
    return accuracy
    
def generate_full_code_snippets(df_row):
    eval_prompt = df_row['eval_prompt']
    ground_truth = df_row['ground_truth']
    model_output = df_row['model_output']
    ground_truth_code = eval_prompt.replace("{{completion}}", ground_truth)
    model_output_code = eval_prompt.replace("{{completion}}", model_output)
    return [ground_truth_code, model_output_code]

def codebleu(df):
    code_bleu_total=0
    total_rows = len(df)
    for i, row in df.iterrows():
        full_snippets=generate_full_code_snippets(row)
        
        code_bleu_score=calc_codebleu([full_snippets[0]],[full_snippets[1]], lang="python")

        code_bleu_total+=code_bleu_score['codebleu']

    return (code_bleu_total/total_rows)




exact_match_accuracy=exact_match(df)
print(f"\nExact Match Accuracy: {exact_match_accuracy:.2f}%")
match_if_contains_accuracy=match_if_contains(df)
print(f"\nMatch if Contains Accuracy: {match_if_contains_accuracy:.2f}%")

code_bleu_accuracy=codebleu(df)
print(f"\nCode BLEU accuracy: {code_bleu_accuracy:.2f}")


