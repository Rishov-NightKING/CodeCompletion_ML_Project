import os
import warnings

import pandas as pd
from codebleu import calc_codebleu

# Ignore all warnings
warnings.filterwarnings("ignore")


def exact_match(df):
    exact_matches = 0
    total_rows = len(df)
    for index, row in df.iterrows():
        ground_truth = str(row["ground_truth"])
        model_output = str(row["model_output"])

        if ground_truth.strip() == model_output.strip():
            exact_matches += 1

    accuracy = (exact_matches / total_rows) * 100
    return accuracy


def match_if_contains(df):
    exact_matches = 0
    total_rows = len(df)
    for index, row in df.iterrows():
        ground_truth = str(row["ground_truth"])
        model_output = str(row["model_output"])

        if ground_truth.strip() in model_output.strip():
            exact_matches += 1

    accuracy = (exact_matches / total_rows) * 100
    return accuracy


def generate_full_code_snippets(df_row):
    eval_prompt = str(df_row["eval_prompt"])
    ground_truth = str(df_row["ground_truth"])
    model_output = str(df_row["model_output"])
    ground_truth_code = eval_prompt.replace("{{completion}}", ground_truth)
    model_output_code = eval_prompt.replace("{{completion}}", model_output)
    return [ground_truth_code, model_output_code]


def codebleu(df):
    code_bleu_total = 0
    total_rows = len(df)
    for i, row in df.iterrows():
        full_snippets = generate_full_code_snippets(row)

        code_bleu_score = calc_codebleu([full_snippets[0]], [full_snippets[1]], lang="python")

        code_bleu_total += code_bleu_score["codebleu"]

    return code_bleu_total / total_rows


results_dir = "../results"

for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            path_parts = root.split(os.sep)

            if len(path_parts) >= 2:
                model_name = path_parts[-2]
                shot_type = path_parts[-1]
            else:
                model_name = "unknown_model"
                shot_type = "unknown_shot_type"

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Print the results
            print(f"Model: {model_name}, Training Type: {shot_type}, File: {file}")
            exact_match_accuracy = exact_match(df)
            print(f"Exact Match Accuracy: {exact_match_accuracy:.2f}%")
            match_if_contains_accuracy = match_if_contains(df)
            print(f"Match if Contains Accuracy: {match_if_contains_accuracy:.2f}%")
            code_bleu_accuracy = codebleu(df)
            print(f"Code BLUE accuracy: {code_bleu_accuracy:.2f}\n\n")
