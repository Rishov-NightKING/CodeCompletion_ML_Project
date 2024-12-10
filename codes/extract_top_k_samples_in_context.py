import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import write_records_to_new_jsonl_file


def calculate_similarity_and_update(jsonl_file, output_file, top_k=2):
    # Read the JSONL file and extract records and prompts
    records = []
    prompts = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            record = json.loads(line)
            records.append(record)
            prompts.append(record['prompt'])

    # Calculate TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(prompts)
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Compute top matches and update records
    for i, record in enumerate(records):
        similarity_scores = list(enumerate(cosine_sim[i]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        # Take top k (excluding self-match)
        top_k_scores = similarity_scores[1:1+top_k]
        few_shot_examples = [
            {
                "prompt": prompts[j],
                "ground_truth": records[j].get("ground_truth"),  # Include ground_truth
                "similarity_score": score
            }
            for j, score in top_k_scores
        ]
        # Add the few_shot_examples field to the record
        record['few_shot_examples'] = few_shot_examples

    write_records_to_new_jsonl_file(output_file, records)


dataset_path = "../Dataset/processed_safim"
few_shot_dataset_path = "../Dataset/processed_safim_few_shot"
os.makedirs(few_shot_dataset_path, exist_ok=True)

jsonl_files = [f for f in os.listdir(dataset_path) if f.endswith(".jsonl")]

for jsonl_file in jsonl_files:
    jsonl_file_path = os.path.join(dataset_path, jsonl_file)
    few_shot_file_name = jsonl_file.split(".jsonl")[0] + "_few_shot.jsonl"
    few_shot_file_path = os.path.join(few_shot_dataset_path, few_shot_file_name)
    calculate_similarity_and_update(jsonl_file_path, few_shot_file_path)

