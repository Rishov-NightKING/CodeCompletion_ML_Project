import os
from utils import read_jsonl_file, write_records_to_new_jsonl_file


def generate_full_ground_truth(records):
    new_field = "full_ground_truth"
    modified_records = []
    for record in records:
        new_value = record.get("eval_prompt").replace("{{completion}}", record.get("ground_truth"))
        record[new_field] = new_value
        modified_records.append(record)
    return modified_records


processed_dataset_dir = "../Dataset/processed_safim"
os.makedirs(processed_dataset_dir, exist_ok=True)

dataset_path = "../Dataset/safim"
jsonl_files = [f for f in os.listdir(dataset_path) if f.endswith(".jsonl")]

for jsonl_file in jsonl_files:
    jsonl_file_path = os.path.join(dataset_path, jsonl_file)
    jsonl_file_records = read_jsonl_file(jsonl_file_path, filter_by_lang="python")

    full_ground_truth_records = generate_full_ground_truth(jsonl_file_records)

    processed_file_name = jsonl_file.split(".jsonl")[0] + "_processed.jsonl"
    processed_jsonl_output_file_path = os.path.join(processed_dataset_dir, processed_file_name)
    write_records_to_new_jsonl_file(processed_jsonl_output_file_path, jsonl_file_records)
