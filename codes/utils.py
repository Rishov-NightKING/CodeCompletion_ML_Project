import json


def read_jsonl_file(file_path, filter_by_lang=None):
    filtered_records = []
    with open(file_path, "r") as file:
        for line in file:
            record = json.loads(line)
            if not filter_by_lang or record.get("lang") == filter_by_lang:
                filtered_records.append(record)

    return filtered_records


def write_records_to_new_jsonl_file(output_file_path, records):
    with open(output_file_path, "w") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")
