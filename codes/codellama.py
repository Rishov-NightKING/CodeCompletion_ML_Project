import csv
import os
import sys
import transformers
import torch
from constants import (
    COMPLETION_PLACEHOLDER,
    COMPLETION_TYPE_REPR,
    END_GT,
    INSTRUCTION_TEMPLATE,
    OUTPUT_CSV_FILE_HEADER,
    START_GT,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, NoBadWordsLogitsProcessor
from utils import read_jsonl_file


class CodeLlama():
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("codellama/CodeLlama")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(["#", "▁#", "/*", "▁/*"])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        generated_text = self.pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.2,
            max_new_tokens=128,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=0.95,
            handle_long_generation="hole",
            logits_processor=self.logits_processor
        )[0]["generated_text"]
        return generated_text[len(prompt):]




def run_zero_shot(samples, completion_placeholder, output_file_path):
    print("######################## RUNNING ZERO SHOT #############################")
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header line
        writer.writerow(OUTPUT_CSV_FILE_HEADER)
        for sample in samples:
            prefix, suffix = sample["prompt"].split(completion_placeholder)
            # prompt = "<PRE>" + prefix + " <SUF>" + suffix + " <MID>"
            prompt = "<|mask:0|>" + suffix + "<|mask:1|>" + "<|mask:0|>" + prefix
            completion = codellama_model.invoke(prompt)

            if completion is None:
                continue

            # Write row to CSV
            writer.writerow([sample["eval_prompt"], sample["ground_truth"], completion])


def run_few_shot(samples, completion_placeholder, instruction, output_file_path):
    print("######################## RUNNING FEW SHOT #############################")
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header line
        writer.writerow(OUTPUT_CSV_FILE_HEADER)
        for sample in samples:
            examples = sample["few_shot_examples"]
            few_shot_prompt = ""
            for example in examples:
                few_shot_prompt += (
                    f"{example['prompt']}\n"
                    f"{instruction}\n"
                    f"{START_GT}\n"
                    f"{example['ground_truth']}\n\n"
                    f"{END_GT}\n"
                )

            # Combine few-shot examples with current prompt
            prompt = few_shot_prompt + sample["prompt"] + f"{instruction}\n"

            # Generate completion
            completion = codellama_model.invoke(prompt)

            if completion is None:
                print("None")
                continue

            # Output results
            # print("\n")
            # print(f"Task ID: {sample['task_id']}")
            # print("Prompt: ", prompt)
            # print(f"example: {count}\n")
            # print("Ground Truth: ", sample["ground_truth"])

            if START_GT in completion and END_GT in completion:
                filtered_completion = completion.split(START_GT)[1].split(END_GT)[0].strip()
            elif END_GT in completion:
                filtered_completion = completion.split(END_GT)[0].strip()
            else:
                filtered_completion = completion

            # Write row to CSV
            writer.writerow([sample["eval_prompt"], sample["ground_truth"], filtered_completion])

        # print("Model Output: ", filtered_completion)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <completion_type> <training_type>")
        sys.exit(1)

    possible_completion_types = COMPLETION_TYPE_REPR.keys()  # block, control, api
    completion_type = sys.argv[1]
    if completion_type not in possible_completion_types:
        print(f"Wrong completion type. please select among {possible_completion_types}")
        sys.exit(1)

    possible_training_types = ["zero-shot", "few-shot"]
    training_type = sys.argv[2]
    if training_type not in possible_training_types:
        print(f"Wrong training type. please select among {possible_training_types}")
        sys.exit(1)

    # initialize the constants
    lang = "python"
    parameters = "7b-Python-hf"
    model_name = f"codellama/CodeLlama-{parameters}"
    block_comments = True
    model_max_length = 4096

    # initiliaze the model
    codellama_model = CodeLlama(model_name, model_max_length, block_comments)
    # read the dataset
    input_file_path = f"../Dataset/processed_safim_few_shot/{completion_type}_completion_processed_few_shot.jsonl"
    python_samples = read_jsonl_file(input_file_path, "python")

    completion_placeholder = COMPLETION_PLACEHOLDER[lang]
    completion_type = COMPLETION_TYPE_REPR[completion_type]
    output_file_name = os.path.basename(input_file_path).split(".jsonl")[0] + "_output.csv"

    ######################## ZERO SHOT #############################
    if training_type == "zero-shot":
        zero_shot_result_dir = f"../results/codellama-{parameters}/zero-shot"
        zero_shot_output_file_path = os.path.join(zero_shot_result_dir, output_file_name)
        os.makedirs(zero_shot_result_dir, exist_ok=True)

        run_zero_shot(python_samples, completion_placeholder, zero_shot_output_file_path)

    ######################## FEW SHOT ################################
    if training_type == "few-shot":
        few_shot_result_dir = f"../results/codellama-{parameters}/few-shot"
        few_shot_output_file_path = os.path.join(few_shot_result_dir, output_file_name)
        os.makedirs(few_shot_result_dir, exist_ok=True)

        instruction = INSTRUCTION_TEMPLATE.render(placeholder=completion_placeholder, completion_type=completion_type)
        run_few_shot(python_samples, completion_placeholder, instruction, few_shot_output_file_path)
