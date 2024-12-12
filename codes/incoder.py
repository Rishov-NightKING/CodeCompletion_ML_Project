import csv
import os
import sys

import torch
from constants import (
    COMPLETION_PLACEHOLDER,
    COMPLETION_TYPE_REPR,
    END_GT,
    INSTRUCTION_TEMPLATE,
    OUTPUT_CSV_FILE_HEADER,
    START_GT,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, NoBadWordsLogitsProcessor
from utils import read_jsonl_file


class Incoder:
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("facebook/incoder")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        device = torch.device("cuda")
        if model_name.endswith("6B"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [
                        [word_idx]
                        for word_idx in self.tokenizer.convert_tokens_to_ids(
                            [
                                "#Ġ",
                                "/*Ġ",
                                "Ġ#Ġ",
                                "ĠĠ#Ġ",
                                "ĠĠ/*Ġ",
                                "ĠĠĠ#Ġ",
                                "ĠĠĠ/*Ġ",
                                "ĠĠĠĠ#Ġ",
                                "ĠĠĠĠ/*Ġ",
                                "ĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠ/*Ġ",
                                "ĠĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠĠĠ/*Ġ",
                                "ĠĠĠĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ",
                                "ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ",
                            ]
                        )
                    ],
                    self.tokenizer.eos_token_id,
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]

        if input_ids_len + 128 > self.max_length:
            return None

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_length=min(input_ids_len + 128, self.max_length),
                top_p=0.95,
                logits_processor=self.logits_processor,
            )
        generated_text = self.tokenizer.decode(generated_ids[0, input_ids_len:], clean_up_tokenization_spaces=False)
        EOM = "<|endofmask|>"
        if EOM not in generated_text:
            generated_text += EOM
        return generated_text[: generated_text.index(EOM)]

    @staticmethod
    def make_sentinel(i):
        # signals (1) a location to insert an infill and (2) the start of the infill generation
        return f"<|mask:{i}|>"

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return prefix + self.make_sentinel(0) + suffix + self.make_sentinel(1) + self.make_sentinel(0)
        else:
            return self.make_sentinel(0) + suffix + self.make_sentinel(1) + self.make_sentinel(0) + prefix


def run_zero_shot(samples, completion_placeholder, output_file_path):
    print("######################## RUNNING ZERO SHOT #############################")
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header line
        writer.writerow(OUTPUT_CSV_FILE_HEADER)
        for sample in tqdm(samples, desc="Processing Samples"):
            prefix, suffix = sample["prompt"].split(completion_placeholder)
            prompt = "<|mask:0|>" + suffix + "<|mask:1|>" + "<|mask:0|>" + prefix
            completion = incoder_model.invoke(prompt)

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
        count = 0
        for sample in tqdm(samples, desc="Processing Samples"):
            examples = sample["few_shot_examples"]
            few_shot_prompt = ""
            for example in examples[:1]:  # only take 1 example
                few_shot_prompt += (
                    f"{example['prompt']}\n"
                    f"{instruction}\n"
                    f"{START_GT}\n"
                    f"{example['ground_truth']}\n"
                    f"{END_GT}\n"
                )

            # Combine few-shot examples with current prompt
            prompt = few_shot_prompt + sample["prompt"] + f"{instruction}\n"

            # Generate completion
            completion = incoder_model.invoke(prompt)

            if completion is None:
                count += 1
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
            elif START_GT in completion:
                filtered_completion = completion.split(START_GT)[1].strip()
            else:
                filtered_completion = completion

            # Write row to CSV
            writer.writerow([sample["eval_prompt"], sample["ground_truth"], filtered_completion])

            # print("Model Output: ", filtered_completion)

        print(f"total skipped: {count}")


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
    parameters = "6B"
    model_name = f"facebook/incoder-{parameters}"
    block_comments = True
    model_max_length = 2048

    # initialize the model
    incoder_model = Incoder(model_name, model_max_length, block_comments)
    # read the dataset
    input_file_path = f"../Dataset/processed_safim_few_shot/{completion_type}_completion_processed_few_shot.jsonl"
    python_samples = read_jsonl_file(input_file_path, "python")

    completion_placeholder = COMPLETION_PLACEHOLDER[lang]
    completion_type = COMPLETION_TYPE_REPR[completion_type]
    output_file_name = os.path.basename(input_file_path).split(".jsonl")[0] + "_output.csv"

    ######################## ZERO SHOT #############################
    if training_type == "zero-shot":
        zero_shot_result_dir = f"../results/incoder-{parameters}/zero-shot"
        zero_shot_output_file_path = os.path.join(zero_shot_result_dir, output_file_name)
        os.makedirs(zero_shot_result_dir, exist_ok=True)

        run_zero_shot(python_samples, completion_placeholder, zero_shot_output_file_path)

    ######################## FEW SHOT ################################
    if training_type == "few-shot":
        few_shot_result_dir = f"../results/incoder-{parameters}/few-shot"
        few_shot_output_file_path = os.path.join(few_shot_result_dir, output_file_name)
        os.makedirs(few_shot_result_dir, exist_ok=True)

        instruction = INSTRUCTION_TEMPLATE.render(placeholder=completion_placeholder, completion_type=completion_type)
        run_few_shot(python_samples, completion_placeholder, instruction, few_shot_output_file_path)
