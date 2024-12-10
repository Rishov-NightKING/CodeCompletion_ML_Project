import jinja2

COMPLETION_PLACEHOLDER = {
    "python": "# TODO: Your code here",
    "java": "/* TODO: Your code here */",
    "cpp": "/* TODO: Your code here */",
    "csharp": "/* TODO: Your code here */",
}

COMPLETION_TYPE_REPR = {"block": "block", "control": "control expression", "api": "API function call"}

INSTRUCTION_TEMPLATE = jinja2.Template(
    'Replace the "{{placeholder}}" in the code above with the appropriate {{completion_type}}. Provide only the replaced {{completion_type}}.'
)

START_GT = "this is the start of the ground truth"
END_GT = "this is the end of the ground truth"

OUTPUT_CSV_FILE_HEADER = ["eval_prompt", "ground_truth", "model_output"]
