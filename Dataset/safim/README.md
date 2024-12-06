---
license: cc-by-4.0
task_categories:
- text2text-generation
language:
- en
tags:
- code-generation
- code-infilling
- fill-in-the-middle
pretty_name: SAFIM
size_categories:
- 10K<n<100K
configs:
  - config_name: block
    data_files:
      - split: test
        path: block_completion.jsonl.gz
  - config_name: control
    data_files:
      - split: test
        path: control_completion.jsonl.gz
  - config_name: api
    data_files:
      - split: test
        path: api_completion.jsonl.gz
  - config_name: block_v2
    data_files:
      - split: test
        path: block_completion_v2.jsonl.gz
  - config_name: control_fixed
    data_files:
      - split: test
        path: control_completion_fixed.jsonl.gz
---

# SAFIM Benchmark

Syntax-Aware Fill-in-the-Middle (SAFIM) is a benchmark for evaluating Large Language Models (LLMs) on
the code Fill-in-the-Middle (FIM) task. SAFIM has three subtasks: Algorithmic Block Completion,
Control-Flow Expression Completion, and API Function Call Completion. SAFIM is sourced from code
submitted from April 2022 to January 2023 to minimize the impact of data contamination on evaluation
results.

- Authors: [Linyuan Gong](https://gonglinyuan.com), Sida Wang, Mostafa Elhoushi, Alvin Cheung
- Paper: [https://arxiv.org/abs/2403.04814](https://arxiv.org/abs/2403.04814)
- Leaderboard: [https://safimbenchmark.com](https://safimbenchmark.com)
- Code & Submission Instructions: [https://github.com/gonglinyuan/safim](https://github.com/gonglinyuan/safim)

## Copyright Information

The SAFIM benchmark is partially derived from problem descriptions and code solutions from
[https://codeforces.com](https://codeforces.com). According to the license of CodeForces, you may publish the texts of
Codeforces problems in any open sources, but you must preserve a direct link to the site.

## Citation

```
@article{
    safim,
    title={Evaluation of {LLM}s on Syntax-Aware Code Fill-in-the-Middle Tasks},
    url={http://arxiv.org/abs/2403.04814},
    note={arXiv:2403.04814 [cs]},
    number={arXiv:2403.04814},
    publisher={arXiv},
    author={Gong, Linyuan and Wang, Sida and Elhoushi, Mostafa and Cheung, Alvin},
    year={2024},
    month=mar
}
```