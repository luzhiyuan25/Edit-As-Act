# [CVPR 2026] Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing

Website: coming soon  
arXiv: coming soon

## Requirements

- Ubuntu 24.04
- `uv`
- Blender 4.0+

## Installation

1. Clone this repository.
2. Sync the environment with `uv`.

```bash
git clone <repository-url>
cd EditAsAct
uv sync
```

## Dataset

Download the dataset from Google Drive:

[Dataset Download](https://drive.google.com/file/d/1frv046LtXh1e2EtuY7rLg5zCdJl6Y75o/view?usp=sharing)

After downloading, place the dataset under the `code/dataset/` directory so that the repository layout looks like this:

```text
EditAsAct/
├── code/
│   ├── dataset/
│   │   └── dataset/
│   │       ├── bedroom/
│   │       ├── dining_room/
│   │       ├── bathroom/
│   │       └── ...
│   └── ...
└── ...
```

## Usage

Set your OpenAI API key before running any LLM-based step:

```bash
export OPENAI_API_KEY="your_api_key"
```

### 1. Run the benchmark

This runs the full benchmark pipeline for a scene: terminal condition extraction followed by regression planning.

```bash
cd code
uv run python run_benchmark.py \
  --scene dataset/dataset/bedroom/scene_layout_edited.json \
  --instructions dataset/dataset/user_instructions_edit/instruction_bedroom.json \
  --output results/bedroom_results.json \
  --verbose
```

### 2. Run goal condition extraction only

Use this if you want to extract terminal goal predicates from the instruction file without running the planner.

```bash
cd code
uv run python -m cli.derive_terminal \
  --scene_json dataset/dataset/bedroom/scene_layout_edited.json \
  --input_json dataset/dataset/user_instructions_edit/instruction_bedroom.json \
  --output_json results/bedroom_terminal_conditions.json \
  --domain_yaml editors/editlang_std.yaml \
  --model gpt-5
```

### 3. Run the planner / model

For the current release, the recommended entry point for running the model is `run_benchmark.py`.
It performs both terminal condition extraction and goal-regressive planning, and saves the predicted plans as JSON.

```bash
cd code
uv run python run_benchmark.py \
  --scene dataset/dataset/bedroom/scene_layout_edited.json \
  --instructions dataset/dataset/user_instructions_edit/instruction_bedroom.json \
  --output results/bedroom_results.json
```

### 4. Convert predicted plans into scene JSON files

After the benchmark run, convert the predicted plans into edited scene layout JSON files:

```bash
cd code
uv run python tools/apply_plan_to_scene.py \
  --scene dataset/dataset/bedroom/scene_layout_edited.json \
  --plans results/bedroom_results.json \
  --outdir dataset/dataset/bedroom/plans_applied
```

This produces per-instruction scene files such as:

```text
code/dataset/dataset/bedroom/plans_applied/scene_layout_instruction_1_add.json
```

## Todo

- [ ] Dataset checked
- [ ] Planner checked
- [ ] Validator checked
- [ ] Visualization code checked

## BibTeX

```bibtex
@inproceedings{editasact2026,
  title     = {Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing},
  author    = {},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
