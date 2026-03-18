# benchmark.py
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

from tools.llm_helpers import evaluate_scene_edit_from_files

def main():

    ap = argparse.ArgumentParser(description="Evaluate scene edits by traversing a multi-task directory structure.")
    ap.add_argument("--input_dir", required=True, help="Path to the root directory containing scene subdirectories (e.g., 'bathroom', 'bedroom').")
    ap.add_argument("--output_json", required=True, help="Output JSON file path for all evaluation results.")
    ap.add_argument("--model", default="gpt-5", help="LVLM model to use for evaluation.")
    args = ap.parse_args()

    input_root = Path(args.input_dir)
    if not input_root.is_dir():
        print(f"Error: Input directory not found at {args.input_dir}", file=sys.stderr)
        return 1

    tasks_to_process = []
    print(f"Scanning for scenes and tasks in '{input_root}'...")
    
    scene_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    for scene_dir in scene_dirs:
        print(f"\n--- Processing Scene: {scene_dir.name} ---")

        source_image_file = scene_dir / "source.png"
        instructions_file = scene_dir / "instructions.json"

        if not source_image_file.exists() or not instructions_file.exists():
            print(f"Warning: Skipping scene '{scene_dir.name}' due to missing 'source.png' or 'instructions.json'.")
            continue

        try:
            with open(instructions_file, "r", encoding="utf-8") as f:
                instructions_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read or parse 'instructions.json' for scene '{scene_dir.name}': {e}", file=sys.stderr)
            continue

        edit_instructions_list = instructions_data.get("edit_instructions", [])
        if not edit_instructions_list:
            print(f"Warning: No 'edit_instructions' list found or list is empty in '{instructions_file.name}' for scene '{scene_dir.name}'.")
            continue

        for item in edit_instructions_list:
            inst_key = item.get("command")
            instruction_text = item.get("instruction")

            if not inst_key or not instruction_text:
                continue

            edited_image_file = scene_dir / f"{inst_key}.png"

            if edited_image_file.exists():
                tasks_to_process.append({
                    "id": f"{scene_dir.name}_{inst_key}",
                    "scene": scene_dir.name,
                    "instruction_key": inst_key,
                    "instruction": instruction_text,
                    "source_image_path": str(source_image_file),
                    "edited_image_path": str(edited_image_file)
                })

    if not tasks_to_process:
        print("\nError: No valid tasks found. Check that 'instructions.json' files contain a valid 'edit_instructions' list and that corresponding .png files exist.", file=sys.stderr)
        return 1

    print(f"\n--- Found a total of {len(tasks_to_process)} valid tasks across all scenes. Starting evaluation. ---")

    all_results = []
    for task in tqdm(tasks_to_process, desc="Evaluating all scene edits"):
        try:
            result = evaluate_scene_edit_from_files(
                instruction=task["instruction"],
                source_image_path=task["source_image_path"],
                edited_image_path=task["edited_image_path"],
                model=args.model,
            )
            task_result = {**task, "evaluation": result}
            all_results.append(task_result)
        except Exception as e:
            print(f"\nError processing task '{task['id']}': {e}", file=sys.stderr)
            task_result = {**task, "evaluation": {"error": str(e)}}
            all_results.append(task_result)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation complete. Results saved to: {args.output_json}")
    print(f"Successfully processed {len(all_results)} tasks.")
    return 0

if __name__ == "__main__":
    main()