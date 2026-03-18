# cli/derive_terminal.py
import json, argparse, sys
from pathlib import Path
from tqdm import tqdm

from tools.llm_helpers import extract_terminal_conditions_from_file
from editors.editlang import EditLangDomain

def main():
    ap = argparse.ArgumentParser(description="Derive terminal goal predicates from a JSON file with multiple instructions.")
    ap.add_argument("--scene_json", required=True, help="Path to scene JSON for context")
    ap.add_argument("--input_json", required=True, help="Path to the input JSON file with instructions")
    ap.add_argument("--output_json", required=True, help="Output JSON file path for all terminal conditions")
    ap.add_argument("--model", default="gpt-5", help="LLM model to use (optional)")
    ap.add_argument("--domain_yaml", help="Optional domain YAML to constrain allowed predicate names")
    args = ap.parse_args()

    allowed_preds = None
    if args.domain_yaml:
        domain = EditLangDomain.from_yaml(args.domain_yaml)
        # FIX: Convert all items to string before sorting to handle YAML parsing 'on' as True
        allowed_preds = sorted([str(p) for p in (domain.predicates or [])])

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_terminal_conditions = []
    
    instructions = data.get("edit_instructions", [])

    for item in tqdm(instructions, desc="Processing instructions"):
        instruction = item.get("instruction")
        command = item.get("command")

        if not instruction:
            continue

        # Use helper to extract terminal conditions; returns dict with key 'terminal_condition'
        result = extract_terminal_conditions_from_file(
            instruction=instruction,
            scene_file=args.scene_json,
            model=args.model,
            allowed_predicates=allowed_preds,
        )
        
        all_terminal_conditions.append({
            "command": command,
            "terminal": result.get("terminal_condition", [])
        })

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_terminal_conditions, f, ensure_ascii=False, indent=2)
    
    print(f"\nTerminal conditions saved to: {args.output_json}")
    print(f"Processed {len(all_terminal_conditions)} instructions.")

if __name__ == "__main__":
    sys.exit(main())