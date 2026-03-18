"""CLI for generating plans from goal conditions."""

import argparse
import json
import sys
from pathlib import Path
from typing import Set, List, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from editors.editlang import EditLangDomain, Predicate, standard_domain
from planners.regression_planner import RegressionPlanner
from validators.llm_semantic_validator import LLMSemanticValidator
from validators.geom_checker import GeomChecker
from tools.llm_helpers import LLMHelper
from runner.execute_plan import extract_initial_state


def load_terminal_conditions(terminal_file: str) -> List[Tuple[str, Set[Predicate]]]:
    with open(terminal_file, 'r') as f:
        data = json.load(f)
    goals = []
    # Handle both list-of-dicts and single dict formats
    items = data if isinstance(data, list) else [data]
    for i, item in enumerate(items):
        command_name = item.get("command", f"plan_{i}")
        pred_list = item.get("terminal", item.get("terminal_condition", []))
        predicates = set()
        for pred_data in pred_list:
            pred_name = pred_data.get("pred", pred_data.get("predicate"))
            pred_args = pred_data.get("args", pred_data.get("arguments", []))
            if pred_name:
                predicates.add((pred_name, tuple(pred_args)))
        goals.append((command_name, predicates))
    return goals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", required=True)
    parser.add_argument("--terminal_json", required=True)
    parser.add_argument("--instructions_json")
    parser.add_argument("--domain_yaml")
    parser.add_argument("--out_plan", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm-validator", choices=["on", "off"], default="on")
    parser.add_argument("--schema-validation", choices=["on", "off"], default="off")
    parser.add_argument("--k-tail", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=400000)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()

    # --- Initial Setup ---
    with open(args.scene_json, 'r') as f:
        scene = json.load(f)
    
    initial_state = extract_initial_state(scene)
    if args.verbose:
        print(f"Initial scene state has {len(initial_state)} predicates.")
    
    all_goals = load_terminal_conditions(args.terminal_json)
    
    instruction_map = {}
    if args.instructions_json:
        try:
            with open(args.instructions_json, 'r') as f:
                inst_data = json.load(f)
            for item in inst_data.get("edit_instructions", []):
                instruction_map[item.get("command")] = item.get("instruction")
        except Exception:
            pass

    domain = EditLangDomain.from_yaml(args.domain_yaml) if args.domain_yaml else standard_domain()

    # --- Helper & Validators Setup ---
    llm_helper = None
    if args.use_llm or args.llm_validator == "on":
        try:
            llm_helper = LLMHelper(verbose=args.verbose)
        except Exception as e:
            print(f"Warning: LLM init failed: {e}")

    llm_validator = None
    if args.llm_validator == "on" and llm_helper:
        llm_validator = LLMSemanticValidator(
            editlang_spec=domain.to_dict(),
            model_client=llm_helper,
            timeout_ms=args.timeout_ms,
            verbose=args.verbose
        )
        if args.verbose: print("LLM Semantic Validator enabled.")

    #Geometric Checker Setup
    geom_checker = None
    try:
        geom_checker = GeomChecker(scene_data=scene)
        if args.verbose: print("Geometric Validator (GeomChecker) enabled.")
    except Exception as e:
        print(f"Warning: GeomChecker init failed: {e}")

    # --- Planner Init ---
    planner = RegressionPlanner(
        domain=domain,
        scene_data=scene,
        llm_helper=llm_helper,
        llm_validator=llm_validator,
        geom_checker=geom_checker,  # Inject Physics Check
        skip_schema_validation=(args.schema_validation == "off"),
        max_steps=args.max_steps,
        verbose=args.verbose
    )

    # --- Planning Loop ---
    all_plans = {}
    try:
        # Load existing if resuming
        if args.start_idx > 0 and Path(args.out_plan).exists():
            with open(args.out_plan, 'r') as f:
                all_plans = json.load(f)

        with open(args.out_plan, 'w') as f:
            json.dump(all_plans, f, indent=2) # Init file

            for idx, (command, G) in enumerate(tqdm(all_goals, desc="Planning")):
                if idx < args.start_idx: continue
                
                instruction_raw = instruction_map.get(command)
                if args.verbose:
                    print(f"\n--- Planning for: {command} ---")
                    print(f"Instruction: {instruction_raw}")

                plan_data = []
                try:
                    plan = planner.plan(
                        s0=initial_state,
                        G=G,
                        instruction_raw=instruction_raw,
                        G_terminal=G
                    )
                    plan_data = [action.to_dict() for action in plan]
                except Exception as e:
                    print(f"Planning failed for {command}: {e}")
                
                all_plans[command] = plan_data
                
                # Save incrementally
                f.seek(0)
                json.dump(all_plans, f, indent=2)
                f.truncate()

    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)

    print(f"\nDone. Plans saved to {args.out_plan}")

if __name__ == "__main__":
    main()