"""CLI for executing plans on scenes.

Usage:
    python -m cli.execute_plan \
        --scene_json scene.json \
        --plan plan.json \
        --out_scene scene_after.json \
        --out_log exec_log.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from editors.editlang import Action, Predicate
from runner.execute_plan import PlanExecutor, extract_initial_state


def load_plan(plan_file: str) -> List[Action]:
    """Load plan from JSON file.
    
    Args:
        plan_file: Path to plan JSON
        
    Returns:
        List of actions
    """
    with open(plan_file, 'r') as f:
        plan_data = json.load(f)
    
    actions = []
    for action_data in plan_data:
        if isinstance(action_data, dict):
            actions.append(Action.from_dict(action_data))
    
    return actions


def load_initial_state(state_file: str) -> Set[Predicate]:
    """Load initial state from JSON file.
    
    Args:
        state_file: Path to state JSON
        
    Returns:
        Set of predicates
    """
    with open(state_file, 'r') as f:
        state_data = json.load(f)
    
    predicates = set()
    
    # Handle both formats: direct list or wrapped
    if isinstance(state_data, list):
        pred_list = state_data
    elif "predicates" in state_data:
        pred_list = state_data["predicates"]
    elif "state" in state_data:
        pred_list = state_data["state"]
    else:
        pred_list = []
    
    for pred_data in pred_list:
        if isinstance(pred_data, dict):
            pred_name = pred_data.get("pred", pred_data.get("predicate"))
            pred_args = pred_data.get("args", pred_data.get("arguments", []))
            if pred_name:
                predicates.add((pred_name, tuple(pred_args)))
    
    return predicates


def execute_plan_on_scene(
    scene_file: str,
    plan_file: str,
    state_file: Optional[str] = None,
    verbose: bool = False,
    validate: bool = True
) -> tuple[Dict[str, Any], List[Dict[str, Any]], Set[Predicate]]:
    """Execute a plan on a scene.
    
    Args:
        scene_file: Path to scene JSON
        plan_file: Path to plan JSON
        state_file: Optional path to initial state JSON
        verbose: Print debug information
        validate: Validate action effects
        
    Returns:
        Tuple of (final_scene, execution_log, final_state)
    """
    # Load scene
    with open(scene_file, 'r') as f:
        scene = json.load(f)
    
    # Load plan
    plan = load_plan(plan_file)
    if verbose:
        print(f"Loaded plan with {len(plan)} actions")
    
    # Load or extract initial state
    if state_file:
        s0 = load_initial_state(state_file)
        if verbose:
            print(f"Loaded initial state with {len(s0)} predicates")
    else:
        s0 = extract_initial_state(scene)
        if verbose:
            print(f"Extracted initial state with {len(s0)} predicates")
    
    # Create executor
    config = {
        "verbose": verbose,
        "validate_effects": validate,
        "log_geometry": True
    }
    executor = PlanExecutor(scene, config)
    
    # Execute plan
    if verbose:
        print("\nExecuting plan...")
    
    final_state, execution_log = executor.execute(s0, plan)
    
    if verbose:
        print(f"\nExecution complete")
        print(f"Final state has {len(final_state)} predicates")
    
    # Get final scene
    final_scene = executor.get_scene_state()
    
    return final_scene, execution_log, final_state


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Execute a plan on a scene"
    )
    
    parser.add_argument(
        "--scene_json",
        required=True,
        help="Path to input scene JSON file"
    )
    parser.add_argument(
        "--plan",
        required=True,
        help="Path to plan JSON file"
    )
    parser.add_argument(
        "--state",
        help="Optional path to initial state JSON file"
    )
    parser.add_argument(
        "--out_scene",
        required=True,
        help="Path to output scene JSON file"
    )
    parser.add_argument(
        "--out_log",
        required=True,
        help="Path to output execution log JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose execution information"
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Disable effect validation"
    )
    parser.add_argument(
        "--print_final_state",
        action="store_true",
        help="Print final state predicates"
    )
    
    args = parser.parse_args()
    
    # Execute plan
    final_scene, execution_log, final_state = execute_plan_on_scene(
        scene_file=args.scene_json,
        plan_file=args.plan,
        state_file=args.state,
        verbose=args.verbose,
        validate=not args.no_validate
    )
    
    # Save final scene
    with open(args.out_scene, 'w') as f:
        json.dump(final_scene, f, indent=2)
    print(f"Final scene saved to: {args.out_scene}")
    
    # Create full log with metadata
    full_log = {
        "timestamp": datetime.now().isoformat(),
        "input_scene": args.scene_json,
        "input_plan": args.plan,
        "steps": execution_log,
        "final_state_size": len(final_state),
        "success": True
    }
    
    # Save execution log
    with open(args.out_log, 'w') as f:
        json.dump(full_log, f, indent=2)
    print(f"Execution log saved to: {args.out_log}")
    
    # Print summary
    print(f"\nExecution Summary:")
    print(f"  - Executed {len(execution_log)} actions")
    print(f"  - Final state has {len(final_state)} predicates")
    
    # Print action summary
    print("\nActions executed:")
    for step in execution_log:
        print(f"  {step['step']}. {step['action']} with args {step['args']}")
        if "geometry" in step and "error" in step["geometry"]:
            print(f"     ERROR: {step['geometry']['error']}")
    
    # Print final state if requested
    if args.print_final_state:
        print("\nFinal state predicates:")
        for pred_name, args in sorted(final_state):
            if args:
                print(f"  - {pred_name}({', '.join(str(a) for a in args)})")
            else:
                print(f"  - {pred_name}")


if __name__ == "__main__":
    main()
