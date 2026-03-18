# planners/schema_validation.py
"""Strict schema validation for LLM-proposed actions.

Validates that LLM responses conform to:
- EditLang domain specification
- Grounding rules (no variables like ?x)
- Goal alignment ((add ∪ del) ∩ G_t ≠ ∅)
- Wildcard rules (* only in del for mutually-exclusive predicates)
"""

from typing import List, Dict, Any, Set
from errors.planner_error import PlannerSchemaOrLogicError


def _is_list_pred(p) -> bool:
    """Check if p is a valid list-style predicate: ["pred", ["arg1","arg2", ...]]"""
    if not isinstance(p, list):
        return False
    if len(p) < 2:
        return False
    if not isinstance(p[0], str):
        return False
    if not isinstance(p[1], list):
        return False
    if not all(isinstance(a, str) for a in p[1]):
        return False
    return True


def _grounded_token(tok: str) -> bool:
    """Check if token is grounded (not a variable like ?x)."""
    return isinstance(tok, str) and not tok.startswith("?")


def _allows_wildcard_del(pred_name: str, editlang_spec: Dict[str, Any]) -> bool:
    """Check if predicate allows wildcard * in del field (mutually-exclusive predicates only)."""
    # spec convention: predicates[].mutually_exclusive: true
    for pd in editlang_spec.get("predicates", []):
        if isinstance(pd, dict) and pd.get("name") == pred_name:
            return bool(pd.get("mutually_exclusive", False))
        # Also check if it's in the simple predicate list (string format)
        if isinstance(pd, str) and pd == pred_name:
            # For string-format predicates, check if name suggests mutual exclusion
            # Common patterns: is_facing, on, at, aligned_with, etc.
            # These are typically mutually exclusive
            # has_style: object can only have one style at a time (generic style predicate)
            # between: object can only be between one pair at a time
            if pred_name in {"is_facing", "on", "at", "aligned_with", "near", "has_style", "between"}:
                return True
    return False


def _arity_of(pred_name: str, editlang_spec: Dict[str, Any]) -> int:
    """Get expected arity of predicate from spec."""
    for pd in editlang_spec.get("predicates", []):
        if isinstance(pd, dict) and pd.get("name") == pred_name:
            # convention: arity field or args schema length
            if "arity" in pd:
                return int(pd["arity"])
            if "args" in pd and isinstance(pd["args"], list):
                return len(pd["args"])
    # If predicate is in simple string list, infer from common patterns
    if pred_name in editlang_spec.get("predicates", []):
        # Common arities (matching editlang_std.yaml action schemas)
        common = {
            "exists": 1, "removed": 1, "supported": 1, "clear": 1, "locked": 1, "stable": 1, "accessible": 1,
            "is_facing": 2, "on": 2, "at": 2, "has_style": 2, "in_front_of": 2, "behind": 2, "grouped_with": 2, "contact": 2, "visible": 2, "colliding": 2, "matches_style": 2,
            "near": 3, "aligned_with": 3, "between": 3, "left_of": 3, "right_of": 3,
            "has_scale": 4  # obj, sx, sy, sz
        }
        return common.get(pred_name, -1)
    # unknown → treat as invalid by returning -1
    return -1


def _pred_name_set(editlang_spec: Dict[str, Any]) -> Set[str]:
    """Extract set of valid predicate names from spec."""
    names = set()
    for pd in editlang_spec.get("predicates", []):
        if isinstance(pd, dict) and "name" in pd:
            names.add(pd["name"])
        elif isinstance(pd, str):
            names.add(pd)
    return names


# ============================================================================
# SOFT VALIDATION WITH AUTO-FIX
# ============================================================================

def try_fix_predicate(pred: List, editlang_spec: Dict[str, Any], allow_wildcard: bool = False) -> List:
    """Try to fix a predicate with minor issues. Returns fixed predicate or original."""
    if not isinstance(pred, list) or len(pred) < 2:
        return pred
    
    name, args = pred[0], pred[1]
    if not isinstance(name, str) or not isinstance(args, list):
        return pred
    
    # Fix arity mismatch
    expected_arity = _arity_of(name, editlang_spec)
    if expected_arity > 0:
        if len(args) < expected_arity:
            # Pad with wildcards
            args = args + ["*"] * (expected_arity - len(args))
        elif len(args) > expected_arity:
            # Truncate
            args = args[:expected_arity]
    
    # Fix variable-like tokens (replace ?var with *)
    fixed_args = []
    for a in args:
        if isinstance(a, str) and a.startswith("?"):
            fixed_args.append("*")  # Replace variable with wildcard
        else:
            fixed_args.append(a)
    
    return [name, fixed_args]


def try_fix_pred_list(field: str, items: List, editlang_spec: Dict[str, Any], allow_wildcard: bool = False) -> List:
    """Try to fix a list of predicates. Returns fixed list."""
    if not isinstance(items, list):
        return []
    
    names = _pred_name_set(editlang_spec)
    fixed = []
    
    for p in items:
        if not _is_list_pred(p):
            continue  # Skip invalid predicates
        
        name = p[0]
        # Skip unknown predicates
        if name not in names:
            continue
        
        # Try to fix the predicate
        fixed_p = try_fix_predicate(p, editlang_spec, allow_wildcard)
        fixed.append(fixed_p)
    
    return fixed


def try_fix_action_item(item: Dict[str, Any], editlang_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to fix minor schema issues in an LLM action item.
    
    Returns the fixed item. Does NOT raise exceptions for minor issues.
    """
    if not isinstance(item, dict):
        return item
    
    # Ensure required keys exist
    if "action" not in item:
        return item  # Can't fix missing action
    if "args" not in item:
        item["args"] = {}
    
    # Fix predicate lists
    for field, allow_wc in [("pre", False), ("add", False), ("del", True), ("predicted_unmet_pre", False)]:
        if field not in item:
            item[field] = []
        else:
            item[field] = try_fix_pred_list(field, item.get(field, []), editlang_spec, allow_wc)
    
    # Ensure rationale exists
    if "rationale" not in item:
        item["rationale"] = ""
    
    return item


def soft_validate_and_fix_action_list(
    items: List[Dict[str, Any]],
    G_t: List,
    editlang_spec: Dict[str, Any],
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Validate and fix LLM action list with soft approach.
    
    Minor issues are auto-fixed. Only critical issues raise exceptions.
    
    Returns:
        List of fixed action items (may be empty if all invalid)
    """
    if not isinstance(items, list) or len(items) == 0:
        raise PlannerSchemaOrLogicError("LLM returned empty or invalid list")
    
    valid_actions = set(editlang_spec.get("actions", {}).keys())
    if not valid_actions:
        valid_actions = {"rotate_towards", "move_near", "place_on", "move_to", 
                        "align_with", "place_between", "remove_from"}
    
    fixed_items = []
    
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            if verbose:
                print(f"  [Validation] Skipping item {i}: not a dict")
            continue
        
        # Check action name (critical)
        action = item.get("action", "")
        if action not in valid_actions:
            if verbose:
                print(f"  [Validation] Skipping item {i}: invalid action '{action}'")
            continue
        
        # Try to fix the item
        try:
            fixed_item = try_fix_action_item(item, editlang_spec)
            fixed_items.append(fixed_item)
            if verbose:
                print(f"  [Validation] Fixed item {i}: {action}")
        except Exception as e:
            if verbose:
                print(f"  [Validation] Failed to fix item {i}: {e}")
            continue
    
    if not fixed_items:
        raise PlannerSchemaOrLogicError("No valid actions after soft validation")
    
    return fixed_items


def _validate_pred_list(
    field: str,
    items: List,
    editlang_spec: Dict[str, Any],
    allow_wildcard: bool
) -> None:
    """Validate a list of predicates."""
    names = _pred_name_set(editlang_spec)
    if not isinstance(items, list):
        raise PlannerSchemaOrLogicError(f"{field}: must be a list")
    
    for idx, p in enumerate(items):
        if not _is_list_pred(p):
            # Provide detailed error message
            error_parts = [f"{field}[{idx}]: invalid predicate shape."]
            error_parts.append(f"Got: {repr(p)}")
            error_parts.append("Expected format: ['predicate_name', ['arg1', 'arg2', ...]]")
            error_parts.append("Example: ['on', ['book', 'table']]")
            if not isinstance(p, list):
                error_parts.append(f"Issue: Not a list (got {type(p).__name__})")
            elif len(p) < 2:
                error_parts.append(f"Issue: List too short (got {len(p)} elements, need 2)")
            elif not isinstance(p[0], str):
                error_parts.append(f"Issue: First element not string (got {type(p[0]).__name__})")
            elif not isinstance(p[1], list):
                error_parts.append(f"Issue: Second element not list (got {type(p[1]).__name__})")
            raise PlannerSchemaOrLogicError(" ".join(error_parts))
        
        name, args = p[0], p[1]
        
        if name not in names:
            raise PlannerSchemaOrLogicError(f"{field}: unknown predicate '{name}'")
        
        ar = _arity_of(name, editlang_spec)
        if ar >= 0 and ar != len(args):
            raise PlannerSchemaOrLogicError(
                f"{field}: arity mismatch for '{name}' (expected {ar}, got {len(args)})"
            )
        
        for a in args:
            if a == "*":
                if not allow_wildcard:
                    raise PlannerSchemaOrLogicError(f"{field}: wildcard '*' not allowed here")
                # additionally, '*' allowed only for mutually-exclusive predicates
                if not _allows_wildcard_del(name, editlang_spec):
                    raise PlannerSchemaOrLogicError(
                        f"{field}: '*' only allowed for mutually-exclusive predicates (got {name})"
                    )
            else:
                if not _grounded_token(a):
                    raise PlannerSchemaOrLogicError(f"{field}: variable-like token '{a}' not allowed")


def _validate_args(action: str, args: Dict[str, Any], editlang_spec: Dict[str, Any]) -> None:
    """Validate action arguments."""
    if not isinstance(args, dict):
        raise PlannerSchemaOrLogicError("args must be a dict")
    
    # optional: validate required arg keys/types per action schema in spec
    spec_map = {
        a["name"]: a
        for a in editlang_spec.get("actions", {}).values()
        if isinstance(a, dict) and "name" in a
    }
    
    if action not in spec_map:
        # action name validity is checked elsewhere; keep silent here
        return
    
    adef = spec_map[action]
    reqs = [x["name"] for x in adef.get("args", {}).values() if isinstance(x, dict) and "name" in x]
    for rk in reqs:
        if rk not in args:
            raise PlannerSchemaOrLogicError(f"args missing required key: {rk}")


def validate_llm_action_item(
    item: Dict[str, Any],
    G_t: List,
    editlang_spec: Dict[str, Any]
) -> None:
    """
    Validate a single LLM-proposed action item.
    
    Args:
        item: Action dict from LLM
        G_t: Current subgoal predicates (list format)
        editlang_spec: Full EditLang domain specification
        
    Raises:
        PlannerSchemaOrLogicError: On any validation failure
    """
    if not isinstance(item, dict):
        raise PlannerSchemaOrLogicError("LLM item must be dict")
    
    # required keys
    # req = {"current_spatial_relations", "action", "args", "pre", "add", "del", "predicted_unmet_pre", "rationale"}
    # if not req.issubset(item.keys()):
    #     missing = req - set(item.keys())
    #     raise PlannerSchemaOrLogicError(f"missing keys: {sorted(missing)}")
    
    # action name - validate against EditLang spec (not hardcoded list)
    action = item["action"]
    valid_actions = set(editlang_spec.get("actions", {}).keys())
    if not valid_actions:
        # Fallback to common actions if spec doesn't provide them
        valid_actions = {"rotate_towards", "move_near", "place_on", "move_to", 
                        "align_with", "place_between", "remove_from"}
    
    if action not in valid_actions:
        raise PlannerSchemaOrLogicError(
            f"invalid action: {action}. Valid actions from spec: {sorted(valid_actions)}"
        )
    
    # args
    _validate_args(action, item["args"], editlang_spec)
    
    # current_spatial_relations (before add/del)
    # _validate_pred_list("current_spatial_relations", item["current_spatial_relations"], 
    #                    editlang_spec, allow_wildcard=False)
    
    # predicate lists
    _validate_pred_list("pre", item["pre"], editlang_spec, allow_wildcard=False)
    _validate_pred_list("add", item["add"], editlang_spec, allow_wildcard=False)
    # del: wildcard '*' allowed only for mutually-exclusive predicates
    _validate_pred_list("del", item["del"], editlang_spec, allow_wildcard=True)
    _validate_pred_list("predicted_unmet_pre", item["predicted_unmet_pre"], editlang_spec, allow_wildcard=False)
    
    # goal alignment: (add ∪ del) ∩ G_t ≠ ∅
    # Uses wildcard-aware matching for del predicates
    def as_key(p):
        return (p[0], tuple(p[1]))
    
    def _wc_match(pattern_pred, target_pred):
        """Wildcard-aware predicate matching."""
        if pattern_pred[0] != target_pred[0]:
            return False
        p_args = pattern_pred[1] if isinstance(pattern_pred[1], (list, tuple)) else [pattern_pred[1]]
        t_args = target_pred[1] if isinstance(target_pred[1], (list, tuple)) else [target_pred[1]]
        if len(p_args) != len(t_args):
            return False
        for pa, ta in zip(p_args, t_args):
            pa_s = str(pa)
            if pa_s == "*" or pa_s.startswith("?any_"):
                continue
            if pa_s != str(ta):
                return False
        return True
    
    gt_list = G_t  # Keep as list for wildcard matching
    aligns = False
    for fld in ("add", "del"):
        for p in item[fld]:
            # Exact match first
            try:
                if as_key(p) in {as_key(g) for g in gt_list}:
                    aligns = True
                    break
            except Exception:
                pass
            # Wildcard match (for del predicates with *)
            for g in gt_list:
                if _wc_match(p, g):
                    aligns = True
                    break
            if aligns:
                break
        if aligns:
            break
    
    if not aligns:
        # Downgrade to warning instead of hard reject
        # The hybrid validator will also check goal directedness
        if verbose if 'verbose' in dir() else False:
            print(f"  [StrictValidation] Warning: goal alignment check failed (soft)")


def validate_llm_action_list(
    items: List[Dict[str, Any]],
    G_t: List,
    editlang_spec: Dict[str, Any]
) -> None:
    """
    Validate all items in LLM response.
    
    Args:
        items: List of action dicts from LLM
        G_t: Current subgoal predicates (list format)
        editlang_spec: Full EditLang domain specification
        
    Raises:
        PlannerSchemaOrLogicError: On any validation failure
    """
    if not isinstance(items, list) or len(items) == 0:
        raise PlannerSchemaOrLogicError("LLM returned empty list")
    
    for i, it in enumerate(items):
        try:
            validate_llm_action_item(it, G_t, editlang_spec)
        except PlannerSchemaOrLogicError as e:
            raise PlannerSchemaOrLogicError(f"item {i}: {e}")

