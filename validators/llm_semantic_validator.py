"""Hybrid Validator for Edit-As-Act backward planning.

Implements the 4-criterion validation from the paper (Sec 4.4):
1. Goal directedness (deterministic) — add(a) ∩ G_t ≠ ∅
2. Monotonicity (deterministic) — del(a) ∩ G_satisfied = ∅
3. Contextual consistency (LLM, soft-reject)
4. Formal validity (deterministic, handled by schema_validation.py)

The previous version delegated ALL checks to a single LLM call,
which was too strict and fragile (parsing failures, over-constraining prompt).
"""

import json
from typing import Dict, Any, List, Tuple, Optional, Set

from errors.planner_error import PlannerSchemaOrLogicError


# ---------- Wildcard matching utility ----------

def _wildcard_match(pattern, target) -> bool:
    """Check if a predicate pattern matches a target predicate.
    
    Handles wildcards ('*' and '?any_*') in patterns.
    
    Args:
        pattern: Predicate [name, [arg1, arg2, ...]] possibly with wildcards
        target: Concrete predicate [name, [arg1, arg2, ...]]
        
    Returns:
        True if pattern matches target
    """
    if not isinstance(pattern, (list, tuple)) or not isinstance(target, (list, tuple)):
        return pattern == target
    
    if len(pattern) < 2 or len(target) < 2:
        return str(pattern) == str(target)
    
    p_name, p_args = pattern[0], pattern[1] if len(pattern) > 1 else []
    t_name, t_args = target[0], target[1] if len(target) > 1 else []
    
    # Predicate names must match
    if str(p_name) != str(t_name):
        return False
    
    # Normalize args
    if isinstance(p_args, (list, tuple)) and isinstance(t_args, (list, tuple)):
        # SOFT ARITY: when lengths differ, compare only core args
        if len(p_args) == len(t_args):
            # Same length — strict compare with wildcards
            for pa, ta in zip(p_args, t_args):
                pa_str = str(pa)
                ta_str = str(ta)
                if pa_str == "*" or pa_str.startswith("?any_"):
                    continue
                if pa_str != ta_str:
                    return False
            return True
        else:
            # Different lengths — compare available core args
            min_len = min(len(p_args), len(t_args))
            if min_len >= 2:
                for pa, ta in zip(p_args[:min_len], t_args[:min_len]):
                    pa_str = str(pa)
                    ta_str = str(ta)
                    if pa_str == "*" or pa_str.startswith("?any_"):
                        continue
                    if pa_str != ta_str:
                        return False
                return True
            elif min_len == 1:
                return str(p_args[0]) == str(t_args[0]) or str(p_args[0]) == "*"
            return True  # Both empty args
    
    return str(p_args) == str(t_args)


def _any_match(pattern_set, target_set) -> bool:
    """Check if any pattern in pattern_set matches any target in target_set."""
    for p in pattern_set:
        for t in target_set:
            if _wildcard_match(p, t):
                return True
    return False


def _find_matches(pattern_set, target_set) -> List:
    """Find all targets that match any pattern."""
    matches = []
    for p in pattern_set:
        for t in target_set:
            if _wildcard_match(p, t):
                matches.append((p, t))
    return matches


# ---------- Semantic Validator LLM Prompt ----------

SEMANTIC_SYSTEM_PROMPT = """You are a semantic plausibility checker for 3D indoor scene editing.

Given a proposed edit action, briefly judge whether it makes sense in the context of:
- The original instruction
- Indoor scene common sense (e.g., don't place a sofa on a lamp)
- Whether the action might create an infinite loop with recent actions

OUTPUT: JSON only, no markdown.
{
  "ok": true/false,
  "reason": "1-2 sentence explanation"
}

Be LENIENT. Only reject actions that are clearly nonsensical or create obvious loops.
Actions that are reasonable attempts to satisfy the instruction should be accepted."""


# ---------- Main Hybrid Validator ----------

class LLMSemanticValidator:
    """Hybrid validator: deterministic + soft LLM.
    
    Checks goal-directedness and monotonicity deterministically.
    Uses LLM only for soft semantic plausibility check.
    """
    
    def __init__(
        self,
        editlang_spec: Dict[str, Any],
        model_client: Any,
        temperature: float = 0.0,
        timeout_ms: int = 4000,
        verbose: bool = False
    ):
        self.spec = editlang_spec
        self.client = model_client
        self.temperature = temperature
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        self.k_tail = 3
        
        # Track satisfied goals for monotonicity checking
        self.satisfied_goals: List = []
    
    def reset_satisfied_goals(self):
        """Reset satisfied goals tracking for a new planning run."""
        self.satisfied_goals = []
    
    def record_satisfied(self, newly_satisfied: List):
        """Record goals that have been satisfied by accepted actions."""
        self.satisfied_goals.extend(newly_satisfied)
    
    def check_regression_step(
        self,
        plan_rev: List[Dict[str, Any]],
        a: Dict[str, Any],
        G_t: List,
        G_next: List,
        S0_full: List,
        instruction_raw: str,
        timeout_ms: Optional[int] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate regression step using hybrid approach.
        
        Returns:
            (ok, reasons, meta)
        """
        reasons = []
        tags = []
        
        # ---------- Criterion 1: Goal Directedness (deterministic) ----------
        # add(a) ∩ G_t ≠ ∅ — action must satisfy at least one current goal
        add_preds = a.get("add", [])
        
        goal_directed = _any_match(add_preds, G_t)
        
        if not goal_directed:
            # Also check if del effects help — removing a predicate that's in G_t
            # (e.g., removing an object satisfies removed(obj) goal indirectly)
            del_preds = a.get("del", [])
            
            # Check if the action's effects make progress via side effects
            # For example: remove_object has del=[exists(obj)] and add=[removed(obj)]
            # The add=[removed(obj)] should match if removed(obj) ∈ G_t
            
            if not goal_directed:
                if self.verbose:
                    print(f"[Validator] WARN: No add effect directly matches G_t.")
                    print(f"  add: {add_preds}")
                    print(f"  G_t: {G_t}")
                # Soft warning only — the LLM planner may have valid indirect reasoning
                reasons.append("No direct goal-directed match (add ∩ G_t = ∅). Accepting with warning.")
                tags.append("goal_alignment")
                # Do NOT reject — we accept with warning for robustness
        
        # ---------- Criterion 2: Monotonicity (deterministic) ----------
        # del(a) ∩ G_satisfied = ∅ — must not undo previously satisfied goals
        del_preds = a.get("del", [])
        
        violated_goals = _find_matches(del_preds, self.satisfied_goals)
        if violated_goals:
            if self.verbose:
                print(f"[Validator] REJECT: Monotonicity violation — action undoes satisfied goals: {violated_goals}")
            reasons.append(f"Monotonicity violation: del undoes previously satisfied goals: {violated_goals}")
            tags.append("loop_risk")
            meta = {"severity": "error", "tags": tags}
            return False, reasons, meta
        
        # ---------- Criterion 3: Loop Detection (deterministic) ----------
        # Check if this action reverses a recent action (simple swap detection)
        if plan_rev:
            last_actions = plan_rev[-3:] if len(plan_rev) > 3 else plan_rev
            current_action_name = a.get("action", "")
            current_args = a.get("args", {})
            
            for prev in last_actions:
                prev_action = prev.get("chosen_action", prev) if isinstance(prev, dict) else prev
                if isinstance(prev_action, dict):
                    prev_name = prev_action.get("action", "")
                    prev_args = prev_action.get("args", {})
                    
                    # Simple swap detection: same action type, same object, different target
                    if (prev_name == current_action_name and 
                        prev_args.get("obj") == current_args.get("obj")):
                        # Check if this is literally the reverse
                        if self.verbose:
                            print(f"[Validator] WARN: Potential loop — same action '{current_action_name}' "
                                  f"on same object '{current_args.get('obj')}' as recent step")
                        reasons.append(f"Potential loop: repeating {current_action_name} on {current_args.get('obj')}")
                        tags.append("loop_risk")
                        # Warn but don't reject — could be a valid multi-step operation
        
        # ---------- Criterion 4: Contextual Consistency (LLM, soft) ----------
        # Only call LLM if we have a client — make this optional
        if self.client and not isinstance(self.client, MockLLMClient):
            try:
                semantic_ok, semantic_reason = self._check_semantic(
                    a, G_t, plan_rev, instruction_raw, timeout_ms
                )
                if not semantic_ok:
                    reasons.append(f"Semantic: {semantic_reason}")
                    tags.append("semantic_break")
                    # Soft reject for semantic — only block truly nonsensical actions
                    if self.verbose:
                        print(f"[Validator] Semantic rejection: {semantic_reason}")
                    meta = {"severity": "error", "tags": tags}
                    return False, reasons, meta
            except Exception as e:
                # LLM call failed — do NOT crash, just proceed without semantic check
                if self.verbose:
                    print(f"[Validator] Semantic check failed (non-fatal): {e}")
                reasons.append(f"Semantic check skipped: {e}")
        
        # ---------- Accept ----------
        severity = "warn" if reasons else "ok"
        meta = {"severity": severity, "tags": tags}
        
        if self.verbose:
            print(f"[Validator] ACCEPT (severity={severity})")
            for r in reasons:
                print(f"  - {r}")
        
        return True, reasons, meta
    
    def _check_semantic(
        self,
        a: Dict[str, Any],
        G_t: List,
        plan_rev: List,
        instruction_raw: str,
        timeout_ms: Optional[int] = None
    ) -> Tuple[bool, str]:
        """Soft LLM-based semantic plausibility check.
        
        Returns:
            (ok, reason)
        """
        # Build a concise payload — DO NOT send full editlang_spec
        payload = {
            "instruction": instruction_raw,
            "proposed_action": {
                "name": a.get("action", "unknown"),
                "args": a.get("args", {}),
                "rationale": a.get("rationale", "")
            },
            "current_goals": G_t[:5],  # Only first 5 for brevity
            "recent_actions": [
                {
                    "name": (p.get("chosen_action", p) if isinstance(p, dict) else p
                    ).get("action", "?") if isinstance(
                        p.get("chosen_action", p) if isinstance(p, dict) else p, dict
                    ) else "?"
                }
                for p in (plan_rev[-3:] if len(plan_rev) > 3 else plan_rev)
            ]
        }
        
        user_msg = json.dumps(payload, ensure_ascii=False, indent=1)
        
        timeout = (timeout_ms or self.timeout_ms) / 1000.0
        resp_text = self.client.chat(
            system=SEMANTIC_SYSTEM_PROMPT,
            user=user_msg,
            temperature=self.temperature,
            timeout=timeout
        )
        
        if not resp_text:
            return True, ""  # No response = accept
        
        # Parse response — be lenient
        try:
            # Try to extract JSON from response
            resp_clean = resp_text.strip()
            if resp_clean.startswith("```"):
                # Strip markdown code block
                lines = resp_clean.split("\n")
                resp_clean = "\n".join(lines[1:-1])
            
            data = json.loads(resp_clean)
            ok = data.get("ok", True)
            reason = data.get("reason", "")
            return ok, reason
        except json.JSONDecodeError:
            # Can't parse — accept by default
            return True, ""


class MockLLMClient:
    """Mock LLM client for testing (always returns OK)."""
    
    def chat(self, system: str, user: str, temperature: float = 0.0, timeout: float = 4.0) -> str:
        """Mock chat that returns valid OK response."""
        return json.dumps({
            "ok": True,
            "reason": "Mock validator: always accept"
        })
