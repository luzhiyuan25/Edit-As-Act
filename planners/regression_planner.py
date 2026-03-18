"""Goal Regression Planner for EditLang.

Implements backward chaining from goal to initial state using regression.
It integrates:
1. LLM-driven action proposal (K candidates per step)
2. Hybrid semantic validation (deterministic + soft LLM)
3. Geometric validation (GeomChecker)
4. Feedback loop (rejection reasons fed back to LLM)
"""

from typing import List, Set, Tuple, Optional, Dict, Any
import uuid

from editors.editlang import EditLangDomain, Action, Predicate, instantiate_action
from errors.planner_error import PlannerSchemaOrLogicError
from utils.logging_utils import write_planner_log
from planners.schema_validation import validate_llm_action_list, soft_validate_and_fix_action_list


class RegressionPlanner:
    """Backward chaining planner using goal regression."""
    
    def __init__(
        self,
        domain: EditLangDomain,
        scene_data: Optional[Dict[str, Any]] = None,
        llm_helper: Optional[Any] = None,
        llm_validator: Optional[Any] = None,
        geom_checker: Optional[Any] = None, 
        skip_schema_validation: bool = False,
        max_steps: int = 64,
        verbose: bool = False
    ):
        self.domain = domain
        self.scene_data = scene_data
        self.llm_helper = llm_helper
        self.llm_validator = llm_validator
        self.geom_checker = geom_checker
        self.skip_schema_validation = skip_schema_validation
        self.max_steps = max_steps
        self.verbose = verbose
        self.backward_history = []
        self.run_id = None
        self.K = 3  # Request top-3 actions from LLM per step
    
    def _log_step(self, **kwargs):
        rec = {"stage": "planner", "run_id": self.run_id, **kwargs}
        try:
            write_planner_log(rec)
        except Exception:
            pass
    
    def plan(
        self,
        s0: Set[Predicate],
        G: Set[Predicate],
        instruction_raw: str = "",
        G_terminal: Optional[Set[Predicate]] = None
    ) -> List[Action]:
        """Generate plan from initial state s0 to achieve goal G.
        
        Uses backward chaining: starts from goal G and regresses until
        all goals are satisfied in s0.
        """
        plan_rev: List[Action] = []
        Gt = set(G)
        self.run_id = str(uuid.uuid4())
        self.backward_history = []
        
        self.S0_full = s0
        self.G_terminal = G_terminal if G_terminal is not None else G
        self.instruction_raw = instruction_raw
        
        # Reset validator's satisfied goals tracking
        if self.llm_validator and hasattr(self.llm_validator, 'reset_satisfied_goals'):
            self.llm_validator.reset_satisfied_goals()
        
        visited_snapshots = set()
        # Track rejection reasons for feedback loop
        rejection_feedback: List[str] = []
        # Track no-progress steps to avoid infinite loops
        no_progress_count = 0
        MAX_NO_PROGRESS = 3
        
        for step in range(self.max_steps):
            # 1. Check Termination (Goal Empty)
            if not Gt:
                if self.verbose: print(f"[Planner] Goal set empty at step {step}. Planning complete.")
                break
            
            # 2. Check Termination (Goal Satisfied in S0)
            if Gt.issubset(s0):
                if self.verbose: print(f"[Planner] Goal satisfied in S0 at step {step}. Planning complete.")
                break
            
            # 3. Cycle Detection (relaxed — warn but don't crash)
            snap = frozenset(Gt)
            if snap in visited_snapshots:
                if self.verbose:
                    print(f"[Planner] Warning: Goal set revisited at step {step} (potential cycle)")
                self._log_step(goal=[as_list(p) for p in Gt], reason="cycle warning")
                no_progress_count += 1
                if no_progress_count >= MAX_NO_PROGRESS:
                    if self.verbose:
                        print(f"[Planner] Too many no-progress steps ({no_progress_count}), stopping.")
                    break
            visited_snapshots.add(snap)
            
            # 4. Action Proposal & Validation Loop
            if not (self.llm_helper and hasattr(self.llm_helper, 'propose_transition_actions')):
                raise RuntimeError("LLM helper is required for this planner.")

            MAX_RETRIES = 5
            action_accepted = False

            for attempt in range(MAX_RETRIES):
                # A. LLM Proposal (request K candidates)
                try:
                    items = self.llm_helper.propose_transition_actions(
                        instruction_raw=self.instruction_raw,
                        G_terminal=[as_list(p) for p in self.G_terminal],
                        G_t=[as_list(p) for p in Gt],
                        backward_history=self.backward_history,
                        S0_full=[as_list(p) for p in self.S0_full],
                        editlang_spec=self.domain.to_dict(),
                        K=self.K,
                        rejection_feedback=rejection_feedback  # Feed back rejection reasons
                    )
                except TypeError:
                    # Fallback if propose_transition_actions doesn't accept rejection_feedback
                    items = self.llm_helper.propose_transition_actions(
                        instruction_raw=self.instruction_raw,
                        G_terminal=[as_list(p) for p in self.G_terminal],
                        G_t=[as_list(p) for p in Gt],
                        backward_history=self.backward_history,
                        S0_full=[as_list(p) for p in self.S0_full],
                        editlang_spec=self.domain.to_dict(),
                        K=self.K
                    )
                except Exception as e:
                    self._log_step(reason=f"LLM failure: {e}")
                    raise

                if not items:
                    rejection_feedback.append("LLM returned no candidates. Try different action.")
                    continue

                # B. Schema Validation (with soft auto-fix)
                if not self.skip_schema_validation:
                    try:
                        items = soft_validate_and_fix_action_list(
                            items, 
                            [as_list(p) for p in Gt], 
                            self.domain.to_dict(),
                            verbose=self.verbose
                        )
                    except PlannerSchemaOrLogicError as e:
                        self._log_step(reason=f"Schema validation failed: {e}")
                        if self.verbose:
                            print(f"[Planner] Schema validation failed: {e}")
                        rejection_feedback.append(f"Schema error: {e}")
                        continue
                
                if not items:
                    rejection_feedback.append("All candidates failed schema validation.")
                    continue
                
                # C. Iterate through ALL K candidates
                candidate_accepted = False
                for candidate_idx, a_dict in enumerate(items):
                    # Instantiate Action Object
                    try:
                        schema = self.domain.actions.get(a_dict['action'])
                        if not schema:
                            if self.verbose: print(f"[Planner] Unknown action: {a_dict['action']}")
                            continue
                        action_obj = instantiate_action(schema, a_dict['args'])
                    except Exception as e:
                        if self.verbose: print(f"[Planner] Action instantiation failed: {e}")
                        continue

                    # --- VALIDATION PIPELINE ---
                    
                    # D. Geometric Validation (GeomChecker)
                    if self.geom_checker:
                        if not self.geom_checker.feasible(action_obj):
                            if self.verbose: 
                                print(f"[GeomChecker] Action {action_obj.name} (candidate {candidate_idx}) "
                                      f"rejected due to collision/support.")
                            self._log_step(reason=f"GeomChecker rejected candidate {candidate_idx}")
                            rejection_feedback.append(
                                f"Action {a_dict['action']}({a_dict.get('args',{})}) rejected by "
                                f"geometry check (collision/support)."
                            )
                            continue

                    # E. Semantic Validation (Hybrid Validator)
                    if self.llm_validator:
                        strict_G_next, _ = regress_strict(Gt, a_dict, s0)
                        
                        k_tail = getattr(self.llm_validator, "k_tail", 3)
                        timeout = getattr(self.llm_validator, "timeout_ms", 4000)
                        plan_tail = self.backward_history[-k_tail:] if len(self.backward_history) > k_tail else self.backward_history
                        
                        try:
                            ok, reasons, _ = self.llm_validator.check_regression_step(
                                plan_rev=plan_tail,
                                a=a_dict,
                                G_t=[as_list(p) for p in Gt],
                                G_next=strict_G_next,
                                S0_full=[as_list(p) for p in self.S0_full],
                                instruction_raw=self.instruction_raw,
                                timeout_ms=timeout
                            )
                        except PlannerSchemaOrLogicError as e:
                            # Validator parsing error — accept action anyway
                            if self.verbose:
                                print(f"[Validator] Error (non-fatal, accepting action): {e}")
                            ok = True
                            reasons = [f"Validator error (ignored): {e}"]
                        
                        if not ok:
                            if self.verbose: 
                                print(f"[Validator] Rejected candidate {candidate_idx}: {reasons}")
                            rejection_feedback.append(
                                f"Validator rejected {a_dict['action']}: {'; '.join(reasons)}"
                            )
                            continue

                    # --- COMMIT ACTION ---
                    strict_G_next, pre_unmet = regress_strict(Gt, a_dict, s0)
                    G_next_set = {as_key(p) for p in strict_G_next}
                    
                    # Record satisfied goals for monotonicity tracking
                    newly_satisfied = Gt - G_next_set
                    if self.llm_validator and hasattr(self.llm_validator, 'record_satisfied'):
                        self.llm_validator.record_satisfied([as_list(p) for p in newly_satisfied])
                    
                    plan_rev.append(action_obj)
                    self.backward_history.append({
                        "run_id": self.run_id,
                        "chosen_action": a_dict,
                        "pre_unmet": pre_unmet,
                        "strict_G_next": strict_G_next
                    })
                    
                    # Update goal via strict regression
                    if Gt != G_next_set:
                        Gt = G_next_set
                        no_progress_count = 0  # Reset no-progress counter
                    else:
                        # No strict progress — don't force-exit, but track
                        no_progress_count += 1
                        if self.verbose:
                            print(f"[Planner] No strict progress at step {step} "
                                  f"({no_progress_count}/{MAX_NO_PROGRESS})")
                        if no_progress_count >= MAX_NO_PROGRESS:
                            if self.verbose:
                                print(f"[Planner] Reached max no-progress steps. Accepting partial plan.")
                            Gt = set()  # Only force-exit after multiple no-progress steps
                    
                    candidate_accepted = True
                    action_accepted = True
                    # Clear rejection feedback on success
                    rejection_feedback = []
                    self._log_step(
                        chosen_action=a_dict['action'], 
                        candidate_idx=candidate_idx,
                        delta=len(newly_satisfied)
                    )
                    break  # Exit candidate loop
                
                if candidate_accepted:
                    break  # Exit retry loop
            
            if not action_accepted:
                if self.verbose:
                    print(f"[Planner] Could not find valid action at step {step} after "
                          f"{MAX_RETRIES} retries × {self.K} candidates. Returning partial plan.")
                break
        
        # Return plan in forward order
        return list(reversed(plan_rev))


# --- Helper Utilities ---

def as_key(p):
    """Convert a predicate to a hashable key (tuple form)."""
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        name = p[0]
        args = p[1]
        if isinstance(args, (list, tuple)):
            return (name, tuple(str(a) for a in args))
    if isinstance(p, (list, tuple)) and len(p) == 2:
        return (str(p[0]), tuple(str(a) for a in p[1]) if isinstance(p[1], (list, tuple)) else p[1])
    return p

def as_list(k):
    """Convert a hashable predicate key back to list form."""
    if isinstance(k, tuple) and len(k) == 2 and isinstance(k[1], tuple):
        return [k[0], list(k[1])]
    return k


def _del_matches_goal(del_pred, goal_pred) -> bool:
    """Check if a del predicate matches a goal predicate (with wildcard support).
    
    Wildcards ('*', '?any_*') in del match any token in goal.
    """
    if not isinstance(del_pred, tuple) or not isinstance(goal_pred, tuple):
        return del_pred == goal_pred
    
    if len(del_pred) != 2 or len(goal_pred) != 2:
        return del_pred == goal_pred
    
    d_name, d_args = del_pred
    g_name, g_args = goal_pred
    
    if str(d_name) != str(g_name):
        return False
    
    if not isinstance(d_args, tuple) or not isinstance(g_args, tuple):
        return str(d_args) == str(g_args)
    
    if len(d_args) != len(g_args):
        return False
    
    for da, ga in zip(d_args, g_args):
        da_str = str(da)
        if da_str == "*" or da_str.startswith("?any_"):
            continue
        if da_str != str(ga):
            return False
    
    return True


def regress_strict(Gt, action, S0_full):
    """Compute strict source-aware regression.
    
    Per paper Eq. 2:
        G_{t-1} = (G_t \\ add(a)) ∪ (pre(a) \\ S_0)
    
    Also handles del effects: predicates in G_t that match del(a)
    are also removed (the action explicitly removes them).
    """
    Gk = {as_key(p) for p in Gt}
    
    if isinstance(action, dict):
        prek = {as_key(p) for p in action.get("pre", [])}
        addk = {as_key(p) for p in action.get("add", [])}
        delk = {as_key(p) for p in action.get("del", [])}
    else:
        prek = {as_key(p) for p in getattr(action, "pre", [])}
        addk = {as_key(p) for p in getattr(action, "add", [])}
        delk = {as_key(p) for p in getattr(action, "del", set())}
    
    s0k = {as_key(p) for p in S0_full}
    
    # Find goal predicates matched by del effects (with wildcard support)
    del_matched = set()
    for dk in delk:
        for gk in Gk:
            if _del_matches_goal(dk, gk):
                del_matched.add(gk)
    
    # Find goal predicates matched by add effects (with soft matching)
    # This handles cases like add: near(lamp, sofa, "default") matching goal: near(lamp, sofa, "0.5")
    # Also handles at(obj, x, y, z) vs at(obj, pos) arity differences
    add_matched = set()
    for ak in addk:
        # Exact match first
        if ak in Gk:
            add_matched.add(ak)
            continue
        # Soft match: same predicate name + first N-1 args match (distance/pos tolerance)
        if isinstance(ak, tuple) and len(ak) == 2:
            a_name, a_args = ak
            if isinstance(a_args, tuple):
                for gk in Gk:
                    if isinstance(gk, tuple) and len(gk) == 2:
                        g_name, g_args = gk
                        if str(a_name) == str(g_name) and isinstance(g_args, tuple):
                            # Same predicate name — check core args match
                            min_len = min(len(a_args), len(g_args))
                            if min_len >= 2:
                                # Core args (first 2) must match, rest (distance etc.) are soft
                                core_match = all(
                                    str(a) == str(g) or str(a) == "*" or str(g) == "*"
                                    for a, g in zip(a_args[:2], g_args[:2])
                                )
                                if core_match:
                                    add_matched.add(gk)
    
    # Derive pre_unmet: preconditions not satisfied in S0
    # Special handling for 'exists' — check if object appears in ANY predicate in S0
    objects_in_s0 = set()
    for pred in s0k:
        if isinstance(pred, tuple) and len(pred) >= 2 and isinstance(pred[1], tuple):
            objects_in_s0.update(pred[1])
    
    pre_unmet_k = set()
    for p in prek:
        if isinstance(p, tuple) and len(p) == 2 and p[0] == "exists":
            args = p[1]
            if isinstance(args, tuple) and len(args) == 1:
                obj = args[0]
                if obj not in objects_in_s0:
                    pre_unmet_k.add(p)
            else:
                if p not in s0k: pre_unmet_k.add(p)
        else:
            if p not in s0k: pre_unmet_k.add(p)
    
    # Regression formula: G_{t-1} = (G_t \ add_matched \ del_matched) ∪ (pre(a) \ S_0)
    G_next_k = (Gk - add_matched - del_matched) | pre_unmet_k

    G_next = [as_list(p) for p in sorted(G_next_k, key=str)]
    pre_unmet = [as_list(p) for p in sorted(pre_unmet_k, key=str)]
    return G_next, pre_unmet