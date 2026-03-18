#!/usr/bin/env python3
"""
Apply EditLang plans to a scene layout JSON and save per-instruction results.

Reads:
  - Original scene_layout_edited.json (mask-key format)
  - benchmark_bedroom_results.json  (plan with semantic IDs)

Outputs:
  - scene_layout_instruction_{index}_{command}.json  (same format as original)

Usage:
  python3 tools/apply_plan_to_scene.py \
    --scene  dataset/dataset/bedroom/scene_layout_edited.json \
    --plans  tests/benchmark_bedroom_results.json \
    --outdir dataset/dataset/bedroom/plans_applied
"""

import json, copy, math, os, argparse
from typing import Dict, Any, Optional, List, Tuple


# ─────────────────────────────────────────────────────────────
#  ID Mapping:  scene_mask_009_armchairs.png  ↔  armchairs_009
# ─────────────────────────────────────────────────────────────
def build_id_maps(scene: Dict[str, Any]):
    """Build bidirectional maps: mask_key ↔ semantic_id."""
    mask_to_sem = {}
    sem_to_mask = {}
    for key in scene:
        if key.startswith("scene_mask_") and key.endswith(".png"):
            if key == "scene_mask_RoomContainer.png":
                mask_to_sem[key] = "RoomContainer"
                sem_to_mask["RoomContainer"] = key
                continue
            parts = key.replace("scene_mask_", "").replace(".png", "").split("_")
            if len(parts) >= 2:
                obj_num = parts[0]
                category = "_".join(parts[1:])
                sem_id = f"{category}_{obj_num}"
                mask_to_sem[key] = sem_id
                sem_to_mask[sem_id] = key
    return mask_to_sem, sem_to_mask


def sem_to_mask_key(sem_id: str, sem_to_mask: Dict[str, str]) -> Optional[str]:
    """Look up the mask key for a semantic ID."""
    return sem_to_mask.get(sem_id)


def next_mask_index(scene: Dict[str, Any]) -> int:
    """Find the next available mask index number."""
    max_idx = -1
    for key in scene:
        if key.startswith("scene_mask_") and key.endswith(".png"):
            parts = key.replace("scene_mask_", "").replace(".png", "").split("_")
            if parts and parts[0].isdigit():
                max_idx = max(max_idx, int(parts[0]))
    return max_idx + 1


# ─────────────────────────────────────────────────────────────
#  Action Applicators
# ─────────────────────────────────────────────────────────────
def apply_action(scene: Dict, action: Dict, sem_to_mask: Dict, mask_to_sem: Dict) -> str:
    """Apply a single action to scene (mutates in place). Returns a log message."""
    name = action["action"]
    args = action.get("args", {})

    if name == "remove_object":
        return _apply_remove(scene, args, sem_to_mask, mask_to_sem)
    elif name == "add_object":
        return _apply_add(scene, args, sem_to_mask, mask_to_sem)
    elif name == "move_to":
        return _apply_move_to(scene, args, sem_to_mask)
    elif name == "place_relative":
        return _apply_place_relative(scene, args, sem_to_mask)
    elif name == "place_on":
        return _apply_place_on(scene, args, sem_to_mask)
    elif name == "place_between":
        return _apply_place_between(scene, args, sem_to_mask)
    elif name == "rotate_towards":
        return _apply_rotate_towards(scene, args, sem_to_mask)
    elif name == "rotate_by":
        return _apply_rotate_by(scene, args, sem_to_mask)
    elif name == "scale":
        return _apply_scale(scene, args, sem_to_mask)
    elif name == "align_with":
        return _apply_align_with(scene, args, sem_to_mask)
    elif name == "stylize":
        return _apply_stylize(scene, args, sem_to_mask)
    elif name == "move_group":
        return _apply_move_to(scene, args, sem_to_mask)  # same as move_to for layout
    else:
        return f"  [SKIP] Unknown action: {name}"


def _resolve(sem_id: str, sem_to_mask: Dict) -> Optional[str]:
    """Resolve semantic ID to mask key, or None."""
    return sem_to_mask.get(sem_id)


def _apply_remove(scene, args, sem_to_mask, mask_to_sem):
    obj = args.get("obj", "")
    mask_key = _resolve(obj, sem_to_mask)
    if mask_key and mask_key in scene:
        del scene[mask_key]
        del sem_to_mask[obj]
        if mask_key in mask_to_sem:
            del mask_to_sem[mask_key]
        return f"  remove_object({obj}) → deleted {mask_key}"
    return f"  remove_object({obj}) → NOT FOUND (skipped)"


def _apply_add(scene, args, sem_to_mask, mask_to_sem):
    obj = args.get("obj", "")
    cat = args.get("cat", "object")
    support = args.get("support", "")

    if obj in sem_to_mask:
        return f"  add_object({obj}) → already exists (skipped)"

    # Find support position
    support_key = _resolve(support, sem_to_mask)
    if support_key and support_key in scene:
        sup = scene[support_key]
        sup_center = sup.get("center", [0, 0, 0])
        sup_dim = sup.get("dim", [0.1, 0.1, 0.1])
        # Place on top of support
        new_center = [sup_center[0], sup_center[1] + sup_dim[1]/2 + 0.05, sup_center[2]]
    else:
        # Default placement at room center
        new_center = [0.0, 0.3, 0.0]

    # Generate mask key
    idx = next_mask_index(scene)
    # Clean category name for mask key
    cat_clean = cat.replace(" ", "_").lower()
    mask_key = f"scene_mask_{idx:03d}_{cat_clean}.png"

    # Default dimensions for new object
    default_dim = [0.05, 0.05, 0.05]

    scene[mask_key] = {
        "dim": default_dim,
        "center": new_center,
        "bbox": [
            new_center[0] - default_dim[0]/2,
            new_center[0] + default_dim[0]/2,
            new_center[1] - default_dim[1]/2,
            new_center[1] + default_dim[1]/2,
            new_center[2] - default_dim[2]/2,
            new_center[2] + default_dim[2]/2,
        ],
        "pose": 0.0,
        "front_face": "MIN_Y",
        "on_floor": False,
        "on_wall": None,
        "_added_by_plan": True,
        "_semantic_id": obj,
        "_category": cat,
    }
    sem_to_mask[obj] = mask_key
    mask_to_sem[mask_key] = obj
    return f"  add_object({obj}, cat={cat}, support={support}) → {mask_key}"


def _apply_move_to(scene, args, sem_to_mask):
    obj = args.get("obj", args.get("parent", ""))
    pos = args.get("pos", "")
    mask_key = _resolve(obj, sem_to_mask)
    if not mask_key or mask_key not in scene:
        return f"  move_to({obj}) → NOT FOUND"

    # Parse position (could be "x,y,z" or a single string)
    try:
        if isinstance(pos, (list, tuple)):
            coords = [float(p) for p in pos]
        elif "," in str(pos):
            coords = [float(p) for p in str(pos).split(",")]
        else:
            # Single value — can't determine 3D position, skip
            return f"  move_to({obj}, pos={pos}) → ambiguous position (skipped)"
        scene[mask_key]["center"] = coords[:3]
        _recompute_bbox(scene[mask_key])
        return f"  move_to({obj}) → center={coords[:3]}"
    except (ValueError, IndexError):
        return f"  move_to({obj}, pos={pos}) → parse error (skipped)"


def _apply_place_relative(scene, args, sem_to_mask):
    obj = args.get("obj", "")
    target = args.get("target", "")
    relation = args.get("relation", "near")

    obj_key = _resolve(obj, sem_to_mask)
    tgt_key = _resolve(target, sem_to_mask)

    if not obj_key or obj_key not in scene:
        return f"  place_relative({obj}) → obj NOT FOUND"
    if not tgt_key or tgt_key not in scene:
        return f"  place_relative({obj}, target={target}) → target NOT FOUND"

    t_center = scene[tgt_key]["center"]
    t_dim = scene[tgt_key].get("dim", [0.1, 0.1, 0.1])
    o_dim = scene[obj_key].get("dim", [0.1, 0.1, 0.1])
    offset = 0.15

    new_center = list(t_center)
    if relation == "left_of":
        new_center[0] = t_center[0] - t_dim[0]/2 - o_dim[0]/2 - offset
    elif relation == "right_of":
        new_center[0] = t_center[0] + t_dim[0]/2 + o_dim[0]/2 + offset
    elif relation == "in_front_of":
        new_center[2] = t_center[2] - t_dim[2]/2 - o_dim[2]/2 - offset
    elif relation == "behind":
        new_center[2] = t_center[2] + t_dim[2]/2 + o_dim[2]/2 + offset
    else:  # "near" or default
        new_center[0] = t_center[0] + t_dim[0]/2 + o_dim[0]/2 + offset

    scene[obj_key]["center"] = new_center
    _recompute_bbox(scene[obj_key])
    return f"  place_relative({obj}, {target}, {relation}) → center={[round(c,3) for c in new_center]}"


def _apply_place_on(scene, args, sem_to_mask):
    obj = args.get("obj", "")
    surface = args.get("surface", "")
    obj_key = _resolve(obj, sem_to_mask)
    srf_key = _resolve(surface, sem_to_mask)

    if not obj_key or obj_key not in scene:
        return f"  place_on({obj}) → NOT FOUND"
    if not srf_key or srf_key not in scene:
        return f"  place_on({obj}, surface={surface}) → surface NOT FOUND"

    s = scene[srf_key]
    o = scene[obj_key]
    new_center = [s["center"][0], s["center"][1] + s["dim"][1]/2 + o["dim"][1]/2, s["center"][2]]
    o["center"] = new_center
    _recompute_bbox(o)
    return f"  place_on({obj}, {surface}) → center={[round(c,3) for c in new_center]}"


def _apply_place_between(scene, args, sem_to_mask):
    obj = args.get("obj", "")
    obj1 = args.get("obj1", "")
    obj2 = args.get("obj2", "")
    ok = _resolve(obj, sem_to_mask)
    k1 = _resolve(obj1, sem_to_mask)
    k2 = _resolve(obj2, sem_to_mask)

    if not ok or ok not in scene:
        return f"  place_between({obj}) → NOT FOUND"
    if not k1 or k1 not in scene or not k2 or k2 not in scene:
        return f"  place_between({obj}) → reference objects NOT FOUND"

    c1 = scene[k1]["center"]
    c2 = scene[k2]["center"]
    midpoint = [(c1[i] + c2[i]) / 2 for i in range(3)]
    scene[ok]["center"] = midpoint
    _recompute_bbox(scene[ok])
    return f"  place_between({obj}, {obj1}, {obj2}) → center={[round(c,3) for c in midpoint]}"


def _apply_rotate_towards(scene, args, sem_to_mask):
    obj = args.get("obj", "")
    target = args.get("target", "")
    ok = _resolve(obj, sem_to_mask)
    tk = _resolve(target, sem_to_mask)

    if not ok or ok not in scene:
        return f"  rotate_towards({obj}) → NOT FOUND"
    if not tk or tk not in scene:
        return f"  rotate_towards({obj}, target={target}) → target NOT FOUND"

    oc = scene[ok]["center"]
    tc = scene[tk]["center"]
    angle = math.atan2(tc[2] - oc[2], tc[0] - oc[0])
    scene[ok]["pose"] = angle
    return f"  rotate_towards({obj}, {target}) → pose={round(angle, 4)} rad"


def _apply_rotate_by(scene, args, sem_to_mask):
    obj = args.get("obj", "")
    degrees = float(args.get("degrees", 0))
    ok = _resolve(obj, sem_to_mask)
    if not ok or ok not in scene:
        return f"  rotate_by({obj}) → NOT FOUND"
    scene[ok]["pose"] = scene[ok].get("pose", 0) + math.radians(degrees)
    return f"  rotate_by({obj}, {degrees}°) → pose={round(scene[ok]['pose'], 4)} rad"


def _apply_scale(scene, args, sem_to_mask):
    obj = args.get("obj", "")
    sx = float(args.get("sx", 1))
    sy = float(args.get("sy", 1))
    sz = float(args.get("sz", 1))
    ok = _resolve(obj, sem_to_mask)
    if not ok or ok not in scene:
        return f"  scale({obj}) → NOT FOUND"

    dim = scene[ok].get("dim", [0.1, 0.1, 0.1])
    scene[ok]["dim"] = [dim[0] * sx, dim[1] * sy, dim[2] * sz]
    _recompute_bbox(scene[ok])
    return f"  scale({obj}, [{sx},{sy},{sz}]) → dim={[round(d,4) for d in scene[ok]['dim']]}"


def _apply_align_with(scene, args, sem_to_mask):
    obj = args.get("obj", "")
    target = args.get("target", "")
    axis = args.get("axis", "x")
    ok = _resolve(obj, sem_to_mask)
    tk = _resolve(target, sem_to_mask)

    if not ok or ok not in scene or not tk or tk not in scene:
        return f"  align_with({obj}, {target}) → NOT FOUND"

    axis_map = {"x": 0, "y": 1, "z": 2}
    ai = axis_map.get(axis.lower(), 0)
    scene[ok]["center"][ai] = scene[tk]["center"][ai]
    _recompute_bbox(scene[ok])
    return f"  align_with({obj}, {target}, axis={axis})"


def _apply_stylize(scene, args, sem_to_mask):
    obj = args.get("obj", "")
    desc = args.get("desc", "")
    ok = _resolve(obj, sem_to_mask)
    if not ok or ok not in scene:
        return f"  stylize({obj}) → NOT FOUND"

    scene[ok]["_style"] = desc
    return f"  stylize({obj}, '{desc}')"


def _recompute_bbox(obj_data: Dict):
    """Recompute bbox from center and dim."""
    c = obj_data["center"]
    d = obj_data["dim"]
    obj_data["bbox"] = [
        c[0] - d[0]/2, c[0] + d[0]/2,
        c[1] - d[1]/2, c[1] + d[1]/2,
        c[2] - d[2]/2, c[2] + d[2]/2,
    ]


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Apply EditLang plans to scene layout")
    parser.add_argument("--scene", required=True, help="Path to scene_layout_edited.json")
    parser.add_argument("--plans", required=True, help="Path to benchmark_bedroom_results.json")
    parser.add_argument("--outdir", required=True, help="Output directory for per-instruction scenes")
    args = parser.parse_args()

    # Load originals
    with open(args.scene) as f:
        original_scene = json.load(f)
    with open(args.plans) as f:
        results = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"╔═══════════════════════════════════════════╗")
    print(f"║  Apply Plans to Scene Layout              ║")
    print(f"╚═══════════════════════════════════════════╝")
    print(f"  Scene: {args.scene} ({len(original_scene)} keys)")
    print(f"  Plans: {args.plans} ({len(results)} instructions)")
    print(f"  Output: {args.outdir}")

    for result in results:
        idx = result["index"]
        cmd = result["command"]
        instr = result["instruction"]
        plan = result.get("plan", [])
        success = result.get("success", False)

        print(f"\n{'='*60}")
        print(f"  [{idx}] {cmd}: {instr[:70]}...")
        print(f"  Plan: {len(plan)} actions, success={success}")

        if not plan:
            print(f"  → SKIPPED (empty plan)")
            continue

        # Deep copy scene
        scene = copy.deepcopy(original_scene)
        mask_to_sem, sem_to_mask = build_id_maps(scene)

        # Apply each action
        for step_i, action in enumerate(plan):
            log = apply_action(scene, action, sem_to_mask, mask_to_sem)
            print(f"  Step {step_i+1}: {log}")

        # Save
        outfile = os.path.join(args.outdir, f"scene_layout_instruction_{idx}_{cmd.lower()}.json")
        with open(outfile, "w") as f:
            json.dump(scene, f, indent=2, ensure_ascii=False)
        print(f"  → Saved: {outfile}")

    print(f"\n{'='*60}")
    print(f"  Done. Results in: {args.outdir}")


if __name__ == "__main__":
    main()
