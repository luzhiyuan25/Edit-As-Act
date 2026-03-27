# ============================================================
# BLENDER SCENE EXPORT SCRIPT
# ============================================================
# Preconditions:
#   - For every mesh object in the scene,
#     {RODIN_DIR}/{object_name}.glb must already exist.
#
# Usage:
#   1. RODIN_DIR  : path to the folder containing GLB files
#   2. EXPORT_DIR : path where scene_layout.json will be saved
#   3. Run Script
#
# dim / bbox fields (EMPTY objects):
#   - dim  : world-space AABB size [dx, dy, dz] (unit: meters)
#   - bbox : world-space AABB range
#            [min_x, max_x, min_y, max_y, min_z, max_z]
#   - Used as metadata for spatial reasoning in the scene planner
#     (apply_plan_to_scene_v2.py)
#   - Ignored during import (not used for transform restoration)
# ============================================================

import bpy
import json
import os
from mathutils import Vector

# ============================================================
# ★ Settings
# ============================================================
RODIN_DIR  = "C:/Users/YourName/Desktop/rodin"
EXPORT_DIR = "C:/Users/YourName/Desktop/scene_export"
JSON_FILENAME = "scene_layout.json"
# ============================================================


def resolve_path(path):
    if path.startswith("//"):
        if not bpy.data.filepath:
            return os.path.join(os.path.expanduser("~"), path[2:])
        return bpy.path.abspath(path)
    return path


def mat4_to_list(m):
    return [list(row) for row in m]


def compute_object_dim(obj):
    """Compute world-space AABB for EMPTY object using child meshes.

    Finds the first MESH in obj.children_recursive,
    projects its 8 bounding box corners into world space using matrix_world,
    then returns AABB dim and bbox.

    Returns:
        {"dim": [dx, dy, dz], "bbox": [...]}
        or None if no MESH child is found.
    """
    # Handle defensive case where EMPTY itself is MESH
    if obj.type == 'MESH':
        target = obj
    else:
        target = None
        for child in obj.children_recursive:
            if child.type == 'MESH':
                target = child
                break

    if target is None:
        return None

    # Transform local bounding box corners → world space
    corners_world = [target.matrix_world @ Vector(c) for c in target.bound_box]

    xs = [v.x for v in corners_world]
    ys = [v.y for v in corners_world]
    zs = [v.z for v in corners_world]

    dim  = [round(max(xs) - min(xs), 6),
            round(max(ys) - min(ys), 6),
            round(max(zs) - min(zs), 6)]
    bbox = [round(min(xs), 6), round(max(xs), 6),
            round(min(ys), 6), round(max(ys), 6),
            round(min(zs), 6), round(max(zs), 6)]

    return {"dim": dim, "bbox": bbox}


def run_export():
    rodin_dir  = resolve_path(RODIN_DIR)
    export_dir = resolve_path(EXPORT_DIR)
    os.makedirs(export_dir, exist_ok=True)

    scene = bpy.context.scene
    print(f"\n{'='*55}")
    print(f"Exporting scene: {scene.name}")
    print(f"RODIN_DIR : {rodin_dir}")
    print(f"EXPORT_DIR: {export_dir}")
    print(f"{'='*55}")

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    objects_data = []
    missing_glbs = []

    for obj in scene.objects:
        entry = {
            "name"               : obj.name,
            "type"               : obj.type,
            "location"           : [round(v, 6) for v in obj.location],
            "rotation_mode"      : obj.rotation_mode,
            "rotation_euler"     : [round(v, 6) for v in obj.rotation_euler],
            "rotation_quaternion": [round(v, 6) for v in obj.rotation_quaternion],
            "scale"              : [round(v, 6) for v in obj.scale],
            "matrix_world"       : mat4_to_list(obj.matrix_world),
            "parent"             : obj.parent.name if obj.parent else None,
            "parent_type"        : obj.parent_type,
            "collections"        : [c.name for c in obj.users_collection],
            "hide_viewport"      : obj.hide_viewport,
            "hide_render"        : obj.hide_render,
        }

        if obj.type == 'MESH':
            glb_filename = f"{obj.name}.glb"
            glb_path     = os.path.join(rodin_dir, glb_filename)

            if os.path.exists(glb_path):
                entry["glb_file"] = glb_filename
                print(f"  ✔ {obj.name}  →  {glb_filename}")
            else:
                entry["glb_file"] = glb_filename
                missing_glbs.append(glb_filename)
                print(f"  ✘ {obj.name}  →  GLB 없음: {glb_path}")

        elif obj.type == 'EMPTY':
            # ── World-space bounding-box metadata (for planner) ──────────────
            # Only added for scene objects that have MESH children
            # Ignored during import (not used for transform restoration)
            bbox_info = compute_object_dim(obj)
            if bbox_info:
                entry["dim"]  = bbox_info["dim"]
                entry["bbox"] = bbox_info["bbox"]
            print(f"  ✔ {obj.name}  (EMPTY{', with dim' if bbox_info else ''})")

        elif obj.type == 'LIGHT' and obj.data:
            l  = obj.data
            ld = {
                "type"  : l.type,
                "color" : [round(v, 6) for v in l.color],
                "energy": l.energy,
            }
            if l.type in ('POINT', 'SPOT'):
                ld["shadow_soft_size"] = l.shadow_soft_size
            if l.type == 'SPOT':
                ld["spot_size"]  = l.spot_size
                ld["spot_blend"] = l.spot_blend
            if l.type == 'SUN':
                ld["angle"] = l.angle
            if l.type == 'AREA':
                ld["shape"]  = l.shape
                ld["size"]   = l.size
                ld["size_y"] = l.size_y
            entry["light_data"] = ld

        elif obj.type == 'CAMERA' and obj.data:
            c = obj.data
            entry["camera_data"] = {
                "type"         : c.type,
                "lens"         : c.lens,
                "lens_unit"    : c.lens_unit,
                "clip_start"   : c.clip_start,
                "clip_end"     : c.clip_end,
                "sensor_width" : c.sensor_width,
                "sensor_height": c.sensor_height,
                "sensor_fit"   : c.sensor_fit,
            }

        objects_data.append(entry)

    out = {
        "version"        : "1.0",
        "blender_version": ".".join(str(v) for v in bpy.app.version),
        "scene_name"     : scene.name,
        "rodin_dir"      : rodin_dir,       
        "scene_settings" : {
            "active_camera"      : scene.camera.name if scene.camera else None,
            "render_resolution_x": scene.render.resolution_x,
            "render_resolution_y": scene.render.resolution_y,
            "render_fps"         : scene.render.fps,
        },
        "objects": objects_data,
    }

    json_path = os.path.join(export_dir, JSON_FILENAME)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    if missing_glbs:
        print(f"⚠ Missing GLBs ({len(missing_glbs)}): {', '.join(missing_glbs)}")
    print(f"✅ Export complete  —  {len(objects_data)} objects")
    print(f"   JSON: {json_path}")
    print(f"{'='*55}\n")


run_export()
