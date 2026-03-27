# ============================================================
# BLENDER SCENE IMPORT SCRIPT
# ============================================================
# Preconditions:
#   - {RODIN_DIR}/{object_name}.glb files must exist.
#   - If RODIN_DIR is empty, the path recorded inside the JSON will be used.
#
# Usage:
#   1. Set JSON_PATH : path to scene_layout.json
#   2. Set RODIN_DIR : path to GLB folder (leave empty to use JSON path)
#   3. Run Script in an empty Blender scene
#
# Requirements:
#   - glTF 2.0 addon must be enabled
#     (Edit > Preferences > Add-ons > search "glTF 2.0" and enable)
#
# Round-trip guarantee:
#   - Transforms are restored only from matrix_world.
#   - EMPTY object's dim / bbox fields are metadata used only by the scene planner
#     (apply_plan_to_scene_v2.py) and are NOT used for transform restoration.
#     Therefore, export → JSON → import round-trip is perfectly guaranteed.
# ============================================================

import bpy
import json
import os
from mathutils import Matrix

# ============================================================
# ★ Settings
# ============================================================
JSON_PATH   = "C:/Users/YourName/Desktop/scene_export/scene_layout.json"
RODIN_DIR   = ""      # If empty, use rodin_dir from JSON
CLEAR_SCENE = True    # True: clear existing scene / False: append to current scene
# ============================================================


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for col in [bpy.data.meshes, bpy.data.cameras,
                bpy.data.lights, bpy.data.materials, bpy.data.images]:
        for block in list(col):
            if block.users == 0:
                col.remove(block)


def get_or_create_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def topological_sort(objects_data):
    """Sort so that parents always come before children"""
    by_name = {e["name"]: e for e in objects_data}
    visited, result = set(), []

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        parent = by_name.get(name, {}).get("parent")
        if parent:
            visit(parent)
        result.append(name)

    for e in objects_data:
        visit(e["name"])
    return [by_name[n] for n in result if n in by_name]


def import_glb_as(glb_path, target_name):
    """
    Import GLB and rename mesh object to target_name.
    Automatically removes unnecessary Empty nodes (e.g., GLB root).
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    new_objs = list(set(bpy.data.objects) - before)

    mesh_objs  = [o for o in new_objs if o.type == 'MESH']
    extra_objs = [o for o in new_objs if o.type != 'MESH']

    # Remove non-mesh artifacts (GLB root Empty, etc.)
    for extra in extra_objs:
        bpy.data.objects.remove(extra, do_unlink=True)

    if not mesh_objs:
        return None

    # If multiple meshes, merge into one
    if len(mesh_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for mo in mesh_objs:
            mo.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()

    result = bpy.context.view_layer.objects.active
    result.name = target_name
    return result


def run_import(json_path):
    print(f"\n{'='*55}")
    print(f"Scene import started")
    print(f"JSON: {json_path}")
    print(f"{'='*55}")

    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    # Determine RODIN_DIR: script setting > JSON path priority
    rodin_dir = RODIN_DIR.strip() or scene_data.get("rodin_dir", "")
    if not rodin_dir:
        print("❌ RODIN_DIR must be set or defined in JSON.")
        return
    if not os.path.isdir(rodin_dir):
        print(f"❌ RODIN_DIR does not exist: {rodin_dir}")
        return

    print(f"RODIN_DIR: {rodin_dir}")

    if CLEAR_SCENE:
        clear_scene()
        print("  Existing scene cleared")

    objects_data   = scene_data.get("objects", [])
    sorted_entries = topological_sort(objects_data)
    created = {}   # name → bpy.types.Object

    # ── Step 1: Create objects ────────────────────────────────
    for entry in sorted_entries:
        name     = entry["name"]
        obj_type = entry["type"]
        obj      = None

        # ── MESH ──────────────────────────────────────────
        if obj_type == 'MESH':
            glb_path = os.path.join(rodin_dir, f"{name}.glb")
            if not os.path.exists(glb_path):
                print(f"  ✘ GLB 없음, 건너뜀: {glb_path}")
                continue

            obj = import_glb_as(glb_path, name)
            if obj is None:
                print(f"  ✘ GLB 임포트 실패 (메시 없음): {name}.glb")
                continue
            print(f"  ✔ 메시 임포트: {name}.glb")

        # ── LIGHT ─────────────────────────────────────────
        elif obj_type == 'LIGHT':
            ld    = entry.get("light_data", {})
            ltype = ld.get("type", "POINT")
            light = bpy.data.lights.new(name=name, type=ltype)
            light.color  = ld.get("color",  [1.0, 1.0, 1.0])
            light.energy = ld.get("energy", 1000.0)
            if ltype in ('POINT', 'SPOT') and "shadow_soft_size" in ld:
                light.shadow_soft_size = ld["shadow_soft_size"]
            if ltype == 'SPOT':
                if "spot_size"  in ld: light.spot_size  = ld["spot_size"]
                if "spot_blend" in ld: light.spot_blend = ld["spot_blend"]
            if ltype == 'SUN'  and "angle"  in ld: light.angle  = ld["angle"]
            if ltype == 'AREA':
                if "shape"  in ld: light.shape  = ld["shape"]
                if "size"   in ld: light.size   = ld["size"]
                if "size_y" in ld: light.size_y = ld["size_y"]
            obj = bpy.data.objects.new(name, light)
            bpy.context.scene.collection.objects.link(obj)
            print(f"  ✔ 조명 생성: {name}  ({ltype})")

        # ── CAMERA ────────────────────────────────────────
        elif obj_type == 'CAMERA':
            cd  = entry.get("camera_data", {})
            cam = bpy.data.cameras.new(name=name)
            cam.type       = cd.get("type",      "PERSP")
            cam.lens       = cd.get("lens",       50.0)
            cam.clip_start = cd.get("clip_start", 0.1)
            cam.clip_end   = cd.get("clip_end",   1000.0)
            if "lens_unit"     in cd: cam.lens_unit     = cd["lens_unit"]
            if "sensor_width"  in cd: cam.sensor_width  = cd["sensor_width"]
            if "sensor_height" in cd: cam.sensor_height = cd["sensor_height"]
            if "sensor_fit"    in cd: cam.sensor_fit    = cd["sensor_fit"]
            obj = bpy.data.objects.new(name, cam)
            bpy.context.scene.collection.objects.link(obj)
            print(f"  ✔ 카메라 생성: {name}")

        elif obj_type == 'EMPTY':
            obj = bpy.data.objects.new(name, None)
            bpy.context.scene.collection.objects.link(obj)
            print(f"  ✔ Empty 생성: {name}")

        else:
            print(f"  — 건너뜀 (미지원 타입): {name}  ({obj_type})")
            continue

        if obj:
            created[name] = obj

    # ── Step 2: Set up parent-child relationships
    # Since the entries are already topologically sorted, parents are guaranteed
    # to have been created before their children.
    for entry in sorted_entries:
        name        = entry["name"]
        parent_name = entry.get("parent")
        if parent_name and name in created and parent_name in created:
            child  = created[name]
            parent = created[parent_name]
            child.parent      = parent
            child.parent_type = entry.get("parent_type", "OBJECT")

    # ── Step 3: Restore transforms (parent → child order) ────────────────
    bpy.context.view_layer.update()

    for entry in sorted_entries:
        name = entry["name"]
        if name not in created:
            continue
        obj = created[name]

        mw = entry.get("matrix_world")
        if mw:
            obj.matrix_world = Matrix(mw)
        else:
            obj.location       = entry.get("location",         [0, 0, 0])
            obj.rotation_mode  = entry.get("rotation_mode",    "XYZ")
            obj.rotation_euler = entry.get("rotation_euler",   [0, 0, 0])
            obj.scale          = entry.get("scale",            [1, 1, 1])

        obj.hide_viewport = entry.get("hide_viewport", False)
        obj.hide_render   = entry.get("hide_render",   False)

    bpy.context.view_layer.update()

    # ── Step 4: Assign collections ──────────────────────────────────
    for entry in sorted_entries:
        name = entry["name"]
        if name not in created:
            continue
        obj = created[name]

        col_names = [c for c in entry.get("collections", [])
                     if c != "Scene Collection"]
        if not col_names:
            continue

        for existing in list(obj.users_collection):
            existing.objects.unlink(obj)
        for col_name in col_names:
            get_or_create_collection(col_name).objects.link(obj)

    # ── Step 5: Restore scene settings ─────────────────────────────────
    settings   = scene_data.get("scene_settings", {})
    active_cam = settings.get("active_camera")
    if active_cam and active_cam in created:
        bpy.context.scene.camera = created[active_cam]
        print(f"  Active camera: {active_cam}")
    if settings.get("render_resolution_x"):
        bpy.context.scene.render.resolution_x = settings["render_resolution_x"]
        bpy.context.scene.render.resolution_y = settings["render_resolution_y"]

    print(f"\n{'='*55}")
    print(f"✅ Import complete  —  {len(created)} objects created")
    print(f"{'='*55}\n")


run_import(JSON_PATH)
