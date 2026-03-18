"""Geometric validation for scene editing actions.

Provides collision detection, boundary checking, and spatial constraint validation.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import json
import math
from dataclasses import dataclass
from pathlib import Path

from editors.editlang import Action, Predicate


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for collision detection."""
    min_point: Tuple[float, float, float]
    max_point: Tuple[float, float, float]
    
    @property
    def center(self) -> Tuple[float, float, float]:
        """Get center point of bounding box."""
        return tuple(
            (self.min_point[i] + self.max_point[i]) / 2
            for i in range(3)
        )
    
    @property
    def dims(self) -> Tuple[float, float, float]:
        """Get dimensions of bounding box."""
        return tuple(
            self.max_point[i] - self.min_point[i]
            for i in range(3)
        )
    
    def intersects(self, other: "BoundingBox", tolerance: float = 0.01) -> bool:
        """Check if this box intersects with another.
        
        Args:
            other: Other bounding box
            tolerance: Small overlap tolerance
            
        Returns:
            True if boxes intersect
        """
        for i in range(3):
            if (self.max_point[i] + tolerance < other.min_point[i] or
                self.min_point[i] - tolerance > other.max_point[i]):
                return False
        return True
    
    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """Check if point is inside bounding box.
        
        Args:
            point: 3D point to check
            
        Returns:
            True if point is inside box
        """
        return all(
            self.min_point[i] <= point[i] <= self.max_point[i]
            for i in range(3)
        )
    
    def distance_to(self, other: "BoundingBox") -> float:
        """Calculate minimum distance between two boxes.
        
        Args:
            other: Other bounding box
            
        Returns:
            Minimum distance (0 if intersecting)
        """
        if self.intersects(other):
            return 0.0
        
        dist_sq = 0.0
        for i in range(3):
            if other.max_point[i] < self.min_point[i]:
                dist_sq += (self.min_point[i] - other.max_point[i]) ** 2
            elif other.min_point[i] > self.max_point[i]:
                dist_sq += (other.min_point[i] - self.max_point[i]) ** 2
        
        return math.sqrt(dist_sq)


@dataclass
class SceneObject:
    """Represents an object in the scene for geometric validation."""
    id: str
    bbox: BoundingBox
    movable: bool = True
    supporting: Set[str] = None  # Objects this one supports
    
    def __post_init__(self):
        if self.supporting is None:
            self.supporting = set()


class GeomChecker:
    """Validates geometric constraints for actions.
    
    Checks collision-free paths, support relationships, clearances, etc.
    """
    
    def __init__(self, scene_data: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize geometry checker.
        
        Args:
            scene_data: Scene description with object positions
            config: Configuration parameters
        """
        self.config = config or {}
        self.objects: Dict[str, SceneObject] = {}
        self.room_bbox: Optional[BoundingBox] = None
        
        # Configuration parameters
        self.collision_tolerance = self.config.get("collision_tolerance", 0.05)
        self.min_clearance = self.config.get("min_clearance", 0.1)
        self.support_tolerance = self.config.get("support_tolerance", 0.01)
        self.facing_angle_threshold = math.radians(self.config.get("facing_angle_deg", 45))
        
        if scene_data:
            self.load_scene(scene_data)
    
    def load_scene(self, scene_data: Dict[str, Any]) -> None:
        """Load scene from JSON data.
        
        Supports two formats:
        1. Standard: {"objects": [{"id": ..., "center": ...}], "room": {...}}
        2. Flat dict: {"obj_id": {"center": [...], "dim": [...]}, ...}
        """
        # Load room boundaries
        if "room" in scene_data:
            room = scene_data["room"]
            if "bbox" in room:
                bbox = room["bbox"]
                self.room_bbox = BoundingBox(
                    tuple(bbox["min"]),
                    tuple(bbox["max"])
                )
            elif "dims" in room and "center" in room:
                center = room["center"]
                dims = room["dims"]
                half_dims = [d/2 for d in dims]
                self.room_bbox = BoundingBox(
                    (center[0] - half_dims[0], center[1] - half_dims[1], center[2] - half_dims[2]),
                    (center[0] + half_dims[0], center[1] + half_dims[1], center[2] + half_dims[2])
                )
        
        # Load objects — standard format
        if "objects" in scene_data:
            for obj_data in scene_data["objects"]:
                obj_id = obj_data.get("id", obj_data.get("name"))
                if not obj_id:
                    continue
                self._load_single_object(obj_id, obj_data)
        else:
            # Flat dict format: {"obj_id": {"center": [...], "dim": [...]}}
            for obj_id, obj_data in scene_data.items():
                if obj_id == "room":
                    continue  # Skip room key
                if not isinstance(obj_data, dict):
                    continue
                if "center" not in obj_data and "bbox" not in obj_data:
                    continue
                self._load_single_object(obj_id, obj_data)
    
    def _load_single_object(self, obj_id: str, obj_data: Dict[str, Any]) -> None:
        """Load a single object into the scene."""
        # Create bounding box
        if "bbox" in obj_data:
            bbox = obj_data["bbox"]
            if isinstance(bbox, dict):
                obj_bbox = BoundingBox(tuple(bbox["min"]), tuple(bbox["max"]))
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 6:
                obj_bbox = BoundingBox(
                    (bbox[0], bbox[1], bbox[2]),
                    (bbox[3], bbox[4], bbox[5])
                )
            else:
                return
        elif "center" in obj_data and ("dims" in obj_data or "dim" in obj_data):
            center = obj_data["center"]
            dims = obj_data.get("dims", obj_data.get("dim"))
            half_dims = [d/2 for d in dims]
            obj_bbox = BoundingBox(
                (center[0] - half_dims[0], center[1] - half_dims[1], center[2] - half_dims[2]),
                (center[0] + half_dims[0], center[1] + half_dims[1], center[2] + half_dims[2])
            )
        else:
            return  # Skip objects without position data
        
        scene_obj = SceneObject(
            id=obj_id,
            bbox=obj_bbox,
            movable=obj_data.get("movable", not obj_data.get("on_wall", False))
        )
        self.objects[obj_id] = scene_obj
    
    def feasible(self, a: Action, simulated_scene: Optional[Dict[str, Any]] = None) -> bool:
        """Check if an action is geometrically feasible.
        
        Validates spatial constraints implied by the action.
        
        Args:
            a: Action to validate
            simulated_scene: Optional simulated scene to use for validation
            
        Returns:
            True if action is geometrically feasible
        """
        # If simulated scene provided, temporarily use it for validation
        if simulated_scene:
            old_objects = self.objects
            old_room_bbox = self.room_bbox
            
            # Load simulated scene data temporarily
            self.objects = {}
            for obj in simulated_scene.get('objects', []):
                obj_id = obj.get('id')
                if obj_id and 'center' in obj and 'dims' in obj:
                    center = obj['center']
                    dims = obj['dims']
                    half_dims = [d/2 for d in dims]
                    obj_bbox = BoundingBox(
                        (center[0] - half_dims[0], center[1] - half_dims[1], center[2] - half_dims[2]),
                        (center[0] + half_dims[0], center[1] + half_dims[1], center[2] + half_dims[2])
                    )
                    self.objects[obj_id] = SceneObject(
                        id=obj_id,
                        bbox=obj_bbox,
                        movable=obj.get('movable', True)
                    )
            
            if 'room' in simulated_scene:
                room = simulated_scene['room']
                if 'bbox' in room:
                    bbox = room['bbox']
                    self.room_bbox = BoundingBox(
                        tuple(bbox['min']),
                        tuple(bbox['max'])
                    )
            
            # Perform feasibility check
            result = self._feasible_check(a)
            
            # Restore original objects
            self.objects = old_objects
            self.room_bbox = old_room_bbox
            
            return result
        else:
            return self._feasible_check(a)
    
    def _feasible_check(self, a: Action) -> bool:
        """Internal feasibility check using current scene state.
        
        Args:
            a: Action to validate
            
        Returns:
            True if action is geometrically feasible
        """
        # Dispatch based on action type
        if a.name == "rotate_towards":
            return self._feasible_rotate(a)
        elif a.name == "move_near":
            return self._feasible_move_near(a)
        elif a.name == "place_on":
            return self._feasible_place_on(a)
        elif a.name == "move_to":
            return self._feasible_move_to(a)
        elif a.name == "align_with":
            return self._feasible_align(a)
        elif a.name == "place_between":
            return self._feasible_place_between(a)
        elif a.name in ("add_object", "remove_object", "stylize", "scale",
                        "rotate_by", "place_relative", "move_group"):
            # These actions are always geometrically feasible
            # (or do not have meaningful geometric constraints)
            return True
        else:
            # Unknown action type, assume feasible
            return True
    
    def _feasible_rotate(self, a: Action) -> bool:
        """Check if rotation is feasible."""
        obj_id = a.args.get("obj")
        # Support both 'target' (YAML schema) and 'anchor' (legacy)
        anchor_id = a.args.get("target") or a.args.get("anchor")
        
        # Check objects exist
        if obj_id not in self.objects:
            return True  # New/unknown objects — accept
        if anchor_id and anchor_id not in self.objects:
            return True  # Target may be new — accept
        
        return True
    
    def _feasible_move_near(self, a: Action) -> bool:
        """Check if move near is feasible.
        
        Args:
            a: Move near action
            
        Returns:
            True if move is feasible
        """
        obj_id = a.args.get("obj")
        target_id = a.args.get("target")
        tau = a.args.get("tau", self.min_clearance)
        
        # Check objects exist
        if obj_id not in self.objects or target_id not in self.objects:
            return False
        
        obj = self.objects[obj_id]
        target = self.objects[target_id]
        
        # Check if object is movable
        if not obj.movable:
            return False
        
        # Check if there's space near target
        # Simplified: just check if we can place object without collision
        return self._has_space_near(obj, target, tau)
    
    def _feasible_place_on(self, a: Action) -> bool:
        """Check if place on is feasible."""
        obj_id = a.args.get("obj")
        # Support both 'surface' (YAML schema) and 'support' (legacy)
        support_id = a.args.get("surface") or a.args.get("support")
        
        # Accept if objects are unknown (may be new)
        if not obj_id or not support_id:
            return True
        if obj_id not in self.objects or support_id not in self.objects:
            return True  # New/unknown objects — accept
        
        obj = self.objects[obj_id]
        support = self.objects[support_id]
        
        if not obj.movable:
            return False
        
        # Relaxed size check: allow if footprint reasonably fits
        obj_dims = obj.bbox.dims
        support_dims = support.bbox.dims
        if obj_dims[0] > support_dims[0] * 1.5 or obj_dims[2] > support_dims[2] * 1.5:
            return False
        
        return True
    
    def _feasible_move_to(self, a: Action) -> bool:
        """Check if move to position is feasible.
        
        Handles both:
        - Coordinate-based pos: x, y, z as separate args or "x,y,z" string
        - Symbolic pos: "near_sofa", "behind_table" etc. → accept (no geometry check possible)
        """
        obj_id = a.args.get("obj")
        
        # Try to extract coordinates from args
        x = a.args.get("x")
        y = a.args.get("y")
        z = a.args.get("z")
        
        # If x/y/z not separate, try parsing 'pos' arg
        if x is None or y is None or z is None:
            pos = a.args.get("pos")
            if pos is None:
                return True  # No position info → accept
            
            # Try to parse pos as "x,y,z" or [x, y, z]
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                try:
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                except (ValueError, TypeError):
                    return True  # Non-numeric → symbolic position, accept
            elif isinstance(pos, str):
                try:
                    parts = pos.replace(",", " ").split()
                    if len(parts) == 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    else:
                        return True  # Symbolic position like "near_sofa" → accept
                except ValueError:
                    return True  # Non-numeric → accept
            else:
                return True  # Unknown pos format → accept
        
        # Convert to float if needed
        try:
            x, y, z = float(x), float(y), float(z)
        except (ValueError, TypeError):
            return True  # Non-numeric → accept
        
        if obj_id not in self.objects:
            return True  # Unknown object → accept
        
        obj = self.objects[obj_id]
        
        if not obj.movable:
            return False
        
        # Create new bounding box at target position
        dims = obj.bbox.dims
        half_dims = [d/2 for d in dims]
        new_bbox = BoundingBox(
            (x - half_dims[0], y - half_dims[1], z - half_dims[2]),
            (x + half_dims[0], y + half_dims[1], z + half_dims[2])
        )
        
        # Check room boundaries
        if self.room_bbox and not self._bbox_in_room(new_bbox):
            return False
        
        # Check collisions with other objects
        for other_id, other in self.objects.items():
            if other_id != obj_id:
                if new_bbox.intersects(other.bbox, self.collision_tolerance):
                    return False
        
        return True
    
    def _feasible_align(self, a: Action) -> bool:
        """Check if alignment is feasible."""
        obj_id = a.args.get("obj")
        # Support both 'target' (YAML schema) and 'reference' (legacy)
        ref_id = a.args.get("target") or a.args.get("reference")
        
        if not obj_id:
            return True
        if obj_id not in self.objects:
            return True  # Unknown object — accept
        
        obj = self.objects[obj_id]
        if not obj.movable:
            return False
        
        return True
    
    def _feasible_place_between(self, a: Action) -> bool:
        """Check if placing between two objects is feasible.
        
        Args:
            a: Place between action
            
        Returns:
            True if placement is feasible
        """
        obj_id = a.args.get("obj")
        obj1_id = a.args.get("obj1")
        obj2_id = a.args.get("obj2")
        
        # Check objects exist
        if (obj_id not in self.objects or 
            obj1_id not in self.objects or 
            obj2_id not in self.objects):
            return False
        
        obj = self.objects[obj_id]
        obj1 = self.objects[obj1_id]
        obj2 = self.objects[obj2_id]
        
        # Check if object is movable
        if not obj.movable:
            return False
        
        # Calculate midpoint between obj1 and obj2
        mid_point = tuple(
            (obj1.bbox.center[i] + obj2.bbox.center[i]) / 2
            for i in range(3)
        )
        
        # Check if there's space at midpoint
        dims = obj.bbox.dims
        half_dims = [d/2 for d in dims]
        new_bbox = BoundingBox(
            (mid_point[0] - half_dims[0], mid_point[1] - half_dims[1], mid_point[2] - half_dims[2]),
            (mid_point[0] + half_dims[0], mid_point[1] + half_dims[1], mid_point[2] + half_dims[2])
        )
        
        # Check room boundaries
        if self.room_bbox and not self._bbox_in_room(new_bbox):
            return False
        
        # Check collisions (except with obj1 and obj2)
        for other_id, other in self.objects.items():
            if other_id not in {obj_id, obj1_id, obj2_id}:
                if new_bbox.intersects(other.bbox, self.collision_tolerance):
                    return False
        
        return True
    
    def _has_space_near(self, obj: SceneObject, target: SceneObject, distance: float) -> bool:
        """Check if there's space to place object near target.
        
        Args:
            obj: Object to place
            target: Target object
            distance: Desired distance
            
        Returns:
            True if space is available
        """
        # Try multiple positions around target
        angles = [0, math.pi/2, math.pi, 3*math.pi/2]
        target_center = target.bbox.center
        obj_dims = obj.bbox.dims
        
        for angle in angles:
            # Calculate potential position
            dx = (target.bbox.dims[0]/2 + obj_dims[0]/2 + distance) * math.cos(angle)
            dz = (target.bbox.dims[2]/2 + obj_dims[2]/2 + distance) * math.sin(angle)
            
            new_center = (
                target_center[0] + dx,
                target_center[1],
                target_center[2] + dz
            )
            
            # Create bounding box at new position
            half_dims = [d/2 for d in obj_dims]
            new_bbox = BoundingBox(
                (new_center[0] - half_dims[0], new_center[1] - half_dims[1], new_center[2] - half_dims[2]),
                (new_center[0] + half_dims[0], new_center[1] + half_dims[1], new_center[2] + half_dims[2])
            )
            
            # Check if position is valid
            if self._is_position_valid(new_bbox, obj.id):
                return True
        
        return False
    
    def _is_position_valid(self, bbox: BoundingBox, obj_id: str) -> bool:
        """Check if a bounding box position is valid.
        
        Args:
            bbox: Bounding box to check
            obj_id: ID of object (to exclude from collision check)
            
        Returns:
            True if position is valid
        """
        # Check room boundaries
        if self.room_bbox and not self._bbox_in_room(bbox):
            return False
        
        # Check collisions with other objects
        for other_id, other in self.objects.items():
            if other_id != obj_id:
                if bbox.intersects(other.bbox, self.collision_tolerance):
                    return False
        
        return True
    
    def _bbox_in_room(self, bbox: BoundingBox) -> bool:
        """Check if bounding box is within room boundaries.
        
        Args:
            bbox: Bounding box to check
            
        Returns:
            True if bbox is within room
        """
        if not self.room_bbox:
            return True
        
        return (bbox.min_point[0] >= self.room_bbox.min_point[0] and
                bbox.max_point[0] <= self.room_bbox.max_point[0] and
                bbox.min_point[1] >= self.room_bbox.min_point[1] and
                bbox.max_point[1] <= self.room_bbox.max_point[1] and
                bbox.min_point[2] >= self.room_bbox.min_point[2] and
                bbox.max_point[2] <= self.room_bbox.max_point[2])
    
    def check_collision(self, obj1_id: str, obj2_id: str) -> bool:
        """Check if two objects collide.
        
        Args:
            obj1_id: First object ID
            obj2_id: Second object ID
            
        Returns:
            True if objects collide
        """
        if obj1_id not in self.objects or obj2_id not in self.objects:
            return False
        
        return self.objects[obj1_id].bbox.intersects(
            self.objects[obj2_id].bbox,
            self.collision_tolerance
        )
    
    def get_collisions(self) -> List[Tuple[str, str]]:
        """Get all collision pairs in the scene.
        
        Returns:
            List of colliding object ID pairs
        """
        collisions = []
        obj_ids = list(self.objects.keys())
        
        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                if self.check_collision(obj_ids[i], obj_ids[j]):
                    collisions.append((obj_ids[i], obj_ids[j]))
        
        return collisions
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "GeomChecker":
        """Create geometry checker from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            GeomChecker instance
        """
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                import yaml
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(config=config.get("geometry", {}))
