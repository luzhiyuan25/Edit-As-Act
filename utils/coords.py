"""Coordinate transformation utilities for Blender integration.

Handles transformations between:
- Source coordinate system (Y-up)
- Blender coordinate system (Z-up)
"""

from math import atan2
from typing import Tuple, Optional

# Use basic tuples instead of mathutils.Vector for standalone operation
Vector = tuple  # We'll use tuples as vectors


def src_to_blender_point(p: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert source point (Y-up) to Blender point (Z-up).
    
    Source: (x, y, z) where y is up
    Blender: (x, z, -y) where z is up
    """
    x, y, z = p
    return (x, z, -y)


def src_to_blender_dir(d: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert source direction vector (Y-up) to Blender direction (Z-up).
    
    Source: (x, y, z) where y is up
    Blender: (x, z, -y) where z is up
    """
    x, y, z = d
    return (x, z, -y)


def src_yaw_to_blender_yaw(yaw_src: float) -> float:
    """Convert source yaw angle to Blender yaw angle.
    
    Y-up (source) yaw -> Z-up (blender) yaw
    """
    return -yaw_src


def front_face_to_src_dir(tag: str) -> Tuple[float, float, float]:
    """Convert axis-aligned face tag to forward direction in source coords (Y-up).
    
    Args:
        tag: Face tag like "MIN_X", "MAX_X", etc.
        
    Returns:
        Forward direction vector in source coordinates
    """
    lookup = {
        "MIN_X": (-1, 0, 0),
        "MAX_X": (1, 0, 0),
        "MIN_Y": (0, -1, 0),
        "MAX_Y": (0, 1, 0),
        "MIN_Z": (0, 0, -1),
        "MAX_Z": (0, 0, 1),
    }
    return lookup.get(tag, (0, -1, 0))  # Default to -Y


def yaw_from_forward_dir_bl(forward_bl: Tuple[float, float, float]) -> float:
    """Calculate yaw angle from forward direction in Blender coords (Z-up).
    
    Args:
        forward_bl: Forward direction in Blender coordinates
        
    Returns:
        Yaw angle in radians (rotation around Z axis)
    """
    fx, fy, fz = forward_bl
    # Project to XY plane (Z is up in Blender)
    length = (fx * fx + fy * fy) ** 0.5
    
    if length < 1e-8:
        return 0.0
    
    # Normalize the projection
    fx_norm = fx / length
    fy_norm = fy / length
    
    # Assuming +Y is the reference forward direction in Blender
    # yaw = atan2(x, y) gives rotation from +Y axis
    return atan2(fx_norm, fy_norm)


def blender_to_src_point(p_bl: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert Blender point (Z-up) back to source point (Y-up).
    
    Inverse of src_to_blender_point.
    """
    x, y, z = p_bl
    return (x, -z, y)


def vector_length(v: Tuple[float, float, float]) -> float:
    """Calculate length of a 3D vector."""
    x, y, z = v
    return (x * x + y * y + z * z) ** 0.5


def normalize_vector(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector to unit length."""
    length = vector_length(v)
    if length < 1e-8:
        return (0, 0, 0)
    x, y, z = v
    return (x / length, y / length, z / length)


def apply_yaw_to_forward(forward: Tuple[float, float, float], yaw: float, up: Tuple[float, float, float] = (0, 0, 1)) -> Tuple[float, float, float]:
    """Apply yaw rotation to a forward vector.
    
    Args:
        forward: Initial forward direction
        yaw: Rotation angle in radians
        up: Up axis (default is Z-up for Blender)
        
    Returns:
        Rotated forward direction
    """
    from math import cos, sin
    
    # For Z-up rotation, it's a 2D rotation in XY plane
    if up == (0, 0, 1):  # Z-up (Blender)
        fx, fy, fz = forward
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        new_fx = fx * cos_yaw - fy * sin_yaw
        new_fy = fx * sin_yaw + fy * cos_yaw
        return (new_fx, new_fy, fz)
    elif up == (0, 1, 0):  # Y-up (Source)
        fx, fy, fz = forward
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        new_fx = fx * cos_yaw - fz * sin_yaw
        new_fz = fx * sin_yaw + fz * cos_yaw
        return (new_fx, fy, new_fz)
    else:
        raise NotImplementedError(f"Rotation around arbitrary axis {up} not implemented")
