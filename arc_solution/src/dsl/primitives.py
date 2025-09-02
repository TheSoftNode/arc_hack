"""
ARC Prize 2025 - Domain Specific Language Primitives

This module defines the primitive operations that can be composed
to create transformation programs for ARC tasks.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
from ..core.types import Grid, GridArray, Object, PrimitiveType


class DSLPrimitives:
    """
    Collection of primitive operations for ARC task transformations.
    
    Operations are organized by type:
    - Geometric: rotate, translate, scale, reflect
    - Color: recolor, fill, paint
    - Logical: AND, OR, XOR operations  
    - Morphological: dilation, erosion, opening, closing
    - Pattern: detect and complete patterns
    - Object: manipulate detected objects
    - Spatial: spatial transformations and relationships
    """
    
    def __init__(self):
        self.primitives = self._build_primitive_registry()
    
    def _build_primitive_registry(self) -> Dict[str, Callable]:
        """Build registry of all available primitive operations"""
        return {
            # Geometric operations
            'rotate': self.rotate,
            'reflect': self.reflect,
            'translate': self.translate,
            'scale': self.scale,
            
            # Color operations
            'recolor': self.recolor,
            'fill_region': self.fill_region,
            'paint_pattern': self.paint_pattern,
            
            # Logical operations
            'grid_and': self.grid_and,
            'grid_or': self.grid_or,
            'grid_xor': self.grid_xor,
            
            # Morphological operations
            'dilate': self.dilate,
            'erode': self.erode,
            'opening': self.opening,
            'closing': self.closing,
            
            # Object operations
            'move_object': self.move_object,
            'duplicate_object': self.duplicate_object,
            'remove_object': self.remove_object,
            
            # Pattern operations
            'complete_pattern': self.complete_pattern,
            'extend_pattern': self.extend_pattern,
            
            # Utility operations
            'identity': self.identity,
            'resize': self.resize,
            'crop': self.crop,
            'pad': self.pad,
        }
    
    def get_primitive(self, name: str) -> Optional[Callable]:
        """Get a primitive operation by name"""
        return self.primitives.get(name)
    
    def list_primitives(self) -> List[str]:
        """List all available primitive operation names"""
        return list(self.primitives.keys())
    
    # Geometric Operations
    def rotate(self, grid: Grid, angle: int = 90) -> Grid:
        """Rotate grid by specified angle (90, 180, 270 degrees)"""
        grid_array = np.array(grid)
        k = angle // 90
        rotated = np.rot90(grid_array, k=k)
        return rotated.tolist()
    
    def reflect(self, grid: Grid, axis: str = 'vertical') -> Grid:
        """Reflect grid along specified axis"""
        grid_array = np.array(grid)
        if axis == 'vertical':
            reflected = np.fliplr(grid_array)
        elif axis == 'horizontal':
            reflected = np.flipud(grid_array)
        else:
            raise ValueError(f"Unknown reflection axis: {axis}")
        return reflected.tolist()
    
    def translate(self, grid: Grid, dx: int = 0, dy: int = 0) -> Grid:
        """Translate (shift) grid by dx, dy"""
        grid_array = np.array(grid)
        h, w = grid_array.shape
        
        # Create new grid with translation
        translated = np.zeros_like(grid_array)
        
        # Compute valid regions
        src_y_start = max(0, -dy)
        src_y_end = min(h, h - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(w, w - dx)
        
        dst_y_start = max(0, dy)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, dx)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            grid_array[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return translated.tolist()
    
    def scale(self, grid: Grid, factor: float = 2.0) -> Grid:
        """Scale grid by specified factor"""
        grid_array = np.array(grid)
        h, w = grid_array.shape
        
        new_h, new_w = int(h * factor), int(w * factor)
        scaled = np.zeros((new_h, new_w), dtype=grid_array.dtype)
        
        # Simple nearest neighbor scaling
        for i in range(new_h):
            for j in range(new_w):
                orig_i = int(i / factor)
                orig_j = int(j / factor)
                if orig_i < h and orig_j < w:
                    scaled[i, j] = grid_array[orig_i, orig_j]
        
        return scaled.tolist()
    
    # Color Operations
    def recolor(self, grid: Grid, color_map: Dict[int, int]) -> Grid:
        """Apply color mapping to grid"""
        grid_array = np.array(grid)
        recolored = grid_array.copy()
        
        for old_color, new_color in color_map.items():
            recolored[grid_array == old_color] = new_color
        
        return recolored.tolist()
    
    def fill_region(self, grid: Grid, target_color: int = 0, fill_color: int = 1) -> Grid:
        """Fill all regions of target_color with fill_color"""
        grid_array = np.array(grid)
        filled = grid_array.copy()
        filled[grid_array == target_color] = fill_color
        return filled.tolist()
    
    def paint_pattern(self, grid: Grid, pattern: List[List[int]], position: tuple = (0, 0)) -> Grid:
        """Paint a pattern at specified position"""
        grid_array = np.array(grid)
        pattern_array = np.array(pattern)
        result = grid_array.copy()
        
        start_row, start_col = position
        end_row = start_row + pattern_array.shape[0]
        end_col = start_col + pattern_array.shape[1]
        
        # Ensure we don't go out of bounds
        if end_row <= grid_array.shape[0] and end_col <= grid_array.shape[1]:
            result[start_row:end_row, start_col:end_col] = pattern_array
        
        return result.tolist()
    
    # Logical Operations
    def grid_and(self, grid1: Grid, grid2: Grid) -> Grid:
        """Logical AND operation between two grids"""
        arr1, arr2 = np.array(grid1), np.array(grid2)
        # Treat non-zero as True
        result = ((arr1 != 0) & (arr2 != 0)).astype(int)
        return result.tolist()
    
    def grid_or(self, grid1: Grid, grid2: Grid) -> Grid:
        """Logical OR operation between two grids"""
        arr1, arr2 = np.array(grid1), np.array(grid2)
        result = ((arr1 != 0) | (arr2 != 0)).astype(int)
        return result.tolist()
    
    def grid_xor(self, grid1: Grid, grid2: Grid) -> Grid:
        """Logical XOR operation between two grids"""
        arr1, arr2 = np.array(grid1), np.array(grid2)
        result = ((arr1 != 0) ^ (arr2 != 0)).astype(int)
        return result.tolist()
    
    # Morphological Operations
    def dilate(self, grid: Grid, kernel_size: int = 3) -> Grid:
        """Morphological dilation operation"""
        from scipy import ndimage
        grid_array = np.array(grid)
        kernel = np.ones((kernel_size, kernel_size))
        dilated = ndimage.binary_dilation(grid_array != 0, structure=kernel).astype(int)
        return dilated.tolist()
    
    def erode(self, grid: Grid, kernel_size: int = 3) -> Grid:
        """Morphological erosion operation"""
        from scipy import ndimage
        grid_array = np.array(grid)
        kernel = np.ones((kernel_size, kernel_size))
        eroded = ndimage.binary_erosion(grid_array != 0, structure=kernel).astype(int)
        return eroded.tolist()
    
    def opening(self, grid: Grid, kernel_size: int = 3) -> Grid:
        """Morphological opening (erosion followed by dilation)"""
        eroded = self.erode(grid, kernel_size)
        opened = self.dilate(eroded, kernel_size)
        return opened
    
    def closing(self, grid: Grid, kernel_size: int = 3) -> Grid:
        """Morphological closing (dilation followed by erosion)"""
        dilated = self.dilate(grid, kernel_size)
        closed = self.erode(dilated, kernel_size)
        return closed
    
    # Object Operations
    def move_object(self, grid: Grid, object_color: int, dx: int, dy: int) -> Grid:
        """Move all pixels of specified color by dx, dy"""
        grid_array = np.array(grid)
        result = grid_array.copy()
        
        # Find object pixels
        object_mask = (grid_array == object_color)
        object_positions = np.where(object_mask)
        
        # Clear original positions
        result[object_mask] = 0
        
        # Place object at new positions
        new_rows = object_positions[0] + dy
        new_cols = object_positions[1] + dx
        
        # Keep only positions within bounds
        valid_mask = (
            (new_rows >= 0) & (new_rows < grid_array.shape[0]) &
            (new_cols >= 0) & (new_cols < grid_array.shape[1])
        )
        
        result[new_rows[valid_mask], new_cols[valid_mask]] = object_color
        
        return result.tolist()
    
    def duplicate_object(self, grid: Grid, object_color: int, positions: List[tuple]) -> Grid:
        """Duplicate object at specified positions"""
        grid_array = np.array(grid)
        result = grid_array.copy()
        
        # Find original object
        object_mask = (grid_array == object_color)
        object_positions = np.where(object_mask)
        
        if len(object_positions[0]) == 0:
            return grid  # No object found
        
        # Get object bounding box
        min_row, max_row = object_positions[0].min(), object_positions[0].max()
        min_col, max_col = object_positions[1].min(), object_positions[1].max()
        
        # Duplicate at each position
        for new_row, new_col in positions:
            for i, (row, col) in enumerate(zip(object_positions[0], object_positions[1])):
                rel_row = row - min_row
                rel_col = col - min_col
                target_row = new_row + rel_row
                target_col = new_col + rel_col
                
                if (0 <= target_row < grid_array.shape[0] and 
                    0 <= target_col < grid_array.shape[1]):
                    result[target_row, target_col] = object_color
        
        return result.tolist()
    
    def remove_object(self, grid: Grid, object_color: int) -> Grid:
        """Remove all pixels of specified color"""
        grid_array = np.array(grid)
        result = grid_array.copy()
        result[grid_array == object_color] = 0
        return result.tolist()
    
    # Pattern Operations
    def complete_pattern(self, grid: Grid, pattern_size: tuple = (2, 2)) -> Grid:
        """Complete repeating patterns in the grid"""
        grid_array = np.array(grid)
        h, w = grid_array.shape
        pattern_h, pattern_w = pattern_size
        
        if h % pattern_h != 0 or w % pattern_w != 0:
            return grid  # Cannot complete pattern
        
        # Extract first pattern block
        pattern = grid_array[:pattern_h, :pattern_w]
        
        # Tile the pattern across the grid
        result = np.tile(pattern, (h // pattern_h, w // pattern_w))
        
        return result.tolist()
    
    def extend_pattern(self, grid: Grid, direction: str = 'right', steps: int = 1) -> Grid:
        """Extend detected pattern in specified direction"""
        grid_array = np.array(grid)
        
        if direction == 'right':
            # Simple extension: repeat last column
            last_col = grid_array[:, -1:]
            extension = np.tile(last_col, (1, steps))
            result = np.hstack([grid_array, extension])
        elif direction == 'down':
            # Simple extension: repeat last row
            last_row = grid_array[-1:, :]
            extension = np.tile(last_row, (steps, 1))
            result = np.vstack([grid_array, extension])
        else:
            return grid  # Unsupported direction
        
        return result.tolist()
    
    # Utility Operations
    def identity(self, grid: Grid) -> Grid:
        """Return grid unchanged"""
        return grid
    
    def resize(self, grid: Grid, target_shape: tuple) -> Grid:
        """Resize grid to target shape"""
        grid_array = np.array(grid)
        target_h, target_w = target_shape
        
        # Simple resize: crop or pad
        current_h, current_w = grid_array.shape
        
        if target_h <= current_h and target_w <= current_w:
            # Crop
            result = grid_array[:target_h, :target_w]
        else:
            # Pad with zeros
            result = np.zeros((target_h, target_w), dtype=grid_array.dtype)
            result[:min(current_h, target_h), :min(current_w, target_w)] = \
                grid_array[:min(current_h, target_h), :min(current_w, target_w)]
        
        return result.tolist()
    
    def crop(self, grid: Grid, bbox: tuple) -> Grid:
        """Crop grid to specified bounding box (min_row, min_col, max_row, max_col)"""
        grid_array = np.array(grid)
        min_row, min_col, max_row, max_col = bbox
        
        # Ensure bounds are valid
        h, w = grid_array.shape
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(h, max_row + 1)
        max_col = min(w, max_col + 1)
        
        cropped = grid_array[min_row:max_row, min_col:max_col]
        return cropped.tolist()
    
    def pad(self, grid: Grid, padding: int = 1, fill_value: int = 0) -> Grid:
        """Pad grid with specified value"""
        grid_array = np.array(grid)
        padded = np.pad(grid_array, padding, mode='constant', constant_values=fill_value)
        return padded.tolist()
