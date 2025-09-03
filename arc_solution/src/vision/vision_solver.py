"""
ARC Prize 2025 - Computer Vision Solver

This module implements computer vision approaches for solving ARC tasks,
including pattern recognition, shape analysis, and visual transformations.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage import measure
    from skimage.feature import match_template
except ImportError:
    measure = morphology = segmentation = match_template = None

from ..core.types import Grid, Task, Hypothesis, Object, Position


@dataclass
class VisualPattern:
    """Represents a detected visual pattern"""
    pattern_type: str  # 'shape', 'color_sequence', 'spatial_relation', etc.
    template: np.ndarray
    positions: List[Position]
    confidence: float
    metadata: Dict[str, Any]


class VisionSolver:
    """
    Computer vision-based solver for ARC tasks.
    
    Uses traditional CV techniques:
    - Template matching
    - Shape analysis
    - Connected component analysis
    - Morphological operations
    - Contour detection
    """
    
    def __init__(self):
        self.min_confidence = 0.3
        self.shape_templates = self._build_shape_templates()
    
    def _build_shape_templates(self) -> Dict[str, np.ndarray]:
        """Build library of common shape templates"""
        templates = {}
        
        # Basic shapes
        templates['square_3x3'] = np.ones((3, 3), dtype=np.uint8)
        templates['square_2x2'] = np.ones((2, 2), dtype=np.uint8)
        
        # Lines
        templates['horizontal_line'] = np.ones((1, 3), dtype=np.uint8)
        templates['vertical_line'] = np.ones((3, 1), dtype=np.uint8)
        
        # L-shapes
        l_shape = np.zeros((3, 3), dtype=np.uint8)
        l_shape[0, :] = 1
        l_shape[:, 0] = 1
        templates['l_shape'] = l_shape
        
        # T-shapes
        t_shape = np.zeros((3, 3), dtype=np.uint8)
        t_shape[0, :] = 1
        t_shape[:, 1] = 1
        templates['t_shape'] = t_shape
        
        # Plus shape
        plus_shape = np.zeros((3, 3), dtype=np.uint8)
        plus_shape[1, :] = 1
        plus_shape[:, 1] = 1
        templates['plus_shape'] = plus_shape
        
        return templates
    
    def generate_hypotheses(self, task: Task, max_hypotheses: int = 10) -> List[Hypothesis]:
        """
        Generate hypotheses using computer vision analysis
        
        Args:
            task: The ARC task to analyze
            max_hypotheses: Maximum number of hypotheses to return
            
        Returns:
            List of vision-based hypotheses
        """
        hypotheses = []
        
        try:
            # Analyze visual patterns in training examples
            patterns = self._analyze_visual_patterns(task)
            
            # Generate transformation hypotheses based on patterns
            pattern_hypotheses = self._generate_pattern_hypotheses(patterns, task)
            hypotheses.extend(pattern_hypotheses)
            
            # Analyze shape transformations
            shape_hypotheses = self._analyze_shape_transformations(task)
            hypotheses.extend(shape_hypotheses)
            
            # Analyze color transformations
            color_hypotheses = self._analyze_color_transformations(task)
            hypotheses.extend(color_hypotheses)
            
            # Analyze spatial relationships
            spatial_hypotheses = self._analyze_spatial_relationships(task)
            hypotheses.extend(spatial_hypotheses)
            
            # Sort by confidence and return top candidates
            hypotheses.sort(key=lambda h: h.confidence, reverse=True)
            return hypotheses[:max_hypotheses]
        
        except Exception as e:
            print(f"Error in vision solver: {e}")
            return []
    
    def _analyze_visual_patterns(self, task: Task) -> List[VisualPattern]:
        """Analyze visual patterns across training examples"""
        patterns = []
        
        for pair in task.train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # Template matching for known shapes
            input_patterns = self._detect_template_patterns(input_grid)
            output_patterns = self._detect_template_patterns(output_grid)
            
            patterns.extend(input_patterns)
            patterns.extend(output_patterns)
            
            # Detect custom patterns
            custom_patterns = self._detect_custom_patterns(input_grid, output_grid)
            patterns.extend(custom_patterns)
        
        return patterns
    
    def _detect_template_patterns(self, grid: np.ndarray) -> List[VisualPattern]:
        """Detect known shape templates in grid"""
        patterns = []
        
        if match_template is None:
            return patterns
        
        for template_name, template in self.shape_templates.items():
            try:
                # Convert grid to match template format
                if len(grid.shape) == 2:
                    grid_binary = (grid > 0).astype(np.uint8)
                else:
                    grid_binary = grid
                
                # Template matching
                result = match_template(grid_binary, template)
                locations = np.where(result >= 0.8)  # High threshold for exact matches
                
                for y, x in zip(locations[0], locations[1]):
                    pattern = VisualPattern(
                        pattern_type=template_name,
                        template=template,
                        positions=[(y, x)],
                        confidence=float(result[y, x]),
                        metadata={'template_name': template_name}
                    )
                    patterns.append(pattern)
            
            except Exception as e:
                continue  # Skip failed template matches
        
        return patterns
    
    def _detect_custom_patterns(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[VisualPattern]:
        """Detect custom patterns by comparing input and output"""
        patterns = []
        
        try:
            # Find connected components
            input_objects = self._find_connected_components(input_grid)
            output_objects = self._find_connected_components(output_grid)
            
            # Analyze object transformations
            for input_obj in input_objects:
                for output_obj in output_objects:
                    if self._objects_related(input_obj, output_obj):
                        pattern = self._extract_transformation_pattern(input_obj, output_obj)
                        if pattern:
                            patterns.append(pattern)
        
        except Exception as e:
            pass  # Continue with other analyses
        
        return patterns
    
    def _find_connected_components(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Find connected components (objects) in grid"""
        objects = []
        
        if measure is None:
            return objects
        
        try:
            # Find connected components for each color
            unique_colors = np.unique(grid)
            
            for color in unique_colors:
                if color == 0:  # Skip background
                    continue
                
                color_mask = (grid == color).astype(np.uint8)
                labeled = measure.label(color_mask)
                
                for region in measure.regionprops(labeled):
                    obj = {
                        'color': color,
                        'area': region.area,
                        'bbox': region.bbox,
                        'centroid': region.centroid,
                        'coords': region.coords,
                        'filled_area': region.filled_area,
                        'perimeter': region.perimeter
                    }
                    objects.append(obj)
        
        except Exception as e:
            pass
        
        return objects
    
    def _objects_related(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """Check if two objects are related (similar size, shape, etc.)"""
        try:
            # Similar area
            area_ratio = min(obj1['area'], obj2['area']) / max(obj1['area'], obj2['area'])
            if area_ratio < 0.5:
                return False
            
            # Similar aspect ratio
            bbox1 = obj1['bbox']
            bbox2 = obj2['bbox']
            
            h1, w1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
            h2, w2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
            
            if h1 > 0 and w1 > 0 and h2 > 0 and w2 > 0:
                aspect1 = h1 / w1
                aspect2 = h2 / w2
                aspect_ratio = min(aspect1, aspect2) / max(aspect1, aspect2)
                if aspect_ratio < 0.7:
                    return False
            
            return True
        
        except:
            return False
    
    def _extract_transformation_pattern(self, input_obj: Dict[str, Any], output_obj: Dict[str, Any]) -> Optional[VisualPattern]:
        """Extract transformation pattern between two related objects"""
        try:
            # Calculate transformation
            input_center = input_obj['centroid']
            output_center = output_obj['centroid']
            
            dx = output_center[1] - input_center[1]
            dy = output_center[0] - input_center[0]
            
            # Color change
            color_changed = input_obj['color'] != output_obj['color']
            
            # Create pattern
            pattern = VisualPattern(
                pattern_type='object_transformation',
                template=np.array([[input_obj['color']]]),  # Simple template
                positions=[(int(input_center[0]), int(input_center[1]))],
                confidence=0.7,
                metadata={
                    'translation': (dx, dy),
                    'color_change': color_changed,
                    'input_color': input_obj['color'],
                    'output_color': output_obj['color'],
                    'area_change': output_obj['area'] - input_obj['area']
                }
            )
            
            return pattern
        
        except:
            return None
    
    def _generate_pattern_hypotheses(self, patterns: List[VisualPattern], task: Task) -> List[Hypothesis]:
        """Generate hypotheses based on detected patterns"""
        hypotheses = []
        
        # Group patterns by type
        pattern_groups = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(pattern)
        
        # Generate hypotheses for each pattern type
        for pattern_type, pattern_list in pattern_groups.items():
            if len(pattern_list) >= 2:  # Need multiple instances to establish pattern
                hypothesis = self._create_pattern_hypothesis(pattern_type, pattern_list)
                if hypothesis:
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _create_pattern_hypothesis(self, pattern_type: str, patterns: List[VisualPattern]) -> Optional[Hypothesis]:
        """Create hypothesis from pattern instances"""
        try:
            avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
            
            if pattern_type == 'object_transformation':
                # Analyze common transformations
                translations = [p.metadata.get('translation', (0, 0)) for p in patterns]
                color_changes = [p.metadata.get('color_change', False) for p in patterns]
                
                # Find most common transformation
                if translations:
                    common_translation = max(set(translations), key=translations.count)
                    
                    hypothesis = Hypothesis(
                        transformations=[],  # Will be filled by execution engine
                        confidence=min(avg_confidence, 0.8),
                        description=f"Move objects by {common_translation}",
                        generated_by="vision_solver",
                        metadata={
                            'pattern_type': pattern_type,
                            'transformation': common_translation,
                            'color_changes': any(color_changes),
                            'pattern_count': len(patterns)
                        }
                    )
                    return hypothesis
            
            elif pattern_type in self.shape_templates:
                hypothesis = Hypothesis(
                    transformations=[],
                    confidence=avg_confidence,
                    description=f"Detected {pattern_type} pattern",
                    generated_by="vision_solver",
                    metadata={
                        'pattern_type': pattern_type,
                        'pattern_count': len(patterns)
                    }
                )
                return hypothesis
        
        except:
            pass
        
        return None
    
    def _analyze_shape_transformations(self, task: Task) -> List[Hypothesis]:
        """Analyze geometric shape transformations"""
        hypotheses = []
        
        for pair in task.train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # Check for rotation
            if self._grids_are_rotated(input_grid, output_grid):
                hypothesis = Hypothesis(
                    transformations=[],
                    confidence=0.9,
                    description="Grid rotation transformation",
                    generated_by="vision_solver",
                    metadata={'transformation_type': 'rotation'}
                )
                hypotheses.append(hypothesis)
            
            # Check for reflection
            if self._grids_are_reflected(input_grid, output_grid):
                hypothesis = Hypothesis(
                    transformations=[],
                    confidence=0.9,
                    description="Grid reflection transformation",
                    generated_by="vision_solver",
                    metadata={'transformation_type': 'reflection'}
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _grids_are_rotated(self, grid1: np.ndarray, grid2: np.ndarray) -> bool:
        """Check if grid2 is a rotation of grid1"""
        try:
            # Check 90, 180, 270 degree rotations
            for k in [1, 2, 3]:
                rotated = np.rot90(grid1, k)
                if np.array_equal(rotated, grid2):
                    return True
            return False
        except:
            return False
    
    def _grids_are_reflected(self, grid1: np.ndarray, grid2: np.ndarray) -> bool:
        """Check if grid2 is a reflection of grid1"""
        try:
            # Check horizontal and vertical reflections
            h_reflected = np.fliplr(grid1)
            v_reflected = np.flipud(grid1)
            
            return np.array_equal(h_reflected, grid2) or np.array_equal(v_reflected, grid2)
        except:
            return False
    
    def _analyze_color_transformations(self, task: Task) -> List[Hypothesis]:
        """Analyze color-based transformations"""
        hypotheses = []
        
        for pair in task.train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # Check for simple color replacement
            color_mapping = self._find_color_mapping(input_grid, output_grid)
            if color_mapping:
                hypothesis = Hypothesis(
                    transformations=[],
                    confidence=0.8,
                    description=f"Color mapping: {color_mapping}",
                    generated_by="vision_solver",
                    metadata={
                        'transformation_type': 'color_mapping',
                        'color_mapping': color_mapping
                    }
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _find_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict[int, int]]:
        """Find color mapping between input and output grids"""
        try:
            if input_grid.shape != output_grid.shape:
                return None
            
            mapping = {}
            for input_color in np.unique(input_grid):
                # Find what this color becomes in output
                mask = (input_grid == input_color)
                output_colors = output_grid[mask]
                
                if len(np.unique(output_colors)) == 1:
                    # Consistent mapping
                    output_color = output_colors[0]
                    mapping[int(input_color)] = int(output_color)
                else:
                    # Inconsistent mapping, not a simple color replacement
                    return None
            
            # Check if mapping is non-trivial
            if all(k == v for k, v in mapping.items()):
                return None  # Identity mapping
            
            return mapping
        
        except:
            return None
    
    def _analyze_spatial_relationships(self, task: Task) -> List[Hypothesis]:
        """Analyze spatial relationship transformations"""
        hypotheses = []
        
        # Placeholder for spatial relationship analysis
        # This would analyze how objects relate to each other spatially
        # and how these relationships change
        
        return hypotheses
