"""
ARC Prize 2025 - Multi-Modal Preprocessor

This module converts raw grids into rich multi-modal representations
for downstream reasoning components.
"""

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from typing import Dict, Any, List, Tuple
try:
    import networkx as nx
except ImportError:
    nx = None
try:
    from scipy import ndimage
except ImportError:
    ndimage = None

from .types import Grid, SceneRepresentation, Object, SpatialRelation, Position, BoundingBox


class MultiModalPreprocessor:
    """
    Converts raw ARC grids into multi-modal representations:
    - Native grid representation
    - Object-based representation
    - Graph representation of spatial relationships
    - Feature extraction for pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_object_detection = config.get('enable_object_detection', True)
        self.enable_graph_representation = config.get('enable_graph_representation', True)
        self.enable_feature_extraction = config.get('enable_feature_extraction', True)
        
    def process_grid(self, grid: Grid) -> SceneRepresentation:
        """
        Convert a raw grid into a comprehensive multi-modal representation.
        
        Args:
            grid: Raw ARC grid (list of lists)
            
        Returns:
            SceneRepresentation with all computed features
        """
        # Convert to numpy array for processing
        grid_array = np.array(grid, dtype=np.int32)
        
        # Initialize representation
        scene = SceneRepresentation(
            grid=grid,
            grid_array=grid_array,
            objects=[],
            spatial_relations=[],
            graph=None,
            features={}
        )
        
        # Extract objects if enabled
        if self.enable_object_detection:
            scene.objects = self._extract_objects(grid_array)
            
        # Build spatial relationship graph if enabled  
        if self.enable_graph_representation and scene.objects:
            scene.spatial_relations = self._compute_spatial_relations(scene.objects)
            scene.graph = self._build_scene_graph(scene.objects, scene.spatial_relations)
            
        # Extract high-level features if enabled
        if self.enable_feature_extraction:
            scene.features = self._extract_features(grid_array, scene.objects)
            
        return scene
    
    def _extract_objects(self, grid_array: np.ndarray) -> List[Object]:
        """Extract distinct objects from the grid using connected components"""
        objects = []
        
        # Find connected components for each color (except background/0)
        unique_colors = np.unique(grid_array)
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
                
            # Create binary mask for this color
            color_mask = (grid_array == color).astype(np.uint8)
            
            # Find connected components
            if cv2 is not None:
                num_labels, labels = cv2.connectedComponents(color_mask)
            elif ndimage is not None:
                # Fallback: use scipy.ndimage.label
                labels, num_labels = ndimage.label(color_mask)
            else:
                # Simple fallback: treat each non-zero pixel as separate object
                num_labels = 1
                labels = color_mask
            
            for label in range(1, num_labels):  # Skip background label 0
                # Get positions of this component
                positions = np.where(labels == label)
                position_list = list(zip(positions[0].tolist(), positions[1].tolist()))
                
                if len(position_list) < 1:  # Skip empty components
                    continue
                    
                # Compute bounding box
                min_row, max_row = positions[0].min(), positions[0].max()
                min_col, max_col = positions[1].min(), positions[1].max()
                bounding_box = (min_row, min_col, max_row, max_col)
                
                # Compute centroid
                centroid = (positions[0].mean(), positions[1].mean())
                
                # Determine basic shape type
                shape_type = self._classify_shape(position_list, bounding_box)
                
                # Create object
                obj = Object(
                    color=int(color),
                    positions=position_list,
                    bounding_box=bounding_box,
                    centroid=centroid,
                    size=len(position_list),
                    shape_type=shape_type
                )
                
                objects.append(obj)
        
        return objects
    
    def _classify_shape(self, positions: List[Position], bbox: BoundingBox) -> str:
        """Classify the basic shape type of an object"""
        min_row, min_col, max_row, max_col = bbox
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        size = len(positions)
        
        # Rectangle/square detection
        expected_rect_size = height * width
        if size == expected_rect_size:
            if height == width:
                return "square"
            else:
                return "rectangle"
                
        # Line detection
        if height == 1 or width == 1:
            return "line"
            
        # Single pixel
        if size == 1:
            return "pixel"
            
        # L-shape, T-shape detection could be added here
        # For now, classify as "complex"
        return "complex"
    
    def _compute_spatial_relations(self, objects: List[Object]) -> List[SpatialRelation]:
        """Compute spatial relationships between objects"""
        relations = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:  # Avoid duplicates and self-relations
                    continue
                    
                # Compute relationship type and distance
                relation_type, distance = self._analyze_spatial_relationship(obj1, obj2)
                
                if relation_type:  # Only add if there's a meaningful relationship
                    relation = SpatialRelation(
                        relation_type=relation_type,
                        object1_id=i,
                        object2_id=j,
                        distance=distance
                    )
                    relations.append(relation)
        
        return relations
    
    def _analyze_spatial_relationship(self, obj1: Object, obj2: Object) -> Tuple[str, float]:
        """Analyze spatial relationship between two objects"""
        # Get centroids
        c1_row, c1_col = obj1.centroid
        c2_row, c2_col = obj2.centroid
        
        # Compute Euclidean distance
        distance = np.sqrt((c1_row - c2_row)**2 + (c1_col - c2_col)**2)
        
        # Determine relative position
        row_diff = c2_row - c1_row
        col_diff = c2_col - c1_col
        
        # Classification thresholds
        if abs(row_diff) > abs(col_diff):
            if row_diff > 0:
                relation = "below"
            else:
                relation = "above" 
        else:
            if col_diff > 0:
                relation = "right"
            else:
                relation = "left"
                
        # Check for adjacency (distance threshold)
        if distance <= 2.0:
            relation = "adjacent_" + relation
            
        # Check for containment
        if self._is_contained(obj1, obj2):
            relation = "contains"
        elif self._is_contained(obj2, obj1):
            relation = "contained_by"
            
        return relation, distance
    
    def _is_contained(self, outer_obj: Object, inner_obj: Object) -> bool:
        """Check if inner_obj is contained within outer_obj's bounding box"""
        outer_bbox = outer_obj.bounding_box
        inner_bbox = inner_obj.bounding_box
        
        return (outer_bbox[0] <= inner_bbox[0] and  # min_row
                outer_bbox[1] <= inner_bbox[1] and  # min_col
                outer_bbox[2] >= inner_bbox[2] and  # max_row
                outer_bbox[3] >= inner_bbox[3])     # max_col
    
    def _build_scene_graph(self, objects: List[Object], relations: List[SpatialRelation]):
        """Build a NetworkX graph representing the scene structure"""
        if nx is None:
            return None  # Return None if networkx is not available
            
        G = nx.Graph()
        
        # Add nodes for each object
        for i, obj in enumerate(objects):
            G.add_node(i, 
                      color=obj.color,
                      size=obj.size,
                      shape_type=obj.shape_type,
                      centroid=obj.centroid,
                      bounding_box=obj.bounding_box)
        
        # Add edges for spatial relationships
        for rel in relations:
            G.add_edge(rel.object1_id, rel.object2_id,
                      relation_type=rel.relation_type,
                      distance=rel.distance)
        
        return G
    
    def _extract_features(self, grid_array: np.ndarray, objects: List[Object]) -> Dict[str, Any]:
        """Extract high-level features for pattern recognition"""
        features = {}
        
        # Grid-level features
        features['grid_shape'] = grid_array.shape
        features['unique_colors'] = len(np.unique(grid_array))
        features['background_color'] = self._get_background_color(grid_array)
        features['color_distribution'] = self._get_color_distribution(grid_array)
        
        # Object-level features
        features['num_objects'] = len(objects)
        features['object_colors'] = [obj.color for obj in objects]
        features['object_sizes'] = [obj.size for obj in objects]
        features['object_shapes'] = [obj.shape_type for obj in objects]
        
        # Symmetry features
        features['vertical_symmetry'] = self._check_vertical_symmetry(grid_array)
        features['horizontal_symmetry'] = self._check_horizontal_symmetry(grid_array)
        features['rotational_symmetry'] = self._check_rotational_symmetry(grid_array)
        
        # Pattern features
        features['has_repeating_pattern'] = self._detect_repeating_patterns(grid_array)
        features['grid_density'] = np.mean(grid_array != 0)  # Non-background ratio
        
        return features
    
    def _get_background_color(self, grid_array: np.ndarray) -> int:
        """Determine the most likely background color"""
        unique, counts = np.unique(grid_array, return_counts=True)
        return int(unique[np.argmax(counts)])
    
    def _get_color_distribution(self, grid_array: np.ndarray) -> Dict[int, float]:
        """Get normalized color distribution"""
        unique, counts = np.unique(grid_array, return_counts=True)
        total = grid_array.size
        return {int(color): count/total for color, count in zip(unique, counts)}
    
    def _check_vertical_symmetry(self, grid_array: np.ndarray) -> bool:
        """Check if grid has vertical line symmetry"""
        return np.array_equal(grid_array, np.fliplr(grid_array))
    
    def _check_horizontal_symmetry(self, grid_array: np.ndarray) -> bool:
        """Check if grid has horizontal line symmetry"""
        return np.array_equal(grid_array, np.flipud(grid_array))
    
    def _check_rotational_symmetry(self, grid_array: np.ndarray) -> bool:
        """Check if grid has 180-degree rotational symmetry"""
        return np.array_equal(grid_array, np.rot90(grid_array, 2))
    
    def _detect_repeating_patterns(self, grid_array: np.ndarray) -> bool:
        """Detect if the grid contains repeating patterns"""
        # Simple pattern detection - check for 2x2 or 3x3 repeating blocks
        h, w = grid_array.shape
        
        # Check 2x2 patterns
        if h >= 4 and w >= 4:
            for block_h in [2, 3]:
                for block_w in [2, 3]:
                    if h % block_h == 0 and w % block_w == 0:
                        # Extract first block
                        first_block = grid_array[:block_h, :block_w]
                        
                        # Check if pattern repeats
                        is_repeating = True
                        for i in range(0, h, block_h):
                            for j in range(0, w, block_w):
                                block = grid_array[i:i+block_h, j:j+block_w]
                                if not np.array_equal(block, first_block):
                                    is_repeating = False
                                    break
                            if not is_repeating:
                                break
                        
                        if is_repeating:
                            return True
        
        return False
