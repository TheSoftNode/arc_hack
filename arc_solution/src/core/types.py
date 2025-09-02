"""
ARC Prize 2025 - Core Data Structures and Types

This module defines the fundamental data structures used throughout
the multi-agent neuro-symbolic reasoning system.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np


class ColorCode(Enum):
    """ARC color codes (0-9)"""
    BLACK = 0
    BLUE = 1  
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7
    SKY = 8
    BROWN = 9


# Type aliases for clarity
Grid = List[List[int]]
GridArray = np.ndarray
Position = Tuple[int, int]  # (row, col)
BoundingBox = Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)


@dataclass
class Object:
    """Represents a detected object in the grid"""
    color: int
    positions: List[Position]
    bounding_box: BoundingBox
    centroid: Tuple[float, float]
    size: int
    shape_type: Optional[str] = None  # 'rectangle', 'line', 'L-shape', etc.
    
    def __post_init__(self):
        if self.size == 0:
            self.size = len(self.positions)


@dataclass
class Task:
    """Represents a single ARC task"""
    task_id: str
    train_pairs: List[Dict[str, Grid]]
    test_inputs: List[Grid]
    test_outputs: Optional[List[Grid]] = None  # For evaluation
    
    def __post_init__(self):
        # Validate structure
        for pair in self.train_pairs:
            assert 'input' in pair and 'output' in pair
            assert isinstance(pair['input'], list)
            assert isinstance(pair['output'], list)


@dataclass  
class Transformation:
    """Represents a transformation rule/hypothesis"""
    rule_type: str
    parameters: Dict[str, Any]
    confidence: float
    description: str
    code: Optional[str] = None  # Generated code for execution
    
    def apply(self, grid: Grid) -> Grid:
        """Apply this transformation to a grid"""
        # This will be implemented by specific transformation types
        raise NotImplementedError


@dataclass
class Hypothesis:
    """A hypothesis about how to solve a task"""
    transformations: List[Transformation]
    confidence: float
    description: str
    generated_by: str  # Which component generated this hypothesis
    reasoning: Optional[str] = None  # LLM reasoning if applicable
    
    def apply_to_grid(self, grid: Grid) -> Grid:
        """Apply all transformations in sequence"""
        result = grid
        for transform in self.transformations:
            result = transform.apply(result)
        return result


@dataclass
class SpatialRelation:
    """Represents spatial relationship between objects"""
    relation_type: str  # 'above', 'below', 'left', 'right', 'inside', 'adjacent'
    object1_id: int
    object2_id: int
    distance: Optional[float] = None
    

@dataclass
class SceneRepresentation:
    """Multi-modal representation of a grid"""
    grid: Grid
    grid_array: GridArray
    objects: List[Object]
    spatial_relations: List[SpatialRelation]
    graph: Optional[Any] = None  # NetworkX graph
    features: Optional[Dict[str, Any]] = None  # Extracted features
    
    def __post_init__(self):
        if self.grid_array is None:
            self.grid_array = np.array(self.grid)
            
    @property
    def height(self) -> int:
        return len(self.grid)
    
    @property 
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class TaskSolution:
    """Represents a complete solution to a task"""
    task_id: str
    hypotheses: List[Hypothesis]
    predictions: List[Dict[str, Grid]]  # attempt_1, attempt_2 for each test input
    confidence_scores: List[float]
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class DSLOperation:
    """Represents an operation in our Domain Specific Language"""
    name: str
    parameters: Dict[str, Any]
    return_type: str
    description: str
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute this DSL operation"""
        raise NotImplementedError
        

class PrimitiveType(Enum):
    """Types of primitive operations in our DSL"""
    GEOMETRIC = "geometric"  # rotate, translate, scale, reflect
    COLOR = "color"  # change colors, fill regions
    LOGICAL = "logical"  # AND, OR, XOR operations
    MORPHOLOGICAL = "morphological"  # dilation, erosion, opening, closing
    PATTERN = "pattern"  # pattern detection and completion
    OBJECT = "object"  # object manipulation and detection
    SPATIAL = "spatial"  # spatial transformations and relationships


# Configuration constants
ARC_COLORS = list(range(10))  # 0-9
MAX_GRID_SIZE = 30
MIN_GRID_SIZE = 1
MAX_TASK_TIME_SECONDS = 30  # Per task execution limit
