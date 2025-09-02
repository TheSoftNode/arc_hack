"""
ARC Prize 2025 - Symbolic Solver

This module implements symbolic reasoning and program synthesis
for ARC task solving using a Domain Specific Language (DSL).
"""

import logging
from typing import List, Dict, Any, Optional
import time

from ..core.types import Task, Hypothesis, Transformation, Grid, SceneRepresentation
from ..dsl.primitives import DSLPrimitives
from ..dsl.program_synthesizer import ProgramSynthesizer

logger = logging.getLogger(__name__)


class SymbolicSolver:
    """
    Symbolic reasoning engine that uses program synthesis to generate
    hypotheses about ARC task transformations.
    
    Core approach:
    1. Analyze input-output pairs to identify transformation patterns
    2. Use DSL primitives to construct candidate programs
    3. Validate programs against all training examples
    4. Generate ranked hypotheses
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_search_depth = config.get('max_search_depth', 5)
        self.timeout_seconds = config.get('timeout_seconds', 10)
        self.use_z3_solver = config.get('use_z3_solver', True)
        
        # Initialize DSL and program synthesizer
        self.dsl = DSLPrimitives()
        self.synthesizer = ProgramSynthesizer(self.dsl, config)
        
        # Statistics tracking
        self.stats = {
            'hypotheses_generated': 0,
            'successful_validations': 0,
            'synthesis_time': 0.0
        }
    
    def generate_hypotheses(self, task: Task, scenes: List[Any]) -> List[Hypothesis]:
        """
        Generate symbolic hypotheses for the given task.
        
        Args:
            task: The ARC task to solve
            scenes: Preprocessed scene representations
            
        Returns:
            List of hypotheses ordered by confidence
        """
        start_time = time.time()
        logger.info(f"Starting symbolic reasoning for task {task.task_id}")
        
        hypotheses = []
        
        try:
            # Extract training pairs from scenes
            training_pairs = self._extract_training_pairs(scenes)
            
            if not training_pairs:
                logger.warning("No training pairs found in scenes")
                return []
            
            # Analyze transformation patterns
            transformation_patterns = self._analyze_transformation_patterns(training_pairs)
            
            # Generate candidate programs for each pattern
            for pattern in transformation_patterns:
                candidate_programs = self._synthesize_programs_for_pattern(pattern, training_pairs)
                
                # Convert programs to hypotheses
                for program in candidate_programs:
                    hypothesis = self._program_to_hypothesis(program, pattern)
                    if hypothesis:
                        hypotheses.append(hypothesis)
            
            # Filter and rank hypotheses
            hypotheses = self._rank_hypotheses(hypotheses, training_pairs)
            
        except Exception as e:
            logger.error(f"Error in symbolic reasoning: {str(e)}")
        
        finally:
            self.stats['synthesis_time'] += time.time() - start_time
            self.stats['hypotheses_generated'] += len(hypotheses)
            
        logger.info(f"Generated {len(hypotheses)} symbolic hypotheses")
        return hypotheses
    
    def _extract_training_pairs(self, scenes: List[Any]) -> List[Dict[str, Any]]:
        """Extract training input-output pairs from scene representations"""
        training_pairs = []
        
        for scene in scenes:
            if 'input' in scene and 'output' in scene:
                training_pairs.append({
                    'input_scene': scene['input'],
                    'output_scene': scene['output'],
                    'pair_id': scene.get('pair_id', len(training_pairs))
                })
        
        return training_pairs
    
    def _analyze_transformation_patterns(self, training_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze training pairs to identify potential transformation patterns.
        
        Returns patterns like:
        - Geometric transformations (rotate, reflect, translate)
        - Color transformations (recolor, fill patterns)
        - Object manipulations (move, resize, duplicate)
        - Logical operations (AND, OR, XOR)
        """
        patterns = []
        
        for pair in training_pairs:
            input_scene = pair['input_scene']
            output_scene = pair['output_scene']
            
            # Pattern 1: Size changes
            size_pattern = self._analyze_size_changes(input_scene, output_scene)
            if size_pattern:
                patterns.append(size_pattern)
            
            # Pattern 2: Color transformations
            color_pattern = self._analyze_color_changes(input_scene, output_scene)
            if color_pattern:
                patterns.append(color_pattern)
                
            # Pattern 3: Geometric transformations
            geometric_pattern = self._analyze_geometric_changes(input_scene, output_scene)
            if geometric_pattern:
                patterns.append(geometric_pattern)
                
            # Pattern 4: Object movement/manipulation
            object_pattern = self._analyze_object_changes(input_scene, output_scene)
            if object_pattern:
                patterns.append(object_pattern)
        
        # Remove duplicate patterns
        unique_patterns = self._deduplicate_patterns(patterns)
        
        return unique_patterns
    
    def _analyze_size_changes(self, input_scene: SceneRepresentation, 
                             output_scene: SceneRepresentation) -> Optional[Dict[str, Any]]:
        """Analyze if there are size/shape changes"""
        input_shape = input_scene.shape
        output_shape = output_scene.shape
        
        if input_shape != output_shape:
            return {
                'type': 'size_change',
                'input_shape': input_shape,
                'output_shape': output_shape,
                'scale_factor': (output_shape[0] / input_shape[0], output_shape[1] / input_shape[1]) if input_shape[0] > 0 and input_shape[1] > 0 else None
            }
        
        return None
    
    def _analyze_color_changes(self, input_scene: SceneRepresentation,
                              output_scene: SceneRepresentation) -> Optional[Dict[str, Any]]:
        """Analyze color transformation patterns"""
        input_colors = set(input_scene.features.get('object_colors', []))
        output_colors = set(output_scene.features.get('object_colors', []))
        
        # Check for color mapping
        if input_colors != output_colors:
            return {
                'type': 'color_change',
                'input_colors': input_colors,
                'output_colors': output_colors,
                'new_colors': output_colors - input_colors,
                'removed_colors': input_colors - output_colors
            }
        
        return None
    
    def _analyze_geometric_changes(self, input_scene: SceneRepresentation,
                                  output_scene: SceneRepresentation) -> Optional[Dict[str, Any]]:
        """Analyze geometric transformations"""
        if input_scene.shape == output_scene.shape:
            # Check for rotation/reflection by comparing grid patterns
            input_grid = input_scene.grid_array
            output_grid = output_scene.grid_array
            
            # Test rotations
            for rotation in [90, 180, 270]:
                import numpy as np
                rotated = np.rot90(input_grid, k=rotation//90)
                if np.array_equal(rotated, output_grid):
                    return {
                        'type': 'rotation',
                        'angle': rotation
                    }
            
            # Test reflections
            if np.array_equal(np.fliplr(input_grid), output_grid):
                return {'type': 'reflection', 'axis': 'vertical'}
            if np.array_equal(np.flipud(input_grid), output_grid):
                return {'type': 'reflection', 'axis': 'horizontal'}
        
        return None
    
    def _analyze_object_changes(self, input_scene: SceneRepresentation,
                               output_scene: SceneRepresentation) -> Optional[Dict[str, Any]]:
        """Analyze object-level changes"""
        input_objects = input_scene.objects
        output_objects = output_scene.objects
        
        # Check for object count changes
        if len(input_objects) != len(output_objects):
            return {
                'type': 'object_count_change',
                'input_count': len(input_objects),
                'output_count': len(output_objects)
            }
        
        # Check for object movement (same objects, different positions)
        if len(input_objects) == len(output_objects) and len(input_objects) > 0:
            # Simple heuristic: check if object centroids changed
            input_centroids = [obj.centroid for obj in input_objects]
            output_centroids = [obj.centroid for obj in output_objects]
            
            if input_centroids != output_centroids:
                return {
                    'type': 'object_movement',
                    'input_centroids': input_centroids,
                    'output_centroids': output_centroids
                }
        
        return None
    
    def _deduplicate_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate patterns"""
        unique_patterns = []
        seen_types = set()
        
        for pattern in patterns:
            pattern_type = pattern.get('type')
            if pattern_type and pattern_type not in seen_types:
                unique_patterns.append(pattern)
                seen_types.add(pattern_type)
        
        return unique_patterns
    
    def _synthesize_programs_for_pattern(self, pattern: Dict[str, Any], 
                                       training_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate candidate programs for a specific transformation pattern"""
        pattern_type = pattern.get('type')
        programs = []
        
        if pattern_type == 'rotation':
            angle = pattern.get('angle', 90)
            programs.append({
                'operations': [{'name': 'rotate', 'angle': angle}],
                'description': f"Rotate grid by {angle} degrees"
            })
            
        elif pattern_type == 'reflection':
            axis = pattern.get('axis', 'vertical')
            programs.append({
                'operations': [{'name': 'reflect', 'axis': axis}],
                'description': f"Reflect grid along {axis} axis"
            })
            
        elif pattern_type == 'color_change':
            # Try simple color mapping
            input_colors = pattern.get('input_colors', set())
            output_colors = pattern.get('output_colors', set())
            
            programs.append({
                'operations': [{'name': 'recolor', 'color_map': dict(zip(input_colors, output_colors))}],
                'description': "Apply color mapping transformation"
            })
            
        elif pattern_type == 'size_change':
            input_shape = pattern.get('input_shape')
            output_shape = pattern.get('output_shape')
            
            if input_shape and output_shape:
                programs.append({
                    'operations': [{'name': 'resize', 'target_shape': output_shape}],
                    'description': f"Resize grid from {input_shape} to {output_shape}"
                })
        
        # Add fallback "identity" program
        programs.append({
            'operations': [{'name': 'identity'}],
            'description': "Return input unchanged"
        })
        
        return programs
    
    def _program_to_hypothesis(self, program: Dict[str, Any], pattern: Dict[str, Any]) -> Optional[Hypothesis]:
        """Convert a synthesized program to a hypothesis"""
        try:
            # Create transformations from operations
            transformations = []
            for op in program.get('operations', []):
                transform = Transformation(
                    rule_type=op.get('name', 'unknown'),
                    parameters=op,
                    confidence=0.7,  # Base confidence for symbolic solutions
                    description=f"Apply {op.get('name', 'unknown')} operation",
                    code=None  # Will be generated by execution engine
                )
                transformations.append(transform)
            
            # Create hypothesis
            hypothesis = Hypothesis(
                transformations=transformations,
                confidence=0.7,  # Base confidence
                description=program.get('description', 'Symbolic transformation'),
                generated_by='symbolic_solver',
                reasoning=f"Pattern detected: {pattern.get('type', 'unknown')}"
            )
            
            return hypothesis
            
        except Exception as e:
            logger.warning(f"Error creating hypothesis from program: {str(e)}")
            return None
    
    def _rank_hypotheses(self, hypotheses: List[Hypothesis], 
                        training_pairs: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Rank hypotheses by predicted effectiveness"""
        
        # Simple ranking: prefer hypotheses with fewer operations (Occam's razor)
        ranked = sorted(hypotheses, key=lambda h: (len(h.transformations), -h.confidence))
        
        # Boost confidence for certain patterns
        for hypothesis in ranked:
            if any('rotation' in t.rule_type for t in hypothesis.transformations):
                hypothesis.confidence += 0.1
            if any('reflection' in t.rule_type for t in hypothesis.transformations):
                hypothesis.confidence += 0.1
        
        return ranked
