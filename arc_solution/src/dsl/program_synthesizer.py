"""
ARC Prize 2025 - Program Synthesizer

This module implements program synthesis for ARC transformations using
symbolic reasoning, genetic programming, and neural search.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from ..core.types import Grid, Task, Hypothesis, TransformationProgram
from .primitives import DSLPrimitives


@dataclass
class ProgramCandidate:
    """Represents a candidate transformation program"""
    operations: List[Dict[str, Any]]
    fitness: float = 0.0
    execution_time: float = 0.0
    complexity: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgramSynthesizer:
    """
    Synthesizes transformation programs using multiple strategies:
    1. Genetic Programming: Evolution of operation sequences
    2. Template-based: Common ARC patterns
    3. Neural-guided: LLM suggestions
    4. Symbolic reasoning: Rule-based synthesis
    """
    
    def __init__(self, primitives: DSLPrimitives):
        self.primitives = primitives
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.max_program_length = 10
        
        # Templates for common ARC patterns
        self.templates = self._build_templates()
        
        # Population for genetic programming
        self.population: List[ProgramCandidate] = []
    
    def _build_templates(self) -> List[List[Dict[str, Any]]]:
        """Build library of common transformation templates"""
        return [
            # Simple transformations
            [{'operation': 'rotate', 'angle': 90}],
            [{'operation': 'reflect', 'axis': 'vertical'}],
            [{'operation': 'reflect', 'axis': 'horizontal'}],
            
            # Color transformations
            [{'operation': 'recolor', 'color_map': {1: 2}}],
            [{'operation': 'fill_region', 'target_color': 0, 'fill_color': 1}],
            
            # Composite transformations
            [
                {'operation': 'rotate', 'angle': 90},
                {'operation': 'reflect', 'axis': 'vertical'}
            ],
            [
                {'operation': 'dilate', 'kernel_size': 3},
                {'operation': 'recolor', 'color_map': {1: 2}}
            ],
            
            # Object-based transformations
            [{'operation': 'move_object', 'object_color': 1, 'dx': 1, 'dy': 0}],
            [{'operation': 'duplicate_object', 'object_color': 1, 'positions': [(1, 1)]}],
            
            # Pattern completion
            [{'operation': 'complete_pattern', 'pattern_size': (2, 2)}],
            [{'operation': 'extend_pattern', 'direction': 'right', 'steps': 1}],
        ]
    
    def synthesize_programs(self, task: Task, max_candidates: int = 20) -> List[Hypothesis]:
        """
        Synthesize transformation programs for the given task
        
        Returns top candidates as hypotheses ranked by fitness
        """
        all_candidates = []
        
        # Strategy 1: Template-based synthesis
        template_candidates = self._synthesize_from_templates(task)
        all_candidates.extend(template_candidates)
        
        # Strategy 2: Genetic programming
        gp_candidates = self._genetic_programming_synthesis(task)
        all_candidates.extend(gp_candidates)
        
        # Strategy 3: Greedy search
        greedy_candidates = self._greedy_synthesis(task)
        all_candidates.extend(greedy_candidates)
        
        # Strategy 4: Random exploration
        random_candidates = self._random_synthesis(task, num_candidates=10)
        all_candidates.extend(random_candidates)
        
        # Evaluate and rank all candidates
        evaluated_candidates = self._evaluate_candidates(all_candidates, task)
        
        # Convert top candidates to hypotheses
        hypotheses = []
        for candidate in evaluated_candidates[:max_candidates]:
            program = TransformationProgram(
                operations=candidate.operations,
                complexity=candidate.complexity
            )
            
            hypothesis = Hypothesis(
                program=program,
                confidence=candidate.fitness,
                explanation=self._generate_explanation(candidate),
                metadata={
                    'execution_time': candidate.execution_time,
                    'synthesis_strategy': candidate.metadata.get('strategy', 'unknown')
                }
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _synthesize_from_templates(self, task: Task) -> List[ProgramCandidate]:
        """Generate candidates from predefined templates"""
        candidates = []
        
        for template in self.templates:
            candidate = ProgramCandidate(
                operations=template.copy(),
                metadata={'strategy': 'template'}
            )
            candidates.append(candidate)
        
        # Generate variations of templates
        for template in self.templates:
            variations = self._generate_template_variations(template)
            for variation in variations:
                candidate = ProgramCandidate(
                    operations=variation,
                    metadata={'strategy': 'template_variation'}
                )
                candidates.append(candidate)
        
        return candidates
    
    def _generate_template_variations(self, template: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Generate variations of a template by modifying parameters"""
        variations = []
        
        for op_dict in template:
            op_name = op_dict['operation']
            
            if op_name == 'rotate':
                # Try different angles
                for angle in [90, 180, 270]:
                    if angle != op_dict.get('angle', 90):
                        variation = template.copy()
                        variation[template.index(op_dict)] = {'operation': 'rotate', 'angle': angle}
                        variations.append(variation)
            
            elif op_name == 'reflect':
                # Try different axes
                for axis in ['vertical', 'horizontal']:
                    if axis != op_dict.get('axis', 'vertical'):
                        variation = template.copy()
                        variation[template.index(op_dict)] = {'operation': 'reflect', 'axis': axis}
                        variations.append(variation)
            
            elif op_name == 'move_object':
                # Try different movements
                for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1)]:
                    if (dx, dy) != (op_dict.get('dx', 0), op_dict.get('dy', 0)):
                        variation = template.copy()
                        variation[template.index(op_dict)] = {
                            'operation': 'move_object',
                            'object_color': op_dict.get('object_color', 1),
                            'dx': dx, 'dy': dy
                        }
                        variations.append(variation)
        
        return variations[:10]  # Limit variations
    
    def _genetic_programming_synthesis(self, task: Task) -> List[ProgramCandidate]:
        """Use genetic programming to evolve transformation programs"""
        # Initialize population
        self.population = self._initialize_population()
        
        best_candidates = []
        
        for generation in range(self.max_generations):
            # Evaluate fitness
            self._evaluate_population(task)
            
            # Store best candidates
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best_candidates.extend(self.population[:5])
            
            # Early stopping if perfect solution found
            if self.population[0].fitness >= 1.0:
                break
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_count = int(0.1 * self.population_size)
            new_population.extend(self.population[:elite_count])
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self._tournament_selection()
                    parent2 = self._tournament_selection()
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._tournament_selection()
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
        
        # Return unique best candidates
        unique_candidates = []
        seen_operations = set()
        
        for candidate in sorted(best_candidates, key=lambda x: x.fitness, reverse=True):
            ops_str = str(candidate.operations)
            if ops_str not in seen_operations:
                seen_operations.add(ops_str)
                candidate.metadata['strategy'] = 'genetic_programming'
                unique_candidates.append(candidate)
        
        return unique_candidates[:20]
    
    def _initialize_population(self) -> List[ProgramCandidate]:
        """Initialize random population for genetic programming"""
        population = []
        
        for _ in range(self.population_size):
            program_length = random.randint(1, self.max_program_length)
            operations = []
            
            for _ in range(program_length):
                operation = self._generate_random_operation()
                operations.append(operation)
            
            candidate = ProgramCandidate(
                operations=operations,
                complexity=program_length
            )
            population.append(candidate)
        
        return population
    
    def _generate_random_operation(self) -> Dict[str, Any]:
        """Generate a random operation with parameters"""
        op_name = random.choice(self.primitives.list_primitives())
        operation = {'operation': op_name}
        
        # Add operation-specific parameters
        if op_name == 'rotate':
            operation['angle'] = random.choice([90, 180, 270])
        elif op_name == 'reflect':
            operation['axis'] = random.choice(['vertical', 'horizontal'])
        elif op_name == 'translate':
            operation['dx'] = random.randint(-3, 3)
            operation['dy'] = random.randint(-3, 3)
        elif op_name == 'scale':
            operation['factor'] = random.choice([0.5, 2.0, 3.0])
        elif op_name == 'recolor':
            # Generate random color mapping
            colors = list(range(10))
            old_color = random.choice(colors)
            new_color = random.choice([c for c in colors if c != old_color])
            operation['color_map'] = {old_color: new_color}
        elif op_name == 'move_object':
            operation['object_color'] = random.randint(1, 9)
            operation['dx'] = random.randint(-2, 2)
            operation['dy'] = random.randint(-2, 2)
        elif op_name in ['dilate', 'erode', 'opening', 'closing']:
            operation['kernel_size'] = random.choice([3, 5])
        
        return operation
    
    def _tournament_selection(self, tournament_size: int = 3) -> ProgramCandidate:
        """Select individual using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: ProgramCandidate, parent2: ProgramCandidate) -> ProgramCandidate:
        """Create child through crossover of two parents"""
        if len(parent1.operations) == 0 or len(parent2.operations) == 0:
            return parent1 if len(parent1.operations) > 0 else parent2
        
        # Single-point crossover
        point1 = random.randint(0, len(parent1.operations))
        point2 = random.randint(0, len(parent2.operations))
        
        child_operations = parent1.operations[:point1] + parent2.operations[point2:]
        
        # Ensure reasonable length
        if len(child_operations) > self.max_program_length:
            child_operations = child_operations[:self.max_program_length]
        
        return ProgramCandidate(
            operations=child_operations,
            complexity=len(child_operations)
        )
    
    def _mutate(self, candidate: ProgramCandidate) -> ProgramCandidate:
        """Mutate a candidate program"""
        operations = candidate.operations.copy()
        
        if len(operations) == 0:
            # Add random operation
            operations.append(self._generate_random_operation())
        else:
            mutation_type = random.choice(['modify', 'add', 'remove'])
            
            if mutation_type == 'modify' and operations:
                # Modify existing operation
                idx = random.randint(0, len(operations) - 1)
                operations[idx] = self._generate_random_operation()
            
            elif mutation_type == 'add' and len(operations) < self.max_program_length:
                # Add new operation
                position = random.randint(0, len(operations))
                operations.insert(position, self._generate_random_operation())
            
            elif mutation_type == 'remove' and len(operations) > 1:
                # Remove operation
                idx = random.randint(0, len(operations) - 1)
                operations.pop(idx)
        
        return ProgramCandidate(
            operations=operations,
            complexity=len(operations)
        )
    
    def _greedy_synthesis(self, task: Task) -> List[ProgramCandidate]:
        """Use greedy search to build transformation programs"""
        candidates = []
        
        # Start with empty program and greedily add operations
        for _ in range(5):  # Multiple greedy attempts
            current_operations = []
            best_fitness = 0.0
            
            for step in range(self.max_program_length):
                best_operation = None
                best_step_fitness = best_fitness
                
                # Try adding each primitive operation
                for op_name in self.primitives.list_primitives():
                    # Generate operation with default parameters
                    operation = self._generate_operation_with_defaults(op_name)
                    test_operations = current_operations + [operation]
                    
                    # Evaluate this extension
                    test_candidate = ProgramCandidate(operations=test_operations)
                    fitness = self._calculate_fitness(test_candidate, task)
                    
                    if fitness > best_step_fitness:
                        best_step_fitness = fitness
                        best_operation = operation
                
                if best_operation is not None:
                    current_operations.append(best_operation)
                    best_fitness = best_step_fitness
                else:
                    break  # No improvement found
            
            if current_operations:
                candidate = ProgramCandidate(
                    operations=current_operations,
                    fitness=best_fitness,
                    complexity=len(current_operations),
                    metadata={'strategy': 'greedy'}
                )
                candidates.append(candidate)
        
        return candidates
    
    def _generate_operation_with_defaults(self, op_name: str) -> Dict[str, Any]:
        """Generate operation with sensible default parameters"""
        operation = {'operation': op_name}
        
        if op_name == 'rotate':
            operation['angle'] = 90
        elif op_name == 'reflect':
            operation['axis'] = 'vertical'
        elif op_name == 'translate':
            operation['dx'] = 1
            operation['dy'] = 0
        elif op_name == 'recolor':
            operation['color_map'] = {1: 2}
        elif op_name == 'move_object':
            operation['object_color'] = 1
            operation['dx'] = 1
            operation['dy'] = 0
        
        return operation
    
    def _random_synthesis(self, task: Task, num_candidates: int = 10) -> List[ProgramCandidate]:
        """Generate random transformation programs"""
        candidates = []
        
        for _ in range(num_candidates):
            program_length = random.randint(1, min(5, self.max_program_length))
            operations = []
            
            for _ in range(program_length):
                operation = self._generate_random_operation()
                operations.append(operation)
            
            candidate = ProgramCandidate(
                operations=operations,
                complexity=program_length,
                metadata={'strategy': 'random'}
            )
            candidates.append(candidate)
        
        return candidates
    
    def _evaluate_candidates(self, candidates: List[ProgramCandidate], task: Task) -> List[ProgramCandidate]:
        """Evaluate fitness of all candidates"""
        for candidate in candidates:
            if candidate.fitness == 0.0:  # Not evaluated yet
                candidate.fitness = self._calculate_fitness(candidate, task)
        
        # Sort by fitness (descending)
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        return candidates
    
    def _evaluate_population(self, task: Task):
        """Evaluate fitness of entire population"""
        for candidate in self.population:
            candidate.fitness = self._calculate_fitness(candidate, task)
    
    def _calculate_fitness(self, candidate: ProgramCandidate, task: Task) -> float:
        """Calculate fitness of a candidate program on the task"""
        try:
            total_matches = 0
            total_examples = len(task.train)
            
            if total_examples == 0:
                return 0.0
            
            for example in task.train:
                # Execute program on input
                result = self._execute_program(candidate.operations, example.input)
                
                # Compare with expected output
                if result is not None and self._grids_match(result, example.output):
                    total_matches += 1
            
            fitness = total_matches / total_examples
            
            # Penalty for complexity
            complexity_penalty = 0.01 * candidate.complexity
            fitness = max(0.0, fitness - complexity_penalty)
            
            return fitness
        
        except Exception:
            return 0.0  # Failed execution
    
    def _execute_program(self, operations: List[Dict[str, Any]], input_grid: Grid) -> Optional[Grid]:
        """Execute a sequence of operations on input grid"""
        try:
            current_grid = input_grid
            
            for operation in operations:
                op_name = operation['operation']
                primitive_func = self.primitives.get_primitive(op_name)
                
                if primitive_func is None:
                    return None
                
                # Extract parameters
                params = {k: v for k, v in operation.items() if k != 'operation'}
                
                # Execute operation
                current_grid = primitive_func(current_grid, **params)
                
                if current_grid is None:
                    return None
            
            return current_grid
        
        except Exception:
            return None
    
    def _grids_match(self, grid1: Grid, grid2: Grid) -> bool:
        """Check if two grids are identical"""
        try:
            arr1 = np.array(grid1)
            arr2 = np.array(grid2)
            return arr1.shape == arr2.shape and np.array_equal(arr1, arr2)
        except:
            return False
    
    def _generate_explanation(self, candidate: ProgramCandidate) -> str:
        """Generate human-readable explanation of the program"""
        if not candidate.operations:
            return "Identity transformation (no operations)"
        
        explanations = []
        for operation in candidate.operations:
            op_name = operation['operation']
            
            if op_name == 'rotate':
                angle = operation.get('angle', 90)
                explanations.append(f"Rotate by {angle} degrees")
            elif op_name == 'reflect':
                axis = operation.get('axis', 'vertical')
                explanations.append(f"Reflect along {axis} axis")
            elif op_name == 'translate':
                dx = operation.get('dx', 0)
                dy = operation.get('dy', 0)
                explanations.append(f"Translate by ({dx}, {dy})")
            elif op_name == 'recolor':
                color_map = operation.get('color_map', {})
                explanations.append(f"Recolor: {color_map}")
            elif op_name == 'move_object':
                color = operation.get('object_color', 1)
                dx = operation.get('dx', 0)
                dy = operation.get('dy', 0)
                explanations.append(f"Move color {color} by ({dx}, {dy})")
            else:
                explanations.append(f"Apply {op_name}")
        
        return " → ".join(explanations)
