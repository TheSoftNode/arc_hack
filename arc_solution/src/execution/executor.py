"""
ARC Prize 2025 - Execution Engine

This module implements the execution engine that runs transformation programs
and validates their outputs against expected results.
"""

import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..core.types import Grid, Task, Hypothesis, TransformationProgram
from ..dsl.primitives import DSLPrimitives


@dataclass
class ExecutionResult:
    """Result of executing a transformation program"""
    success: bool
    output: Optional[Grid]
    error: Optional[str]
    execution_time: float
    metadata: Dict[str, Any]


class ExecutionEngine:
    """
    Executes transformation programs and validates their correctness.
    
    Features:
    - Safe execution with error handling
    - Timeout protection
    - Result validation
    - Performance monitoring
    - Debug information
    """
    
    def __init__(self, primitives: DSLPrimitives):
        self.primitives = primitives
        self.execution_timeout = 5.0  # seconds
        self.max_grid_size = 30
        self.debug_mode = False
    
    def execute_program(self, program: TransformationProgram, input_grid: Grid) -> ExecutionResult:
        """
        Execute a transformation program on an input grid
        
        Args:
            program: The transformation program to execute
            input_grid: The input grid to transform
            
        Returns:
            ExecutionResult with success status and output
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not self._validate_input_grid(input_grid):
                return ExecutionResult(
                    success=False,
                    output=None,
                    error="Invalid input grid",
                    execution_time=time.time() - start_time,
                    metadata={'validation_failed': True}
                )
            
            # Execute operations sequentially
            current_grid = input_grid
            executed_operations = []
            
            for i, operation in enumerate(program.operations):
                step_start = time.time()
                
                # Check timeout
                if time.time() - start_time > self.execution_timeout:
                    return ExecutionResult(
                        success=False,
                        output=None,
                        error=f"Execution timeout after {self.execution_timeout}s",
                        execution_time=time.time() - start_time,
                        metadata={
                            'timeout': True,
                            'completed_operations': executed_operations
                        }
                    )
                
                # Execute single operation
                result = self._execute_operation(operation, current_grid)
                
                if not result.success:
                    return ExecutionResult(
                        success=False,
                        output=None,
                        error=f"Operation {i} failed: {result.error}",
                        execution_time=time.time() - start_time,
                        metadata={
                            'failed_operation_index': i,
                            'failed_operation': operation,
                            'completed_operations': executed_operations
                        }
                    )
                
                current_grid = result.output
                executed_operations.append({
                    'operation': operation,
                    'execution_time': time.time() - step_start,
                    'output_shape': (len(current_grid), len(current_grid[0]) if current_grid else 0)
                })
                
                # Validate intermediate result
                if not self._validate_grid(current_grid):
                    return ExecutionResult(
                        success=False,
                        output=None,
                        error=f"Invalid grid after operation {i}",
                        execution_time=time.time() - start_time,
                        metadata={
                            'validation_failed_at_operation': i,
                            'completed_operations': executed_operations
                        }
                    )
            
            return ExecutionResult(
                success=True,
                output=current_grid,
                error=None,
                execution_time=time.time() - start_time,
                metadata={
                    'operations_count': len(program.operations),
                    'executed_operations': executed_operations
                }
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Execution exception: {str(e)}",
                execution_time=time.time() - start_time,
                metadata={
                    'exception_type': type(e).__name__,
                    'traceback': traceback.format_exc() if self.debug_mode else None
                }
            )
    
    def _execute_operation(self, operation: Dict[str, Any], input_grid: Grid) -> ExecutionResult:
        """Execute a single operation"""
        start_time = time.time()
        
        try:
            op_name = operation.get('operation')
            if not op_name:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error="Missing operation name",
                    execution_time=time.time() - start_time,
                    metadata={'operation': operation}
                )
            
            # Get primitive function
            primitive_func = self.primitives.get_primitive(op_name)
            if not primitive_func:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {op_name}",
                    execution_time=time.time() - start_time,
                    metadata={'operation': operation}
                )
            
            # Extract parameters
            params = {k: v for k, v in operation.items() if k != 'operation'}
            
            # Execute operation
            output_grid = primitive_func(input_grid, **params)
            
            if output_grid is None:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Operation {op_name} returned None",
                    execution_time=time.time() - start_time,
                    metadata={'operation': operation, 'parameters': params}
                )
            
            return ExecutionResult(
                success=True,
                output=output_grid,
                error=None,
                execution_time=time.time() - start_time,
                metadata={'operation': operation, 'parameters': params}
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Operation execution failed: {str(e)}",
                execution_time=time.time() - start_time,
                metadata={
                    'operation': operation,
                    'exception_type': type(e).__name__,
                    'traceback': traceback.format_exc() if self.debug_mode else None
                }
            )
    
    def validate_hypothesis(self, hypothesis: Hypothesis, task: Task) -> Dict[str, Any]:
        """
        Validate a hypothesis against all training examples
        
        Args:
            hypothesis: The hypothesis to validate
            task: The task with training examples
            
        Returns:
            Validation results with success rate and details
        """
        if not hypothesis.program:
            return {
                'success': False,
                'error': 'No program in hypothesis',
                'matches': 0,
                'total': len(task.train_pairs)
            }
        
        results = []
        matches = 0
        
        for i, pair in enumerate(task.train_pairs):
            input_grid = pair['input']
            expected_output = pair['output']
            
            # Execute program
            execution_result = self.execute_program(hypothesis.program, input_grid)
            
            # Check if output matches expected
            output_matches = False
            if execution_result.success and execution_result.output:
                output_matches = self._grids_equal(execution_result.output, expected_output)
                if output_matches:
                    matches += 1
            
            results.append({
                'example_index': i,
                'execution_success': execution_result.success,
                'output_matches': output_matches,
                'execution_time': execution_result.execution_time,
                'error': execution_result.error,
                'expected_shape': (len(expected_output), len(expected_output[0]) if expected_output else 0),
                'actual_shape': (
                    len(execution_result.output), 
                    len(execution_result.output[0]) if execution_result.output else 0
                ) if execution_result.output else None
            })
        
        success_rate = matches / len(task.train_pairs) if task.train_pairs else 0.0
        
        return {
            'success': success_rate == 1.0,
            'success_rate': success_rate,
            'matches': matches,
            'total': len(task.train_pairs),
            'results': results,
            'hypothesis_id': hypothesis.description[:50]  # Truncated for logging
        }
    
    def execute_on_test_inputs(self, hypothesis: Hypothesis, test_inputs: List[Grid]) -> List[Optional[Grid]]:
        """
        Execute hypothesis program on test inputs
        
        Args:
            hypothesis: The validated hypothesis
            test_inputs: List of test input grids
            
        Returns:
            List of output grids (None for failed executions)
        """
        outputs = []
        
        if not hypothesis.program:
            return [None] * len(test_inputs)
        
        for test_input in test_inputs:
            execution_result = self.execute_program(hypothesis.program, test_input)
            
            if execution_result.success:
                outputs.append(execution_result.output)
            else:
                outputs.append(None)
                if self.debug_mode:
                    print(f"Test execution failed: {execution_result.error}")
        
        return outputs
    
    def _validate_input_grid(self, grid: Grid) -> bool:
        """Validate input grid format and constraints"""
        try:
            if not grid or not isinstance(grid, list):
                return False
            
            if not grid[0] or not isinstance(grid[0], list):
                return False
            
            # Check dimensions
            height = len(grid)
            width = len(grid[0])
            
            if height > self.max_grid_size or width > self.max_grid_size:
                return False
            
            if height == 0 or width == 0:
                return False
            
            # Check consistent width
            for row in grid:
                if not isinstance(row, list) or len(row) != width:
                    return False
                
                # Check cell values
                for cell in row:
                    if not isinstance(cell, int) or cell < 0 or cell > 9:
                        return False
            
            return True
        
        except:
            return False
    
    def _validate_grid(self, grid: Grid) -> bool:
        """Validate grid during execution"""
        return self._validate_input_grid(grid)
    
    def _grids_equal(self, grid1: Grid, grid2: Grid) -> bool:
        """Check if two grids are exactly equal"""
        try:
            if len(grid1) != len(grid2):
                return False
            
            for row1, row2 in zip(grid1, grid2):
                if len(row1) != len(row2):
                    return False
                
                for cell1, cell2 in zip(row1, row2):
                    if cell1 != cell2:
                        return False
            
            return True
        
        except:
            return False
    
    def get_execution_stats(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Compute execution statistics"""
        if not results:
            return {}
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        stats = {
            'total_executions': len(results),
            'successful_executions': len(successful),
            'failed_executions': len(failed),
            'success_rate': len(successful) / len(results),
            'avg_execution_time': sum(r.execution_time for r in results) / len(results),
            'max_execution_time': max(r.execution_time for r in results),
            'min_execution_time': min(r.execution_time for r in results)
        }
        
        if successful:
            stats['avg_successful_time'] = sum(r.execution_time for r in successful) / len(successful)
        
        if failed:
            stats['avg_failed_time'] = sum(r.execution_time for r in failed) / len(failed)
            
            # Common failure reasons
            error_types = {}
            for result in failed:
                error = result.error or 'Unknown error'
                error_types[error] = error_types.get(error, 0) + 1
            
            stats['common_errors'] = sorted(
                error_types.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        
        return stats
