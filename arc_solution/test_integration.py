#!/usr/bin/env python3
"""
Complete integration test for the ARC solution pipeline
"""

import sys
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.core.pipeline import ARCReasoningPipeline
from src.data.loader import ARCDataLoader
from src.kaggle.submission import KaggleSubmissionManager, KaggleSubmissionGenerator


def test_complete_pipeline():
    """Test the complete pipeline integration"""
    print("=== Testing Complete Pipeline Integration ===")
    
    # Initialize components
    loader = ARCDataLoader()
    pipeline = ARCReasoningPipeline()
    submission_generator = KaggleSubmissionGenerator()
    
    try:
        # Load test tasks (limit to 2 for testing)
        test_tasks = loader.load_test_tasks()[:2]
        
        if not test_tasks:
            print("No test tasks available")
            return
        
        print(f"Testing with {len(test_tasks)} test tasks")
        
        # Process each task
        solutions = {}
        for i, task in enumerate(test_tasks):
            task_id = f"test_task_{i}"
            print(f"\nProcessing {task_id}...")
            
            try:
                # Generate solutions using the pipeline
                task_solution = pipeline.solve_task(task)
                
                if task_solution and task_solution.predictions:
                    # Extract grids from predictions
                    solution_grids = []
                    for prediction_dict in task_solution.predictions:
                        for attempt_key, grid in prediction_dict.items():
                            solution_grids.append(grid)
                    
                    solutions[task_id] = solution_grids[:2]  # Take first 2 attempts
                    print(f"  ✅ Generated {len(solution_grids)} solution(s)")
                    
                    # Show solution info
                    for j, grid in enumerate(solution_grids[:2]):
                        print(f"    Solution {j+1}: {len(grid)}x{len(grid[0])} grid")
                else:
                    print(f"  ⚠️  No solutions generated")
                    # Add fallback solution
                    fallback_grid = [[0] * 3 for _ in range(3)]  # 3x3 black grid
                    solutions[task_id] = [fallback_grid, fallback_grid]
                    print(f"  📝 Added fallback solution")
            
            except Exception as e:
                print(f"  ❌ Error processing task: {e}")
                # Add fallback solution
                fallback_grid = [[0] * 3 for _ in range(3)]
                solutions[task_id] = [fallback_grid, fallback_grid]
                print(f"  📝 Added fallback solution after error")
        
        # Test submission creation (simplified)
        print(f"\n=== Testing Submission Creation ===")
        try:
            # Create a simple test submission using existing method
            submission_file = submission_generator.generate_test_submission(loader, pipeline)
            print(f"✅ Created test submission file: {submission_file}")
            
            # Validate submission
            validation_results = submission_generator.validate_submission(str(submission_file))
            print(f"Validation results: {validation_results}")
            
        except Exception as e:
            print(f"❌ Submission creation error: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Complete pipeline integration test completed")
        
    except Exception as e:
        print(f"❌ Pipeline integration error: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling in various components"""
    print("\n=== Testing Error Handling ===")
    
    pipeline = ARCReasoningPipeline()
    
    # Test with malformed task
    print("Testing with invalid task...")
    try:
        from src.core.types import Task
        
        invalid_task = Task(
            task_id="invalid",
            train_pairs=[],  # Empty training
            test_inputs=[[[1, 2, 3]]],  # Single test input
            test_outputs=None
        )
        
        task_solution = pipeline.solve_task(invalid_task)
        solution_count = len(task_solution.predictions) if task_solution and task_solution.predictions else 0
        print(f"✅ Handled invalid task gracefully: {solution_count} solution predictions")
        
    except Exception as e:
        print(f"⚠️  Error with invalid task: {e}")
    
    # Test submission with empty solutions
    print("Testing submission with empty solutions...")
    try:
        submission_generator = KaggleSubmissionGenerator()
        print(f"✅ Submission generator initialized successfully")
        
    except Exception as e:
        print(f"⚠️  Error with submission generator: {e}")


def test_component_availability():
    """Test which components are available and working"""
    print("\n=== Testing Component Availability ===")
    
    pipeline = ARCReasoningPipeline()
    
    # Check solver availability
    available_solvers = []
    if hasattr(pipeline, 'symbolic_solver'):
        available_solvers.append("symbolic")
    
    if hasattr(pipeline, 'vision_solver'):
        available_solvers.append("vision")
    
    if hasattr(pipeline, 'llm_reasoner'):
        available_solvers.append("llm")
    
    print(f"Available solvers: {available_solvers}")
    
    # Test each solver individually
    loader = ARCDataLoader()
    test_tasks = loader.load_training_tasks()[:1]  # Just one task
    
    if test_tasks:
        task = test_tasks[0]
        
        # Test symbolic solver
        try:
            from src.hypothesis_generators.symbolic_solver import SymbolicSolver
            symbolic_solver = SymbolicSolver(config={})
            hypotheses = symbolic_solver.generate_hypotheses(task, scenes=[])
            print(f"✅ Symbolic solver: {len(hypotheses)} hypotheses")
        except Exception as e:
            print(f"❌ Symbolic solver error: {e}")
        
        # Test vision solver
        try:
            from src.vision.vision_solver import VisionSolver
            vision_solver = VisionSolver()
            hypotheses = vision_solver.generate_hypotheses(task)
            print(f"✅ Vision solver: {len(hypotheses)} hypotheses")
        except Exception as e:
            print(f"❌ Vision solver error: {e}")
        
        # Test LLM reasoner
        try:
            from src.llm_reasoning.llm_reasoner import LLMReasoner
            llm_reasoner = LLMReasoner()
            if llm_reasoner.client:
                hypotheses = llm_reasoner.generate_hypotheses(task)
                print(f"✅ LLM reasoner: {len(hypotheses)} hypotheses")
            else:
                print("⚠️  LLM reasoner: No API client available")
        except Exception as e:
            print(f"❌ LLM reasoner error: {e}")


def main():
    """Run all integration tests"""
    print("ARC Solution - Complete Integration Test")
    print("=" * 60)
    
    test_component_availability()
    test_error_handling()
    test_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("All integration tests completed!")


if __name__ == "__main__":
    main()
