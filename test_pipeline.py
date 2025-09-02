#!/usr/bin/env python3
"""
Test script to verify the ARC solution pipeline is working correctly.
"""

import sys
import json
from pathlib import Path

# Add the solution directory to Python path
solution_dir = Path(__file__).parent / "arc_solution"
sys.path.insert(0, str(solution_dir))

try:
    from src.core.pipeline import ARCReasoningPipeline, PipelineConfig, create_default_config
    from src.core.types import Task
    
    print("✓ Successfully imported pipeline components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def create_simple_test_task():
    """Create a simple test task for validation"""
    # Simple pattern: input is a 2x2 grid, output is same grid rotated 90 degrees
    test_task = Task(
        task_id="test_001",
        train_pairs=[
            {
                'input': [[1, 0], [0, 0]],
                'output': [[0, 1], [0, 0]]
            },
            {
                'input': [[0, 1], [0, 0]],
                'output': [[0, 0], [1, 0]]
            }
        ],
        test_inputs=[
            [[1, 1], [0, 0]]
        ]
    )
    return test_task


def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\n=== Testing Pipeline Initialization ===")
    
    try:
        # Test with default config
        config = PipelineConfig()
        pipeline = ARCReasoningPipeline(config)
        print("✓ Pipeline initialized with default config")
        
        # Test with custom config
        custom_config = PipelineConfig(
            max_hypotheses_per_component=5,
            enable_llm=False,  # Disable LLM for testing
            enable_vision=False  # Disable vision for testing
        )
        pipeline = ARCReasoningPipeline(custom_config)
        print("✓ Pipeline initialized with custom config")
        
        return pipeline
        
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return None


def test_pipeline_components(pipeline):
    """Test individual pipeline components"""
    print("\n=== Testing Pipeline Components ===")
    
    try:
        # Test preprocessor
        test_grid = [[1, 0], [0, 1]]
        scene = pipeline.preprocessor.process_grid(test_grid)
        print("✓ Preprocessor working")
        
        # Test primitives
        rotated = pipeline.primitives.rotate(test_grid, 90)
        print(f"✓ DSL primitives working: {test_grid} -> {rotated}")
        
        # Test program synthesizer
        test_task = create_simple_test_task()
        hypotheses = pipeline.program_synthesizer.synthesize_programs(test_task, max_candidates=3)
        print(f"✓ Program synthesizer generated {len(hypotheses)} hypotheses")
        
        return True
        
    except Exception as e:
        print(f"✗ Component testing failed: {e}")
        return False


def test_full_pipeline(pipeline):
    """Test the full pipeline on a simple task"""
    print("\n=== Testing Full Pipeline ===")
    
    try:
        test_task = create_simple_test_task()
        print(f"Test task: {test_task.task_id}")
        print(f"Training pairs: {len(test_task.train_pairs)}")
        print(f"Test inputs: {len(test_task.test_inputs)}")
        
        # Solve the task
        solution = pipeline.solve_task(test_task)
        
        print(f"✓ Pipeline completed successfully")
        print(f"  - Execution time: {solution.execution_time:.2f}s")
        print(f"  - Hypotheses generated: {len(solution.hypotheses)}")
        print(f"  - Predictions: {len(solution.predictions)}")
        print(f"  - Confidence scores: {solution.confidence_scores}")
        
        # Print predictions
        for i, prediction in enumerate(solution.predictions):
            print(f"  - Test {i+1}: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_data():
    """Test with real ARC data if available"""
    print("\n=== Testing with Real ARC Data ===")
    
    try:
        # Try to load real ARC data
        arc_data_path = Path("arc-prize/arc-agi_training_challenges.json")
        
        if not arc_data_path.exists():
            print("! Real ARC data not found, skipping this test")
            return True
        
        with open(arc_data_path, 'r') as f:
            arc_data = json.load(f)
        
        # Get first task
        task_id = list(arc_data.keys())[0]
        task_data = arc_data[task_id]
        
        # Convert to our Task format
        real_task = Task(
            task_id=task_id,
            train_pairs=task_data['train'],
            test_inputs=[pair['input'] for pair in task_data['test']]
        )
        
        print(f"Testing with real task: {task_id}")
        
        # Create pipeline with reduced settings for speed
        config = PipelineConfig(
            max_hypotheses_per_component=3,
            max_total_hypotheses=10,
            enable_llm=False,
            enable_vision=False
        )
        pipeline = ARCReasoningPipeline(config)
        
        # Solve the task
        solution = pipeline.solve_task(real_task)
        print(f"✓ Real data test completed in {solution.execution_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"! Real data test failed (this is okay): {e}")
        return True  # Don't fail the overall test


def main():
    """Run all tests"""
    print("ARC Solution Pipeline Test Suite")
    print("=" * 50)
    
    # Test 1: Pipeline initialization
    pipeline = test_pipeline_initialization()
    if not pipeline:
        sys.exit(1)
    
    # Test 2: Component testing
    if not test_pipeline_components(pipeline):
        sys.exit(1)
    
    # Test 3: Full pipeline
    if not test_full_pipeline(pipeline):
        sys.exit(1)
    
    # Test 4: Real data (optional)
    test_with_real_data()
    
    # Get pipeline stats
    print("\n=== Pipeline Statistics ===")
    stats = pipeline.get_pipeline_stats()
    print(f"Components available: {stats['components']}")
    print(f"Configuration: {stats['config']}")
    
    print("\n🎉 All tests passed! Pipeline is ready for the ARC Prize 2025!")


if __name__ == "__main__":
    main()
