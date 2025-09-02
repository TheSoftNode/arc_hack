#!/usr/bin/env python3
"""
ARC Prize 2025 - Quick Test Script

Test the basic functionality of our solution pipeline.
"""

import sys
import json
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.types import Task
from core.pipeline import ARCReasoningPipeline, PipelineConfig


def load_sample_task():
    """Load a sample ARC task for testing"""
    # Simple test task - identity transformation
    task_data = {
        "train": [
            {
                "input": [[1, 0], [0, 1]],
                "output": [[1, 0], [0, 1]]
            },
            {
                "input": [[2, 1], [1, 2]],
                "output": [[2, 1], [1, 2]]
            }
        ],
        "test": [
            {
                "input": [[3, 0], [0, 3]]
            }
        ]
    }
    
    # Convert to our Task format
    task = Task(
        task_id="test_001",
        train_pairs=[
            {"input": pair["input"], "output": pair["output"]}
            for pair in task_data["train"]
        ],
        test_inputs=[test["input"] for test in task_data["test"]]
    )
    
    return task


def test_pipeline():
    """Test the basic pipeline functionality"""
    print("🚀 Testing ARC Solution Pipeline")
    print("=" * 50)
    
    # Create pipeline with minimal configuration
    config = PipelineConfig(
        max_hypotheses_per_component=5,
        max_total_hypotheses=10,
        enable_llm=False,  # Disable LLM for quick test
        enable_vision=False,  # Disable vision for quick test
        enable_symbolic=True
    )
    
    pipeline = ARCReasoningPipeline(config)
    
    # Load test task
    task = load_sample_task()
    print(f"📝 Loaded test task: {task.task_id}")
    print(f"   Training examples: {len(task.train_pairs)}")
    print(f"   Test inputs: {len(task.test_inputs)}")
    
    # Test pipeline components
    print("\n🔧 Testing pipeline components...")
    stats = pipeline.get_pipeline_stats()
    print(f"   Components loaded: {stats['components']}")
    
    # Solve the task
    print("\n🧠 Solving task...")
    try:
        solution = pipeline.solve_task(task)
        
        print(f"✅ Task solved successfully!")
        print(f"   Execution time: {solution.execution_time:.2f}s")
        print(f"   Hypotheses generated: {len(solution.hypotheses)}")
        print(f"   Predictions generated: {len(solution.predictions)}")
        
        if solution.predictions:
            print(f"   First prediction: {solution.predictions[0]}")
        
        if 'error' in solution.metadata:
            print(f"⚠️  Error: {solution.metadata['error']}")
        
    except Exception as e:
        print(f"❌ Error solving task: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Test completed!")


if __name__ == "__main__":
    test_pipeline()
