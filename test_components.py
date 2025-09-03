#!/usr/bin/env python3
"""
Simple component test for ARC solution
"""

import sys
from pathlib import Path

# Add arc_solution to path
arc_solution_path = Path(__file__).parent / 'arc_solution'
sys.path.insert(0, str(arc_solution_path))

def test_basic_imports():
    """Test that all components can be imported"""
    print("=== Testing Basic Imports ===")
    
    try:
        from src.data.loader import ARCDataLoader
        print("✅ Data loader imported")
    except Exception as e:
        print(f"❌ Data loader import error: {e}")
    
    try:
        from src.core.pipeline import ARCReasoningPipeline
        print("✅ Pipeline imported")
    except Exception as e:
        print(f"❌ Pipeline import error: {e}")
    
    try:
        from src.hypothesis_generators.symbolic_solver import SymbolicSolver
        print("✅ Symbolic solver imported")
    except Exception as e:
        print(f"❌ Symbolic solver import error: {e}")
    
    try:
        from src.vision.vision_solver import VisionSolver
        print("✅ Vision solver imported")
    except Exception as e:
        print(f"❌ Vision solver import error: {e}")
    
    try:
        from src.llm_reasoning.llm_reasoner import LLMReasoner
        print("✅ LLM reasoner imported")
    except Exception as e:
        print(f"❌ LLM reasoner import error: {e}")
    
    try:
        from src.kaggle.submission import KaggleSubmissionGenerator
        print("✅ Submission generator imported")
    except Exception as e:
        print(f"❌ Submission generator import error: {e}")


def test_data_loading():
    """Test data loading functionality"""
    print("\n=== Testing Data Loading ===")
    
    try:
        from src.data.loader import ARCDataLoader
        loader = ARCDataLoader()
        
        # Test loading training tasks
        train_tasks = loader.load_training_tasks()
        print(f"✅ Loaded {len(train_tasks)} training tasks")
        
        # Test loading test tasks
        test_tasks = loader.load_test_tasks()
        print(f"✅ Loaded {len(test_tasks)} test tasks")
        
        # Show sample task info
        if train_tasks:
            sample_task = train_tasks[0]
            print(f"Sample task - train pairs: {len(sample_task.train_pairs)}, test inputs: {len(sample_task.test_inputs)}")
            
            # Show grid dimensions
            if sample_task.train_pairs:
                input_grid = sample_task.train_pairs[0]['input']
                output_grid = sample_task.train_pairs[0]['output']
                print(f"Sample grid dimensions - input: {len(input_grid)}x{len(input_grid[0])}, output: {len(output_grid)}x{len(output_grid[0])}")
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        import traceback
        traceback.print_exc()


def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\n=== Testing Pipeline Initialization ===")
    
    try:
        from src.core.pipeline import ARCReasoningPipeline
        pipeline = ARCReasoningPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Check if solvers are available
        if hasattr(pipeline, 'symbolic_solver') and pipeline.symbolic_solver:
            print("✅ Symbolic solver available in pipeline")
        else:
            print("⚠️  Symbolic solver not available in pipeline")
            
        if hasattr(pipeline, 'vision_solver') and pipeline.vision_solver:
            print("✅ Vision solver available in pipeline")
        else:
            print("⚠️  Vision solver not available in pipeline")
            
        if hasattr(pipeline, 'llm_reasoner') and pipeline.llm_reasoner:
            print("✅ LLM reasoner available in pipeline")
        else:
            print("⚠️  LLM reasoner not available in pipeline")
        
    except Exception as e:
        print(f"❌ Pipeline initialization error: {e}")
        import traceback
        traceback.print_exc()


def test_vision_solver():
    """Test vision solver functionality"""
    print("\n=== Testing Vision Solver ===")
    
    try:
        from src.vision.vision_solver import VisionSolver
        from src.data.loader import ARCDataLoader
        
        vision_solver = VisionSolver()
        print("✅ Vision solver initialized")
        
        # Test with sample data
        loader = ARCDataLoader()
        train_tasks = loader.load_training_tasks()
        
        if train_tasks:
            sample_task = train_tasks[0]
            hypotheses = vision_solver.generate_hypotheses(sample_task, max_hypotheses=3)
            print(f"✅ Generated {len(hypotheses)} hypotheses")
            
            for i, hypothesis in enumerate(hypotheses):
                print(f"  Hypothesis {i+1}: {hypothesis.description} (confidence: {hypothesis.confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Vision solver error: {e}")


def test_submission_system():
    """Test submission system"""
    print("\n=== Testing Submission System ===")
    
    try:
        from src.kaggle.submission import KaggleSubmissionGenerator
        from src.data.loader import ARCDataLoader
        
        submission_gen = KaggleSubmissionGenerator()
        print("✅ Submission generator initialized")
        
        loader = ARCDataLoader()
        print("✅ Data loader for submission testing ready")
        
    except Exception as e:
        print(f"❌ Submission system error: {e}")


def main():
    """Run all basic tests"""
    print("ARC Solution - Basic Component Tests")
    print("=" * 50)
    
    test_basic_imports()
    test_data_loading()
    test_pipeline_initialization()
    test_vision_solver()
    test_submission_system()
    
    print("\n" + "=" * 50)
    print("Basic tests completed!")


if __name__ == "__main__":
    main()
