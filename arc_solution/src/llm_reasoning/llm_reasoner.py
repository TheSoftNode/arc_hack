"""
ARC Prize 2025 - LLM Reasoning Engine

This module implements the Large Language Model reasoning component that
provides high-level reasoning about transformations and guides the search.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import groq
except ImportError:
    groq = None

from ..core.types import Grid, Task, Hypothesis, TransformationProgram


@dataclass
class LLMResponse:
    """Response from LLM with structured output"""
    reasoning: str
    transformations: List[Dict[str, Any]]
    confidence: float
    explanation: str
    metadata: Dict[str, Any]


class LLMReasoner:
    """
    Large Language Model reasoning engine for ARC tasks.
    
    Supports multiple LLM providers:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Groq (Mixtral, Llama)
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.provider = provider.lower()
        self.model = model
        self.client = None
        self.max_tokens = 2000
        self.temperature = 0.1  # Low temperature for consistent reasoning
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == "openai" and openai:
            self.client = openai.OpenAI()
        elif self.provider == "anthropic" and anthropic:
            self.client = anthropic.Anthropic()
        elif self.provider == "groq" and groq:
            self.client = groq.Groq()
        else:
            print(f"Warning: LLM provider {self.provider} not available or not installed")
    
    def generate_hypotheses(self, task: Task, max_hypotheses: int = 5) -> List[Hypothesis]:
        """
        Generate hypotheses for solving the ARC task using LLM reasoning
        
        Args:
            task: The ARC task to solve
            max_hypotheses: Maximum number of hypotheses to generate
            
        Returns:
            List of hypotheses with LLM reasoning
        """
        if not self.client:
            return []
        
        try:
            # Prepare task context for LLM
            task_context = self._format_task_for_llm(task)
            
            # Generate reasoning and transformations
            llm_response = self._query_llm_for_reasoning(task_context)
            
            if not llm_response:
                return []
            
            # Convert LLM response to hypotheses
            hypotheses = self._convert_to_hypotheses(llm_response, task)
            
            return hypotheses[:max_hypotheses]
        
        except Exception as e:
            print(f"Error in LLM reasoning: {e}")
            return []
    
    def _format_task_for_llm(self, task: Task) -> str:
        """Format ARC task for LLM consumption"""
        context = "# ARC Task Analysis\n\n"
        context += "You are an expert at solving Abstract Reasoning Corpus (ARC) tasks. "
        context += "These are visual reasoning puzzles where you need to identify the underlying rule "
        context += "from training examples and apply it to test inputs.\n\n"
        
        context += "## Training Examples:\n"
        for i, pair in enumerate(task.train_pairs):
            context += f"### Example {i+1}:\n"
            context += "Input:\n"
            context += self._grid_to_string(pair['input'])
            context += "\nOutput:\n"
            context += self._grid_to_string(pair['output'])
            context += "\n"
        
        context += "## Test Input:\n"
        if task.test_inputs:
            context += self._grid_to_string(task.test_inputs[0])
        
        context += "\n## Task:\n"
        context += "1. Analyze the training examples to identify the transformation rule\n"
        context += "2. Describe your reasoning step by step\n"
        context += "3. Provide a list of primitive operations that could implement this transformation\n"
        context += "4. Estimate your confidence in this solution (0.0 to 1.0)\n"
        context += "5. Provide a clear explanation of the transformation\n\n"
        
        context += "Available primitive operations include:\n"
        context += "- Geometric: rotate, reflect, translate, scale\n"
        context += "- Color: recolor, fill_region, paint_pattern\n"
        context += "- Logical: grid_and, grid_or, grid_xor\n"
        context += "- Morphological: dilate, erode, opening, closing\n"
        context += "- Object: move_object, duplicate_object, remove_object\n"
        context += "- Pattern: complete_pattern, extend_pattern\n"
        context += "- Utility: identity, resize, crop, pad\n\n"
        
        context += "Please respond in the following JSON format:\n"
        context += "{\n"
        context += '  "reasoning": "Your step-by-step analysis...",\n'
        context += '  "transformations": [\n'
        context += '    {"operation": "operation_name", "parameters": {...}},\n'
        context += '    ...\n'
        context += '  ],\n'
        context += '  "confidence": 0.85,\n'
        context += '  "explanation": "Clear description of the transformation rule"\n'
        context += "}\n"
        
        return context
    
    def _grid_to_string(self, grid: Grid) -> str:
        """Convert grid to readable string representation"""
        result = ""
        for row in grid:
            result += " ".join(str(cell) for cell in row) + "\n"
        return result
    
    def _query_llm_for_reasoning(self, context: str) -> Optional[LLMResponse]:
        """Query the LLM with the formatted context"""
        try:
            if self.provider == "openai":
                return self._query_openai(context)
            elif self.provider == "anthropic":
                return self._query_anthropic(context)
            elif self.provider == "groq":
                return self._query_groq(context)
            else:
                return None
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None
    
    def _query_openai(self, context: str) -> Optional[LLMResponse]:
        """Query OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at solving ARC tasks. Always respond with valid JSON."},
                    {"role": "user", "content": context}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            return self._parse_llm_response(content)
        
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
    
    def _query_anthropic(self, context: str) -> Optional[LLMResponse]:
        """Query Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model if "claude" in self.model else "claude-3-sonnet-20240229",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": context}
                ]
            )
            
            content = response.content[0].text
            return self._parse_llm_response(content)
        
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None
    
    def _query_groq(self, context: str) -> Optional[LLMResponse]:
        """Query Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model if "mixtral" in self.model or "llama" in self.model else "mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert at solving ARC tasks. Always respond with valid JSON."},
                    {"role": "user", "content": context}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            return self._parse_llm_response(content)
        
        except Exception as e:
            print(f"Groq API error: {e}")
            return None
    
    def _parse_llm_response(self, content: str) -> Optional[LLMResponse]:
        """Parse LLM response content into structured format"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = content[json_start:json_end]
            response_data = json.loads(json_str)
            
            return LLMResponse(
                reasoning=response_data.get('reasoning', ''),
                transformations=response_data.get('transformations', []),
                confidence=response_data.get('confidence', 0.5),
                explanation=response_data.get('explanation', ''),
                metadata={'raw_response': content}
            )
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM JSON response: {e}")
            # Try to extract reasoning even if JSON is malformed
            return LLMResponse(
                reasoning=content[:500],  # First 500 chars as reasoning
                transformations=[],
                confidence=0.3,
                explanation="Failed to parse structured response",
                metadata={'raw_response': content, 'parse_error': str(e)}
            )
        
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None
    
    def _convert_to_hypotheses(self, llm_response: LLMResponse, task: Task) -> List[Hypothesis]:
        """Convert LLM response to hypothesis objects"""
        hypotheses = []
        
        if not llm_response.transformations:
            # Create hypothesis with reasoning even if no operations were parsed
            hypothesis = Hypothesis(
                transformations=[],  # Empty transformations list
                confidence=llm_response.confidence,
                description=llm_response.explanation,
                generated_by="llm_reasoner",
                reasoning=llm_response.reasoning,
                explanation=llm_response.explanation,
                metadata={
                    'llm_provider': self.provider,
                    'llm_model': self.model,
                    **llm_response.metadata
                }
            )
            hypotheses.append(hypothesis)
            return hypotheses
        
        # Create transformation program from LLM operations
        program = TransformationProgram(
            operations=llm_response.transformations,
            metadata={'source': 'llm_reasoning'}
        )
        
        # Create main hypothesis
        hypothesis = Hypothesis(
            transformations=[],  # Will be populated by execution engine
            confidence=llm_response.confidence,
            description=llm_response.explanation,
            generated_by="llm_reasoner",
            reasoning=llm_response.reasoning,
            program=program,
            explanation=llm_response.explanation,
            metadata={
                'llm_provider': self.provider,
                'llm_model': self.model,
                **llm_response.metadata
            }
        )
        hypotheses.append(hypothesis)
        
        # Generate variations with different confidence levels
        if len(llm_response.transformations) > 1:
            # Create simplified version with fewer operations
            simplified_ops = llm_response.transformations[:len(llm_response.transformations)//2]
            simplified_program = TransformationProgram(
                operations=simplified_ops,
                metadata={'source': 'llm_reasoning_simplified'}
            )
            
            simplified_hypothesis = Hypothesis(
                transformations=[],
                confidence=llm_response.confidence * 0.8,  # Lower confidence for simplified
                description=f"Simplified: {llm_response.explanation}",
                generated_by="llm_reasoner",
                reasoning=f"Simplified version: {llm_response.reasoning}",
                program=simplified_program,
                explanation=f"Simplified: {llm_response.explanation}",
                metadata={
                    'llm_provider': self.provider,
                    'llm_model': self.model,
                    'variation': 'simplified',
                    **llm_response.metadata
                }
            )
            hypotheses.append(simplified_hypothesis)
        
        return hypotheses
    
    def refine_hypothesis(self, hypothesis: Hypothesis, task: Task, execution_results: Dict[str, Any]) -> Optional[Hypothesis]:
        """
        Refine a hypothesis based on execution results and LLM feedback
        
        Args:
            hypothesis: The hypothesis to refine
            task: The original task
            execution_results: Results from attempting to execute the hypothesis
            
        Returns:
            Refined hypothesis or None if no improvement possible
        """
        if not self.client:
            return None
        
        try:
            # Prepare refinement context
            refinement_context = self._format_refinement_context(hypothesis, task, execution_results)
            
            # Query LLM for refinement suggestions
            llm_response = self._query_llm_for_reasoning(refinement_context)
            
            if not llm_response:
                return None
            
            # Create refined hypothesis
            refined_hypotheses = self._convert_to_hypotheses(llm_response, task)
            
            if refined_hypotheses:
                refined = refined_hypotheses[0]
                refined.metadata['refinement_of'] = hypothesis.description
                refined.metadata['original_confidence'] = hypothesis.confidence
                return refined
            
            return None
        
        except Exception as e:
            print(f"Error refining hypothesis: {e}")
            return None
    
    def _format_refinement_context(self, hypothesis: Hypothesis, task: Task, execution_results: Dict[str, Any]) -> str:
        """Format context for hypothesis refinement"""
        context = "# ARC Task Hypothesis Refinement\n\n"
        context += "A previous hypothesis failed to solve the task correctly. "
        context += "Please analyze the failure and suggest improvements.\n\n"
        
        context += "## Original Task:\n"
        context += self._format_task_for_llm(task)
        
        context += "\n## Previous Hypothesis:\n"
        context += f"Reasoning: {hypothesis.reasoning}\n"
        context += f"Explanation: {hypothesis.explanation}\n"
        if hypothesis.program:
            context += f"Operations: {hypothesis.program.operations}\n"
        
        context += "\n## Execution Results:\n"
        context += f"Success: {execution_results.get('success', False)}\n"
        context += f"Error: {execution_results.get('error', 'No error')}\n"
        context += f"Output matches: {execution_results.get('output_matches', [])}\n"
        
        context += "\n## Task:\n"
        context += "Based on the failure, suggest a refined approach. "
        context += "Focus on what went wrong and how to fix it.\n"
        
        return context
