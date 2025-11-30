"""LLM-based reranker for topK element selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ..runtime.browser_manager import Candidate


class LLMReranker:
    """Reranker that uses LLM to select topK elements, then trains a small model."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize reranker.
        
        Args:
            model_path: Path to trained reranker model (if available)
            llm_provider: LLM provider ("openai", "anthropic", etc.)
            llm_model: LLM model name
        """
        self.model_path = model_path
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.reranker_model = None
        
        if model_path and Path(model_path).exists():
            self._load_model()
    
    def _load_model(self):
        """Load trained reranker model."""
        # TODO: Implement model loading (could be a simple MLP or transformer)
        logger.info(f"Loading reranker model from {self.model_path}")
        # Placeholder for model loading
        pass
    
    def _call_llm(self, task_prompt: str, candidates: List["Candidate"], k: int) -> List[int]:
        """Use LLM to select top K candidates.
        
        Args:
            task_prompt: Task description
            candidates: List of candidate elements
            k: Number of top candidates to select
            
        Returns:
            List of candidate indices selected by LLM
        """
        # Format candidates for LLM
        candidate_descriptions = []
        for i, cand in enumerate(candidates):
            desc = f"{i}. {cand.tag}"
            if cand.role:
                desc += f" [role={cand.role}]"
            if cand.text:
                desc += f" text='{cand.text[:50]}'"
            if cand.clickable:
                desc += " [clickable]"
            candidate_descriptions.append(desc)
        
        prompt = f"""Given this task: "{task_prompt}"

And these {len(candidates)} candidate elements:
{chr(10).join(candidate_descriptions)}

Select the top {k} most relevant elements for completing this task.
Return only a JSON array of indices, e.g., [0, 5, 12, ...]
"""
        
        try:
            if self.llm_provider == "openai":
                import openai
                response = openai.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                result_text = response.choices[0].message.content.strip()
            else:
                # Fallback or other providers
                logger.warning(f"LLM provider {self.llm_provider} not implemented. Using fallback.")
                return self._fallback_selection(candidates, k)
            
            # Parse JSON response
            try:
                selected_indices = json.loads(result_text)
                if isinstance(selected_indices, list):
                    # Validate indices
                    valid_indices = [idx for idx in selected_indices if 0 <= idx < len(candidates)]
                    if len(valid_indices) >= k:
                        return valid_indices[:k]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {result_text}")
        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Using fallback.")
        
        return self._fallback_selection(candidates, k)
    
    def _fallback_selection(self, candidates: List[Candidate], k: int) -> List[int]:
        """Fallback selection using simple heuristics."""
        # Simple scoring (similar to original topk)
        scores = []
        for i, cand in enumerate(candidates):
            score = 0.0
            if cand.clickable and cand.visible and cand.enabled:
                score += 1.0
            if cand.focusable:
                score += 0.3
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:k]]
    
    def rerank(
        self,
        task_prompt: str,
        candidates: List["Candidate"],
        k: int,
        use_llm: bool = False,
    ) -> Tuple[List["Candidate"], np.ndarray]:
        """Rerank candidates and return top K.
        
        Args:
            task_prompt: Task description
            candidates: List of candidate elements
            k: Number of top candidates to return
            use_llm: Whether to use LLM for selection (if False, uses trained model)
            
        Returns:
            Tuple of (top_k_candidates, click_mask)
        """
        if use_llm or self.reranker_model is None:
            # Use LLM directly
            selected_indices = self._call_llm(task_prompt, candidates, k)
            top_candidates = [candidates[i] for i in selected_indices if i < len(candidates)]
        else:
            # Use trained reranker model
            # TODO: Implement model inference
            top_candidates = candidates[:k]
        
        # Ensure we have exactly k candidates (pad if needed)
        from ..runtime.browser_manager import Candidate
        while len(top_candidates) < k:
            top_candidates.append(Candidate(
                idx=len(top_candidates),
                tag="",
                role=None,
                text="",
                clickable=False,
                focusable=False,
                editable=False,
                visible=False,
                enabled=False,
                bbox=None,
            ))
        
        top_candidates = top_candidates[:k]
        
        # Create click mask
        click_mask = np.array([c.clickable and c.visible and c.enabled for c in top_candidates], dtype=np.bool_)
        
        return top_candidates, click_mask


class RerankerTrainer:
    """Train a reranker model on LLM-selected data."""
    
    def __init__(self, output_path: str):
        """Initialize trainer.
        
        Args:
            output_path: Path to save trained model
        """
        self.output_path = Path(output_path)
        self.training_data = []
    
    def collect_llm_labels(
        self,
        task_prompt: str,
        candidates: List["Candidate"],
        k: int,
        reranker: LLMReranker,
    ):
        """Collect LLM labels for training.
        
        Args:
            task_prompt: Task description
            candidates: List of candidates
            k: Top K to select
            reranker: Reranker instance to use for LLM calls
        """
        selected_indices = reranker._call_llm(task_prompt, candidates, k)
        
        # Create training example
        example = {
            "task_prompt": task_prompt,
            "candidates": [
                {
                    "tag": c.tag,
                    "role": c.role,
                    "text": c.text,
                    "clickable": c.clickable,
                    "focusable": c.focusable,
                    "editable": c.editable,
                    "visible": c.visible,
                    "enabled": c.enabled,
                }
                for c in candidates
            ],
            "selected_indices": selected_indices,
            "k": k,
        }
        self.training_data.append(example)
    
    def train(self):
        """Train reranker model on collected data."""
        if not self.training_data:
            logger.warning("No training data collected. Cannot train model.")
            return
        
        logger.info(f"Training reranker on {len(self.training_data)} examples")
        # TODO: Implement model training (could use scikit-learn, PyTorch, etc.)
        # For now, just save the training data
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path / "training_data.jsonl", "w") as f:
            for example in self.training_data:
                f.write(json.dumps(example) + "\n")
        
        logger.info(f"Saved training data to {self.output_path / 'training_data.jsonl'}")
        logger.info("TODO: Implement actual model training")


# Global reranker instance
_reranker: Optional[LLMReranker] = None


def get_reranker(model_path: Optional[str] = None) -> LLMReranker:
    """Get global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = LLMReranker(model_path=model_path)
    return _reranker

