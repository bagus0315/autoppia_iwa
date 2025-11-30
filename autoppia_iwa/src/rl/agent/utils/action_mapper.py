"""Improved action mapping between trajectories and agent action space."""

from __future__ import annotations

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ..runtime.action_adapter import BaseAction
    from ..runtime.browser_manager import Candidate


class ActionMapper:
    """Improved action mapper that handles mismatches between trajectory actions and agent actions."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        fallback_threshold: float = 0.3,
        position_tolerance: float = 50.0,
    ):
        """Initialize action mapper.
        
        Args:
            similarity_threshold: Primary similarity threshold for matching
            fallback_threshold: Fallback threshold for weaker matches
            position_tolerance: Position matching tolerance in pixels
        """
        self.similarity_threshold = similarity_threshold
        self.fallback_threshold = fallback_threshold
        self.position_tolerance = position_tolerance
    
    def map_action_to_topk(
        self,
        expert_action: "BaseAction",
        candidates: list["Candidate"],
        task_prompt: str = "",
    ) -> Optional[int]:
        """Map expert action to topK candidate index.
        
        Args:
            expert_action: Expert action from trajectory
            candidates: List of topK candidates from environment
            task_prompt: Task description (for context)
            
        Returns:
            Candidate index if match found, None otherwise
        """
        if not candidates:
            return None
        
        best_idx = None
        best_similarity = 0.0
        
        for idx, candidate in enumerate(candidates):
            similarity = self._compute_similarity(expert_action, candidate, task_prompt)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        # Use thresholds
        if best_similarity >= self.similarity_threshold:
            logger.debug(f"✅ Action mapped with similarity {best_similarity:.2f} (idx={best_idx})")
            return best_idx
        elif best_similarity >= self.fallback_threshold:
            logger.debug(f"⚠️ Fallback match with similarity {best_similarity:.2f} (idx={best_idx})")
            return best_idx
        else:
            logger.debug(f"❌ No match found (best similarity: {best_similarity:.2f})")
            return None
    
    def _compute_similarity(
        self,
        action: "BaseAction",
        candidate: "Candidate",
        task_prompt: str = "",
    ) -> float:
        """Compute similarity score between action and candidate.
        
        Args:
            action: Expert action
            candidate: Environment candidate
            task_prompt: Task description
            
        Returns:
            Similarity score [0, 1]
        """
        score = 0.0
        max_score = 0.0
        
        # Type matching
        action_type = type(action).__name__
        if "Click" in action_type:
            if candidate.clickable:
                score += 0.3
                max_score += 0.3
        elif "Type" in action_type or "Input" in action_type:
            if candidate.editable or candidate.focusable:
                score += 0.3
                max_score += 0.3
        elif "Submit" in action_type:
            if candidate.clickable and ("submit" in (candidate.role or "").lower() or "button" in candidate.tag.lower()):
                score += 0.3
                max_score += 0.3
        
        # Text matching
        action_text = getattr(action, "text", None) or getattr(action, "value", None) or ""
        if action_text and candidate.text:
            import re
            action_tokens = set(re.findall(r'\w+', action_text.lower()))
            candidate_tokens = set(re.findall(r'\w+', candidate.text.lower()))
            if action_tokens and candidate_tokens:
                overlap = len(action_tokens & candidate_tokens) / max(len(action_tokens), len(candidate_tokens))
                score += 0.3 * overlap
                max_score += 0.3
        
        # Position matching (for click actions)
        if hasattr(action, "x") and hasattr(action, "y") and candidate.bbox:
            action_x, action_y = float(action.x), float(action.y)
            cand_x, cand_y, cand_w, cand_h = candidate.bbox
            
            # Check if action point is within candidate bbox (with tolerance)
            if (cand_x - self.position_tolerance <= action_x <= cand_x + cand_w + self.position_tolerance and
                cand_y - self.position_tolerance <= action_y <= cand_y + cand_h + self.position_tolerance):
                score += 0.2
                max_score += 0.2
            else:
                # Distance-based penalty
                center_x, center_y = cand_x + cand_w / 2, cand_y + cand_h / 2
                distance = np.sqrt((action_x - center_x)**2 + (action_y - center_y)**2)
                if distance < self.position_tolerance * 2:
                    score += 0.1 * (1.0 - distance / (self.position_tolerance * 2))
                    max_score += 0.1
        
        # Role/tag matching
        if hasattr(action, "role") and action.role and candidate.role:
            if action.role.lower() == candidate.role.lower():
                score += 0.1
                max_score += 0.1
        
        # Normalize
        if max_score > 0:
            return score / max_score
        return 0.0


# Global mapper instance
_action_mapper: Optional[ActionMapper] = None


def get_action_mapper(**kwargs) -> ActionMapper:
    """Get global action mapper instance."""
    global _action_mapper
    if _action_mapper is None:
        _action_mapper = ActionMapper(**kwargs)
    return _action_mapper

