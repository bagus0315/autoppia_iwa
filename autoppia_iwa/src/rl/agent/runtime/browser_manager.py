from __future__ import annotations

"""
BrowserManager: obtiene estado del DOM desde Playwright Page y construye
Top‑K de candidatos + máscaras y features para la observación.

Para simplicidad y velocidad inicial:
- Candidatos = elementos con tag/button/a/input/textarea o role relevante o editable.
- Heurística de ranking: clicabilidad + role bonus + similitud con prompt + centro en viewport.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from ..utils.dom_cleaner import get_dom_cleaner
from ..utils.reranker import get_reranker


@dataclass
class Candidate:
    idx: int
    tag: str
    role: str | None
    text: str
    clickable: bool
    focusable: bool
    editable: bool
    visible: bool
    enabled: bool
    bbox: Tuple[float, float, float, float] | None

    def center(self) -> Tuple[float | None, float | None]:
        if not self.bbox:
            return None, None
        x, y, w, h = self.bbox
        try:
            return float(x + w / 2.0), float(y + h / 2.0)
        except Exception:
            return None, None


def _tokenize(s: str) -> list[str]:
    import re

    s = (s or "").lower().strip()
    return [t for t in re.split(r"[^\w]+", s) if t]


class BrowserManager:
    def __init__(self, page, use_reranker: bool = True, reranker_model_path: Optional[str] = None):
        self.page = page
        self.use_reranker = use_reranker
        self.reranker = get_reranker(model_path=reranker_model_path) if use_reranker else None
        self.dom_cleaner = get_dom_cleaner()

    async def snapshot_text(self, clean_dom: bool = True) -> Tuple[str, str]:
        """Get page snapshot with optional DOM cleaning.
        
        Args:
            clean_dom: Whether to clean DOM using BetterSoup
            
        Returns:
            Tuple of (html, url)
        """
        html = await self.page.content()
        url = self.page.url
        
        if clean_dom:
            html = self.dom_cleaner.clean(html)
        
        return html, url

    async def candidates(self) -> List[Candidate]:
        js = """
        () => {
          const nodes = [];
          const q = document.querySelectorAll('button, a, input, textarea, [role]');
          let i = 0;
          for (const el of q) {
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            const visible = !!(rect && rect.width > 1 && rect.height > 1) && style.visibility !== 'hidden' && style.display !== 'none';
            const enabled = !el.disabled;
            const tag = (el.tagName || '').toLowerCase();
            const role = (el.getAttribute('role') || '').toLowerCase();
            const type = (el.getAttribute('type') || '').toLowerCase();
            const text = (el.innerText || el.value || '').trim();
            const clickable = ['button','a'].includes(tag) || role === 'button' || type === 'submit' || type === 'search';
            const focusable = (el.tabIndex >= 0) || ['input','textarea','select'].includes(tag) || el.isContentEditable === true;
            const editable = ['input','textarea'].includes(tag) || el.isContentEditable === true;
            nodes.push({
              idx: i++, tag, role, text, clickable, focusable, editable,
              visible, enabled,
              bbox: rect ? [rect.x, rect.y, rect.width, rect.height] : null,
            });
          }
          return nodes;
        }
        """
        raw = await self.page.evaluate(js)
        out: List[Candidate] = []
        for d in raw or []:
            out.append(
                Candidate(
                    idx=int(d.get("idx", 0)),
                    tag=str(d.get("tag", "")),
                    role=str(d.get("role")) if d.get("role") else None,
                    text=str(d.get("text", "")),
                    clickable=bool(d.get("clickable", False)),
                    focusable=bool(d.get("focusable", False)),
                    editable=bool(d.get("editable", False)),
                    visible=bool(d.get("visible", True)),
                    enabled=bool(d.get("enabled", True)),
                    bbox=tuple(d.get("bbox")) if d.get("bbox") else None,
                )
            )
        return out

    async def topk(self, task_prompt: str, K: int, use_llm: bool = False) -> Tuple[List[Candidate], np.ndarray, dict]:
        """Get top K candidates using reranker or fallback to hardcoded method.
        
        Args:
            task_prompt: Task description
            K: Number of top candidates to return
            use_llm: Whether to use LLM for reranking (if reranker available)
            
        Returns:
            Tuple of (top_candidates, click_mask, macros)
        """
        cands = await self.candidates()
        
        # Use reranker if available and enabled
        if self.use_reranker and self.reranker:
            try:
                top, click_mask = self.reranker.rerank(task_prompt, cands, K, use_llm=use_llm)
                logger.debug(f"Reranker selected {len(top)} candidates")
            except Exception as e:
                logger.warning(f"Reranker failed: {e}. Falling back to hardcoded method.")
                top, click_mask = self._hardcoded_topk(task_prompt, cands, K)
        else:
            # Fallback to hardcoded method
            top, click_mask = self._hardcoded_topk(task_prompt, cands, K)
        
        # Ensure we have exactly K candidates
        while len(top) < K:
            top.append(Candidate(
                idx=len(top),
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
        top = top[:K]
        
        # Update click mask to match K
        if len(click_mask) < K:
            click_mask = np.pad(click_mask, (0, K - len(click_mask)), constant_values=False)
        click_mask = click_mask[:K]
        
        # macros (unchanged)
        macros = {
            "type_confirm": any(c.focusable and c.visible and c.enabled for c in cands),
            "submit": any(
                (c.clickable and c.visible and c.enabled and ((c.role or "").lower() in {"button", "submit"} or any(k in (c.text or "").lower() for k in ("submit", "search", "go"))))
                for c in cands
            ),
            "scroll_down": True,
            "scroll_up": True,
            "back": True,
        }
        
        return top, click_mask, macros
    
    def _hardcoded_topk(self, task_prompt: str, cands: List[Candidate], K: int) -> Tuple[List[Candidate], np.ndarray]:
        """Fallback hardcoded topK selection (original method)."""
        toks_goal = set(_tokenize(task_prompt))

        def score(c: Candidate) -> float:
            if not (c.visible and c.enabled):
                return 0.0
            s = 0.0
            if c.clickable:
                s += 1.0
            if c.focusable:
                s += 0.3
            if c.editable:
                s += 0.2
            role_bonus = {"button": 1.0, "link": 0.9, "submit": 0.95, "textbox": 0.8}.get((c.role or "").lower())
            if role_bonus:
                s += role_bonus
            # similitud simple por solapamiento
            toks = _tokenize(c.text)
            if toks and toks_goal:
                ov = sum(1 for t in toks if t in toks_goal)
                s += ov / max(1.0, np.sqrt(len(toks_goal) * len(toks)))
            # ligera preferencia por centro
            cx, cy = c.center()
            if cx is not None and cy is not None:
                # normalizamos suponiendo 1920x1080
                cxn, cyn = min(1.0, max(0.0, cx / 1920.0)), min(1.0, max(0.0, cy / 1080.0))
                s += 0.05 * (1.0 - abs(0.5 - cxn)) + 0.05 * (1.0 - abs(0.5 - cyn))
            return float(s)

        scored = [(c, score(c)) for c in cands]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [c for c, s in scored[:K] if s > 0.0]

        # máscara CLICK_K
        click_mask = np.zeros((K,), dtype=np.bool_)
        for i, c in enumerate(top):
            if i < K:
                click_mask[i] = bool(c.clickable and c.visible and c.enabled)

        return top, click_mask

