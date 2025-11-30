from __future__ import annotations

import enum
import hashlib
from collections import deque
from typing import Deque, Mapping, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

from ..runtime.action_adapter import ActionAdapter, ActionLayout
from ..runtime.browser_manager import BrowserManager
from ..runtime.stateful_evaluator import PartialScore, StatefulEvaluator
from ..utils.tokenizer import get_goal_tokenizer, get_dom_tokenizer, get_url_tokenizer
from ..utils.dom_cleaner import get_dom_cleaner
from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.data_generation.tasks.classes import Task
from .dataset_task_loader import DatasetTaskLoader


class MacroAction(enum.IntEnum):
    TYPE_CONFIRM = 0
    SUBMIT = 1
    SCROLL_DOWN = 2
    SCROLL_UP = 3
    BACK = 4


# Legacy hash-based tokenization (kept for backward compatibility)
def _hash_token(token: str, vocab: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % vocab


def _tokenize(text: str) -> list[str]:
    import re

    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return [token for token in re.split(r"[^\w]+", normalized) if token]


class IWAWebEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[Mapping[str, object]] = None) -> None:
        super().__init__()
        self.cfg = dict(cfg or {})
        self.K = int(self.cfg.get("topk", 12))
        self.max_steps = int(self.cfg.get("max_steps", 30))
        self.goal_vocab = int(self.cfg.get("goal_vocab_size", 4096))
        self.dom_vocab = int(self.cfg.get("dom_vocab_size", 8192))
        self.url_vocab = int(self.cfg.get("url_vocab_size", 1024))
        self.max_goal_tokens = int(self.cfg.get("max_goal_tokens", 48))
        self.max_dom_tokens = int(self.cfg.get("max_dom_tokens", 200))
        self.max_element_tokens = int(self.cfg.get("max_element_tokens", 12))
        self.history_len = int(self.cfg.get("action_history", 10))
        # Start sampling projects from this index (exclude first N)
        self.project_start_index = max(0, int(self.cfg.get("project_start_index", 0)))
        # Tasks cache directory (align with benchmark)
        self.tasks_cache_dir = str(self.cfg.get("tasks_cache_dir", "data/cache/tasks"))
        # Whether to use cached tasks or force regeneration
        self.use_cached_tasks = bool(self.cfg.get("use_cached_tasks", False))

        # Support variable K (don't always use same K)
        self.use_variable_k = bool(self.cfg.get("use_variable_k", False))
        self.k_min = int(self.cfg.get("k_min", self.K))
        self.k_max = int(self.cfg.get("k_max", self.K))
        
        self.layout = ActionLayout(topk=self.K)
        self.action_adapter = ActionAdapter(self.layout)

        self.action_space = spaces.Discrete(1 + self.K + len(self.layout.macros))
        
        # Initialize tokenizers
        self.goal_tokenizer = get_goal_tokenizer(vocab_size=self.goal_vocab)
        self.dom_tokenizer = get_dom_tokenizer(vocab_size=self.dom_vocab)
        self.url_tokenizer = get_url_tokenizer(vocab_size=self.url_vocab)
        self.dom_cleaner = get_dom_cleaner()
        self.observation_space = spaces.Dict(
            {
                "goal_ids": spaces.Box(low=0, high=self.goal_vocab, shape=(self.max_goal_tokens,), dtype=np.int32),
                "dom_ids": spaces.Box(low=0, high=self.dom_vocab, shape=(self.max_dom_tokens,), dtype=np.int32),
                "url_id": spaces.Box(low=0, high=self.url_vocab, shape=(1,), dtype=np.int32),
                "prev_actions": spaces.Box(low=0, high=self.action_space.n, shape=(self.history_len,), dtype=np.int32),
                "topk_text_ids": spaces.Box(low=0, high=self.dom_vocab, shape=(self.K, self.max_element_tokens), dtype=np.int32),
                "topk_meta": spaces.Box(low=0.0, high=1.0, shape=(self.K, 8), dtype=np.float32),
                "score": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        # Runtime
        self._task: Optional[Task] = None
        self._evaluator: Optional[StatefulEvaluator] = None
        self._browser: Optional[BrowserManager] = None
        self._cands = []
        self._click_mask = np.zeros((self.K,), dtype=np.bool_)
        self._macros = {k: False for k in ("type_confirm", "submit", "scroll_down", "scroll_up", "back")}
        self._history: Deque[int] = deque(maxlen=self.history_len)
        self._step = 0
        self._last_partial = PartialScore()
        self._reward_blender = None
        reward_model_path = self.cfg.get("reward_model_path")
        if reward_model_path:
            try:
                from autoppia_iwa.src.rl.score_model.rl.reward_wrapper import RewardBlender

                alpha = float(self.cfg.get("reward_alpha", 0.5))
                beta = float(self.cfg.get("reward_beta", 0.5))
                gamma = float(self.cfg.get("reward_gamma", 0.995))
                self._reward_blender = RewardBlender(reward_model_path, alpha=alpha, beta=beta, gamma=gamma)
                logger.info(f"Loaded RewardBlender from {reward_model_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize RewardBlender ({e}). Continuing without shaped reward.")
                self._reward_blender = None
        
        # Dataset task loader for diverse PPO training
        self._dataset_loader = None
        dataset_path = self.cfg.get("dataset_path")
        if dataset_path:
            try:
                project_filter = self.cfg.get("project_name")  # e.g., "autoppia_cinema"
                self._dataset_loader = DatasetTaskLoader(dataset_path, project_filter=project_filter)
                logger.info(f"Loaded {self._dataset_loader.get_task_count()} tasks from {dataset_path}")
            except Exception as e:
                logger.warning(f"Failed to load dataset tasks ({e}). Will use cached tasks.")
                self._dataset_loader = None

    # -------------------------
    # Helpers
    # -------------------------
    def _encode_tokens(self, tokens: list[str], limit: int, vocab: int) -> np.ndarray:
        """Legacy hash-based encoding (kept for backward compatibility)."""
        arr = np.zeros((limit,), dtype=np.int32)
        for i, t in enumerate(tokens[:limit]):
            arr[i] = _hash_token(t, vocab)
        return arr

    def _encode_text(self, text: str, limit: int, vocab: int, use_tokenizer: bool = True) -> np.ndarray:
        """Encode text using tokenizer (preferred) or hash fallback."""
        if use_tokenizer:
            # Use proper tokenizer
            if vocab == self.goal_vocab:
                token_ids = self.goal_tokenizer.encode(text, max_length=limit)
            elif vocab == self.dom_vocab:
                token_ids = self.dom_tokenizer.encode(text, max_length=limit)
            elif vocab == self.url_vocab:
                token_ids = self.url_tokenizer.encode(text, max_length=limit)
            else:
                # Fallback to hash
                token_ids = self._encode_tokens(_tokenize(text), limit, vocab)
            
            arr = np.zeros((limit,), dtype=np.int32)
            arr[:len(token_ids)] = token_ids[:limit]
            return arr
        else:
            # Legacy hash-based encoding
            return self._encode_tokens(_tokenize(text), limit, vocab)

    def _obs(self, html: str, url: str, score: float) -> dict:
        goal = self._task.prompt if self._task else ""
        # topk text and meta
        use_tokenizer = bool(self.cfg.get("use_tokenizer", True))
        topk_text_ids = np.zeros((self.K, self.max_element_tokens), dtype=np.int32)
        topk_meta = np.zeros((self.K, 8), dtype=np.float32)
        for i, c in enumerate(self._cands[: self.K]):
            topk_text_ids[i] = self._encode_text(c.text, self.max_element_tokens, self.dom_vocab, use_tokenizer=use_tokenizer)
            cx, cy = c.center()
            cxn = min(1.0, max(0.0, (cx or 0.0) / 1920.0))
            cyn = min(1.0, max(0.0, (cy or 0.0) / 1080.0))
            role_id = {
                "button": 1,
                "link": 2,
                "submit": 3,
                "textbox": 4,
            }.get((c.role or "").lower(), 0)
            topk_meta[i] = np.array(
                [
                    role_id / 8.0,
                    float(bool(c.clickable)),
                    float(bool(c.focusable)),
                    float(bool(c.editable)),
                    float(bool(c.visible)),
                    min(1.0, len(_tokenize(c.text)) / 64.0),
                    cxn,
                    cyn,
                ],
                dtype=np.float32,
            )

        # Use tokenizer for encoding (preferred over hash)
        use_tokenizer = bool(self.cfg.get("use_tokenizer", True))
        
        # Encode URL using tokenizer
        if use_tokenizer:
            url_token_ids = self.url_tokenizer.encode(url or "", max_length=1)
            url_id = np.array([url_token_ids[0] if url_token_ids else 0], dtype=np.int32)
        else:
            url_id = np.array([_hash_token(url or "", self.url_vocab)], dtype=np.int32)
        
        return {
            "goal_ids": self._encode_text(goal, self.max_goal_tokens, self.goal_vocab, use_tokenizer=use_tokenizer),
            "dom_ids": self._encode_text(html, self.max_dom_tokens, self.dom_vocab, use_tokenizer=use_tokenizer),
            "url_id": url_id,
            "prev_actions": np.array(list(self._history) + [0] * (self.history_len - len(self._history)), dtype=np.int32),
            "topk_text_ids": topk_text_ids,
            "topk_meta": topk_meta,
            "score": np.array([float(score)], dtype=np.float32),
        }

    def _mask(self) -> np.ndarray:
        m = np.zeros((self.action_space.n,), dtype=np.bool_)
        m[0] = True
        for i in range(self.K):
            m[1 + i] = bool(i < self._click_mask.shape[0] and self._click_mask[i])
        base = 1 + self.K
        m[base + MacroAction.TYPE_CONFIRM] = bool(self._macros.get("type_confirm", False))
        m[base + MacroAction.SUBMIT] = bool(self._macros.get("submit", False))
        m[base + MacroAction.SCROLL_DOWN] = bool(self._macros.get("scroll_down", False))
        m[base + MacroAction.SCROLL_UP] = bool(self._macros.get("scroll_up", False))
        m[base + MacroAction.BACK] = bool(self._macros.get("back", False))
        return m

    def get_action_mask(self) -> np.ndarray:
        return self._mask().copy()

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, object]] = None):
        # Be careful with seed - ensure proper seeding for diverse exploration
        # If seed is None, gymnasium will use a random seed automatically
        super().reset(seed=seed)
        options = dict(options or {})

        # Ensure any previous evaluator/browser session is fully closed to avoid leaks
        # across episodes (which would otherwise spawn many headless Chromium processes).
        try:
            self.close()
        except Exception:
            pass

        # Obtener task real: se puede pasar via options['task'] o generarla
        task: Optional[Task] = options.get("task") if isinstance(options.get("task"), Task) else None
        if not task:
            # Try to sample from dataset loader first (for diverse PPO training)
            if self._dataset_loader:
                task = self._dataset_loader.sample_task()
            else:
                # Fallback to cached tasks
                # Select project honoring project_start_index and repository bounds
                try:
                    idx = max(0, min(len(demo_web_projects) - 1, int(self.project_start_index)))
                except Exception:
                    idx = 0
                project = demo_web_projects[idx]
                tasks = self._generate_tasks_sync(project)
                # Randomly sample a task for RL exploration (instead of always tasks[0])
                if tasks:
                    task = self.np_random.choice(tasks) if len(tasks) > 1 else tasks[0]
                else:
                    task = None
        if not task:
            raise RuntimeError("No se pudo obtener una Task real. Pase options={'task': Task} o revise pipeline.")
        self._task = task

        # Evaluador + navegador
        self._evaluator = StatefulEvaluator(task=task, web_agent_id="rl-env", should_record_gif=False)
        logger.info("ENV.reset: start")
        self._evaluator.reset()
        logger.info("ENV.reset: evaluator.reset done")
        # Initialize browser manager with reranker support
        use_reranker = bool(self.cfg.get("use_reranker", True))
        reranker_model_path = self.cfg.get("reranker_model_path")
        self._browser = BrowserManager(
            self._evaluator.page,
            use_reranker=use_reranker,
            reranker_model_path=reranker_model_path,
        )
        self._history.clear()
        self._step = 0
        self._last_partial = self._evaluator.get_partial_score()
        if self._reward_blender:
            try:
                self._reward_blender.reset()
            except Exception as e:
                logger.warning(f"RewardBlender reset failed: {e}")

        # Estado inicial de DOM y Top‑K
        logger.info("ENV.reset: snapshot start")
        try:
            # Use DOM cleaner
            html, url = self._evaluator.run_with_timeout(self._browser.snapshot_text(clean_dom=True), 3.0)
        except Exception as e:
            logger.warning(f"ENV.reset: snapshot failed: {e}")
            html, url = "", ""
        logger.info("ENV.reset: snapshot done")

        logger.info("ENV.reset: topk start")
        try:
            # Support variable K (be careful not to always use same K)
            current_k = self.K
            if self.use_variable_k and self.k_min != self.k_max:
                current_k = int(self.np_random.integers(self.k_min, self.k_max + 1))
                logger.debug(f"Using variable K={current_k} (range: {self.k_min}-{self.k_max})")
            
            use_llm_rerank = bool(self.cfg.get("use_llm_rerank", False))
            self._cands, self._click_mask, self._macros = self._evaluator.run_with_timeout(
                self._browser.topk(self._task.prompt, current_k, use_llm=use_llm_rerank), 5.0
            )
            # Ensure we have exactly self.K candidates (pad if needed)
            while len(self._cands) < self.K:
                from ..runtime.browser_manager import Candidate
                self._cands.append(Candidate(
                    idx=len(self._cands),
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
            self._cands = self._cands[:self.K]
            # Update click mask to match self.K
            if len(self._click_mask) < self.K:
                self._click_mask = np.pad(self._click_mask, (0, self.K - len(self._click_mask)), constant_values=False)
            self._click_mask = self._click_mask[:self.K]
        except Exception as e:
            logger.warning(f"ENV.reset: topk failed: {e}")
            self._cands, self._click_mask, self._macros = [], np.zeros((self.K,), dtype=np.bool_), {k: False for k in ("type_confirm", "submit", "scroll_down", "scroll_up", "back")}
        logger.info("ENV.reset: topk done")

        logger.info(
            f"ENV.reset: partial start")
        logger.info(
            f"ENV.reset: partial done tests={self._last_partial.tests_passed}/{self._last_partial.total_tests}")
        obs = self._obs(html, url, self._last_partial.raw_score)
        info = {"action_mask": self._mask(), "raw_score": self._last_partial.raw_score}
        return obs, info

    def step(self, action: int):
        if not self._evaluator or not self._browser or not self._task:
            raise RuntimeError("Env no inicializado. Llama a reset().")

        self._step += 1
        self._history.append(int(action))

        # Build human-readable action description for logging/infos
        invalid = False
        action_kind = "noop"
        action_desc = "noop"
        selected_candidate: dict | None = None
        selected_macro: str | None = None

        if int(action) != 0:
            # Classify selection
            if 1 <= int(action) < (1 + self.K):
                action_kind = "click"
                cand_idx = int(action) - 1
                if 0 <= cand_idx < len(self._cands):
                    c = self._cands[cand_idx]
                    selected_candidate = {
                        "index": cand_idx,
                        "tag": c.tag,
                        "role": c.role,
                        "text": (c.text or "")[:80],
                        "clickable": c.clickable,
                        "visible": c.visible,
                        "enabled": c.enabled,
                    }
                    t = selected_candidate["text"]
                    role = selected_candidate["role"]
                    action_desc = f"click[{cand_idx}] '{t}' role={role}"
                else:
                    action_desc = f"click[{cand_idx}] <out-of-range>"
            else:
                action_kind = "macro"
                m_idx = int(action) - (1 + self.K)
                names = list(self.layout.macros)
                selected_macro = names[m_idx] if 0 <= m_idx < len(names) else None
                action_desc = f"macro {selected_macro or m_idx}"

            # Execute
            base = self.action_adapter.adapt(int(action), self._cands, self._task)
            if base is None:
                invalid = True
            else:
                try:
                    logger.info(f"ENV.step: executing {action_desc}")
                    self._evaluator.execute_action(base)
                except Exception as e:
                    logger.warning(f"ENV.step: execute failed: {e}")
                    invalid = True
        logger.info(f"ENV.step: exec done ok={not invalid}")

        # Recalcular estado y score parcial
        logger.info("ENV.step: snapshot start")
        try:
            # Use DOM cleaner
            html, url = self._evaluator.run_with_timeout(self._browser.snapshot_text(clean_dom=True), 3.0)
        except Exception as e:
            logger.warning(f"ENV.step: snapshot failed: {e}")
            html, url = "", ""
        logger.info("ENV.step: snapshot done")

        logger.info("ENV.step: topk start")
        try:
            # Support variable K
            current_k = self.K
            if self.use_variable_k and self.k_min != self.k_max:
                current_k = int(self.np_random.integers(self.k_min, self.k_max + 1))
            
            use_llm_rerank = bool(self.cfg.get("use_llm_rerank", False))
            self._cands, self._click_mask, self._macros = self._evaluator.run_with_timeout(
                self._browser.topk(self._task.prompt, current_k, use_llm=use_llm_rerank), 5.0
            )
            # Ensure we have exactly self.K candidates
            while len(self._cands) < self.K:
                from ..runtime.browser_manager import Candidate
                self._cands.append(Candidate(
                    idx=len(self._cands),
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
            self._cands = self._cands[:self.K]
            # Update click mask
            if len(self._click_mask) < self.K:
                self._click_mask = np.pad(self._click_mask, (0, self.K - len(self._click_mask)), constant_values=False)
            self._click_mask = self._click_mask[:self.K]
        except Exception as e:
            logger.warning(f"ENV.step: topk failed: {e}")
            self._cands, self._click_mask, self._macros = [], np.zeros((self.K,), dtype=np.bool_), {k: False for k in ("type_confirm", "submit", "scroll_down", "scroll_up", "back")}
        logger.info("ENV.step: topk done")
        partial = self._evaluator.get_partial_score()
        logger.info(f"ENV.step: partial done tests={partial.tests_passed}/{partial.total_tests}")

        # Reward = partial raw score blended with reward model. Override to 1.0 on global success.
        base_reward = float(partial.raw_score)
        step_penalty = 0.001
        invalid_penalty = 0.05 if invalid else 0.0
        base_reward = max(0.0, base_reward - step_penalty - invalid_penalty)
        shaped_reward = base_reward
        if self._reward_blender:
            try:
                shaped_reward = float(self._reward_blender.step_reward(url or "", html or "", base_reward))
            except Exception as e:
                logger.warning(f"RewardBlender shaping failed: {e}")
                shaped_reward = base_reward
        if partial.success:
            shaped_reward = 1.0

        self._last_partial = partial
        terminated = bool(partial.success)
        truncated = bool(self._step >= self.max_steps)
        # Extract latest backend events for lightweight logging
        backend_event_names: list[str] = []
        try:
            last = self._evaluator.history[-1] if self._evaluator.history else None
            if last and getattr(last, "browser_snapshot", None):
                evs = getattr(last.browser_snapshot, "backend_events", None) or []
                backend_event_names = [getattr(e, "event_name", "?") for e in evs][:5]
        except Exception:
            backend_event_names = []

        obs = self._obs(html, url, partial.raw_score)
        info = {
            "action_mask": self._mask(),
            "raw_score": partial.raw_score,
            "tests_passed": partial.tests_passed,
            "total_tests": partial.total_tests,
            "invalid_action": bool(invalid),
            # Extra debugging / introspection
            "selected_action_index": int(action),
            "selected_action_kind": action_kind,
            "selected_macro": selected_macro,
            "selected_candidate": selected_candidate,
            "action_desc": action_desc,
            "current_url": url,
            "backend_event_names": backend_event_names,
            "base_reward": float(base_reward),
            "shaped_reward": float(shaped_reward),
        }
        return obs, shaped_reward, terminated, truncated, info

    def close(self):  # type: ignore[override]
        try:
            if self._evaluator:
                self._evaluator.close()
        finally:
            self._evaluator = None

    # -------------------------
    # Introspection helpers
    # -------------------------
    def get_execution_history(self):
        """Return the list of ActionExecutionResult accumulated so far.

        Useful to adapt an episode rollout to a TaskSolution.
        """
        try:
            return self._evaluator.history if self._evaluator else []
        except Exception:
            return []

    # -------------------------
    # Async helpers
    # -------------------------
    def _run_sync(self, awaitable):
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_event_loop()
        except RuntimeError:
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
        if loop.is_running():
            # Avoid calling asyncio.run when a loop is already active (e.g., notebooks)
            new_loop = _asyncio.new_event_loop()
            try:
                _asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(awaitable)
                return result
            finally:
                new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                new_loop.close()
                _asyncio.set_event_loop(loop)
        return loop.run_until_complete(awaitable)

    def _generate_tasks_sync(self, project):
        async def _go():
            # Local import to avoid DI/bootstrap initialization when not needed
            from autoppia_iwa.entrypoints.benchmark.task_generation import generate_tasks_for_project
            tasks = await generate_tasks_for_project(
                project,
                use_cached=self.use_cached_tasks,
                cache_dir=self.tasks_cache_dir,
                prompts_per_use_case=1,
                num_use_cases=1,
                use_cases=None,
                enable_dynamic_html=False,
            )
            return tasks

        return self._run_sync(_go())


__all__ = ["IWAWebEnv", "MacroAction"]
