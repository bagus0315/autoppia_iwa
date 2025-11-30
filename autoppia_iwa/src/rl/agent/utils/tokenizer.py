"""Tokenizer for RL agent observations - replaces hash-based encoding."""

from __future__ import annotations

from typing import List, Optional

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

from loguru import logger


class ObservationTokenizer:
    """Tokenizer for encoding text observations into token IDs."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        vocab_size: int = 30522,
        max_length: int = 512,
    ):
        """Initialize tokenizer.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            vocab_size: Vocabulary size (for observation space)
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.vocab_size = len(self.tokenizer)
                logger.info(f"✅ Loaded tokenizer: {model_name} (vocab_size={self.vocab_size})")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {model_name}: {e}. Using fallback.")
                self.tokenizer = None
        else:
            logger.warning("transformers not available. Using fallback tokenization.")
            self.tokenizer = None
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to encode
            max_length: Maximum length (defaults to self.max_length)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        max_len = max_length or self.max_length
        
        if self.tokenizer:
            try:
                encoded = self.tokenizer(
                    text,
                    max_length=max_len,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=None,
                )
                return encoded["input_ids"][:max_len]
            except Exception as e:
                logger.warning(f"Tokenizer encoding failed: {e}. Using fallback.")
        
        # Fallback: simple word-based tokenization with hash
        import hashlib
        import re
        
        tokens = re.findall(r'\w+', (text or "").lower())
        token_ids = []
        for token in tokens[:max_len]:
            # Hash-based fallback (same as original but with proper vocab size)
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            token_id = int.from_bytes(digest, "little") % self.vocab_size
            token_ids.append(token_id)
        
        # Pad if needed
        if padding and len(token_ids) < max_len:
            token_ids.extend([0] * (max_len - len(token_ids)))
        
        return token_ids[:max_len]
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> List[List[int]]:
        """Encode batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum length
            padding: Whether to pad
            truncation: Whether to truncate
            
        Returns:
            List of token ID lists
        """
        return [self.encode(text, max_length, padding, truncation) for text in texts]


# Global tokenizer instances (lazy-loaded)
_goal_tokenizer: Optional[ObservationTokenizer] = None
_dom_tokenizer: Optional[ObservationTokenizer] = None
_url_tokenizer: Optional[ObservationTokenizer] = None


def get_goal_tokenizer(vocab_size: int = 4096) -> ObservationTokenizer:
    """Get tokenizer for goal/prompt text."""
    global _goal_tokenizer
    if _goal_tokenizer is None:
        _goal_tokenizer = ObservationTokenizer(
            model_name="bert-base-uncased",
            vocab_size=vocab_size,
            max_length=48,
        )
    return _goal_tokenizer


def get_dom_tokenizer(vocab_size: int = 8192) -> ObservationTokenizer:
    """Get tokenizer for DOM text."""
    global _dom_tokenizer
    if _dom_tokenizer is None:
        _dom_tokenizer = ObservationTokenizer(
            model_name="bert-base-uncased",
            vocab_size=vocab_size,
            max_length=200,
        )
    return _dom_tokenizer


def get_url_tokenizer(vocab_size: int = 1024) -> ObservationTokenizer:
    """Get tokenizer for URL text."""
    global _url_tokenizer
    if _url_tokenizer is None:
        _url_tokenizer = ObservationTokenizer(
            model_name="bert-base-uncased",
            vocab_size=vocab_size,
            max_length=32,
        )
    return _url_tokenizer

