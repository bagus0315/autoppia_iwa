"""DOM cleaning using BetterSoup - cleaner DOM representation for model."""

from __future__ import annotations

from typing import Optional

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    BeautifulSoup = None

from loguru import logger


class DOMCleaner:
    """Clean and simplify DOM for model input."""
    
    def __init__(self, parser: str = "html.parser"):
        """Initialize DOM cleaner.
        
        Args:
            parser: BeautifulSoup parser to use
        """
        if not BEAUTIFULSOUP_AVAILABLE:
            logger.warning("BeautifulSoup not available. DOM cleaning will be minimal.")
        self.parser = parser
    
    def clean(self, html: str, max_length: int = 50000) -> str:
        """Clean and simplify DOM HTML.
        
        Args:
            html: Raw HTML string
            max_length: Maximum output length
            
        Returns:
            Cleaned HTML string
        """
        if not html:
            return ""
        
        if not BEAUTIFULSOUP_AVAILABLE:
            # Fallback: simple truncation
            return html[:max_length]
        
        try:
            soup = BeautifulSoup(html, self.parser)
            
            # Remove script and style elements
            for element in soup(["script", "style", "noscript", "meta", "link"]):
                element.decompose()
            
            # Remove comments
            from bs4 import Comment
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()
            
            # Remove empty elements
            for element in soup.find_all():
                if not element.get_text(strip=True) and not element.attrs:
                    element.decompose()
            
            # Simplify attributes (keep only important ones)
            important_attrs = {"id", "class", "role", "type", "name", "href", "src", "alt", "title"}
            for element in soup.find_all():
                attrs_to_remove = [attr for attr in element.attrs if attr not in important_attrs]
                for attr in attrs_to_remove:
                    del element[attr]
            
            # Get cleaned HTML
            cleaned = str(soup)
            
            # Truncate if too long
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length]
                logger.debug(f"DOM truncated from {len(str(soup))} to {max_length} chars")
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"DOM cleaning failed: {e}. Returning original HTML.")
            return html[:max_length]
    
    def extract_text_content(self, html: str, max_length: int = 10000) -> str:
        """Extract only text content from DOM.
        
        Args:
            html: Raw HTML string
            max_length: Maximum output length
            
        Returns:
            Text content only
        """
        if not html:
            return ""
        
        if not BEAUTIFULSOUP_AVAILABLE:
            # Fallback: simple text extraction
            import re
            text = re.sub(r'<[^>]+>', '', html)
            return text[:max_length]
        
        try:
            soup = BeautifulSoup(html, self.parser)
            
            # Remove script and style
            for element in soup(["script", "style"]):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator=" ", strip=True)
            
            # Clean up whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            
            if len(text) > max_length:
                text = text[:max_length]
            
            return text
            
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return html[:max_length]


# Global cleaner instance
_dom_cleaner: Optional[DOMCleaner] = None


def get_dom_cleaner() -> DOMCleaner:
    """Get global DOM cleaner instance."""
    global _dom_cleaner
    if _dom_cleaner is None:
        _dom_cleaner = DOMCleaner()
    return _dom_cleaner

