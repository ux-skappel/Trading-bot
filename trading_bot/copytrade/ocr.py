"""Read the text in an image with macOS's built-in text recognition.

Uses Apple's Vision framework (via ocrmac): free, offline, no API key, and the
image never leaves the computer. On other systems it quietly returns None.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def image_text(data: bytes) -> Optional[str]:
    """All text found in the image, one line per text block, or None if OCR is unavailable."""
    try:
        from ocrmac import ocrmac  # type: ignore
        from PIL import Image
    except ImportError:
        return None
    try:
        lines = ocrmac.OCR(Image.open(io.BytesIO(data)), recognition_level="accurate",
                           detail=False).recognize()
    except Exception as exc:  # unreadable image, Vision error
        logger.warning("could not read text in image: %s", exc)
        return None
    return "\n".join(str(line) for line in lines if str(line).strip())
