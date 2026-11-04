"""CampusGPT: a small, local retrieval-augmented generation toolkit.

The package is intentionally dependency-light.  It ingests lecture notes and
campus documents, splits them into overlapping chunks, and retrieves the most
relevant chunks with BM25.  No external API or hosted service is required.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
