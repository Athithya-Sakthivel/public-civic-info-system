"""
Core inference package for public-civic-info-system.

This package contains the authoritative business logic for inference:
- query.py      : orchestration, policy, validation, audit
- retriever.py  : embedding + semantic retrieval (pgvector / HNSW)
- generator.py  : grounded generation + strict output validation

Design invariants:
- No channel-specific logic lives here.
- All functions are callable locally and in AWS Lambda without modification.
- Imports are absolute from inference_pipeline/ root.
"""

__all__ = [
    "query",
    "retriever",
    "generator",
]
