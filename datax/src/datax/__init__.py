"""datax -- a classified .docx dataset for ML file classification.

Industry tags (healthcare, finance, government) and PII/PHI tags, with the PII
vocabulary held byte-compatible with nvidia/Nemotron-PII.
"""

from .taxonomy import Taxonomy, default_taxonomy, load_taxonomy

__version__ = "0.1.0"
__all__ = ["Taxonomy", "default_taxonomy", "load_taxonomy", "__version__"]
