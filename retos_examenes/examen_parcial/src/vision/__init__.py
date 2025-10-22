"""
Módulo de visión artificial para reconocimiento de cartas.
Técnicas clásicas de procesamiento de imágenes.
"""

from .preprocessing import (
    preprocess_and_warp,
    detect_multiple_cards,
    order_points,
    is_red_card,
    extract_roi_region,
    binarize_roi,
    auto_rotate_card
)

from .template_matching import (
    get_template_library,
    match_value_templates,
    match_suit_templates,
    get_best_match
)

from .classification import (
    classify_card,
    classify_multiple_cards,
    get_classification_stats,
    format_classification_text
)

__all__ = [
    'preprocess_and_warp',
    'detect_multiple_cards',
    'order_points',
    'is_red_card',
    'extract_roi_region',
    'binarize_roi',
    'auto_rotate_card',
    'get_template_library',
    'match_value_templates',
    'match_suit_templates',
    'get_best_match',
    'classify_card',
    'classify_multiple_cards',
    'get_classification_stats',
    'format_classification_text'
]