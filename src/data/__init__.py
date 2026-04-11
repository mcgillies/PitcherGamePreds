# Lazy imports to avoid loading mlb_data dependency when not needed
from .preprocess import (
    MatchupPreprocessor,
    prepare_temporal_split,
    OUTCOME_CLASSES,
    OUTCOME_MAP,
)

# These require mlb_data package - import explicitly when needed
# from .collect import collect_all_data
# from .features import build_features
