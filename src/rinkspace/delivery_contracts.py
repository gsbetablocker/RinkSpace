"""Shared delivery vocabulary and the historical inputs required by checks.

Visibility is a display state, never an independent accuracy verdict. Historical
inputs are explicit so the source archive can run the same checks as the project.
"""

DRAWABLE_LOCALIZATION_STATES = frozenset({
    'candidate', 'predicted',  # Frozen pre-continuity runs retain their labels.
    'model_reobserved', 'image_tracked', 'motion_predicted',
})
MODEL_REOBSERVATION_STATES = frozenset({'candidate', 'model_reobserved'})

CONFIRMED_LAYOUT_HISTORY_REVISIONS = (
    'v0.4.1', 'v0.5-ring-rows', 'v0.6-confirmed-layout',
    'v0.6.1-confirmed-layout',
)
CHECKER_HISTORY_MODELS = {
    'scripts/check_confirmed_layout.py': tuple(
        f'outputs/rink_model/ankara_2026_shared/{revision}/rink_model.json'
        for revision in CONFIRMED_LAYOUT_HISTORY_REVISIONS
    ),
    'scripts/check_whole_delivery.py': (
        'outputs/rink_model/ankara_2026_shared/v0.3/rink_model.json',
    ),
}


def has_drawable_localization(frame):
    return (frame.get('status') in DRAWABLE_LOCALIZATION_STATES and
            frame.get('projection_chain') is not None)
