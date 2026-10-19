"""Independent backward traversal of the existing typed late source reader."""
from .vision_board_backward_context_reader import BackwardContextTemplateBoardSourceReader
from .vision_board_late_original_context_reader import LateOriginalContextBoardSourceReader


class BackwardLateOriginalContextBoardSourceReader(
        BackwardContextTemplateBoardSourceReader, LateOriginalContextBoardSourceReader):
    """Exact clocks remain original; only source continuity traversal descends."""
    pass
