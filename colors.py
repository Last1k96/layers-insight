# colors.py
from enum import Enum, auto

# Border colors for node states as an enum
class BorderColorType(Enum):
    DEFAULT = auto()
    ERROR = auto()
    SUCCESS = auto()
    PROCESSING = auto()
    SELECTED = auto()
    SELECTED_DIFFERENT_TYPE = auto()  # New color for selected nodes of different types

# Mapping from enum to actual color values
BORDER_COLORS = {
    BorderColorType.DEFAULT: "#242424",  # Default state
    BorderColorType.ERROR: "red",        # Error state
    BorderColorType.SUCCESS: "green",    # Success state
    BorderColorType.PROCESSING: "#BA8E23", # Processing state
    BorderColorType.SELECTED: "orange",   # Selected state
    BorderColorType.SELECTED_DIFFERENT_TYPE: "purple"  # Different type selected state
}
