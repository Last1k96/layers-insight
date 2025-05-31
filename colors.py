# colors.py
from enum import Enum

# Border colors for node states as an enum with direct color values
class BorderColor(Enum):
    DEFAULT = "#242424"                # Default state
    ERROR = "red"                      # Error state
    SUCCESS = "green"                  # Success state
    PROCESSING = "#BA8E23"             # Processing state
    SELECTED = "#AA2424"                # Selected state
    SELECTED_DIFFERENT_TYPE = "purple" # Different type selected state
