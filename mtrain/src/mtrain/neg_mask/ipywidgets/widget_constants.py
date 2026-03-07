# ------------------------------------------------------------------
# Widget Constants - Shared across all annotation widgets
# ------------------------------------------------------------------

# Label constants for widget_2.py (original)
WIDGET2_LABEL_TRASH = 1
WIDGET2_LABEL_OTHER = 2
WIDGET2_LABEL_UNKNOWN = 3

WIDGET2_LABEL_FOLDER = {
    WIDGET2_LABEL_TRASH: "trash",
    WIDGET2_LABEL_OTHER: "other",
    WIDGET2_LABEL_UNKNOWN: "unknown",
}

# Label constants for widget_6.py (binary classification)
WIDGET6_LABEL_OTHER = 0
WIDGET6_LABEL_TRASH = 1

WIDGET6_LABEL_FOLDER = {
    WIDGET6_LABEL_OTHER: "other",
    WIDGET6_LABEL_TRASH: "trash",
}

# Theme support removed - dark mode doesn't work properly

# UI constants
BBOX_INSET = 5  # pixels the drawn rectangle is inset from the actual bbox edge
DEFAULT_CROP_PAD = 220  # default crop padding
DEFAULT_BBOX_PAD = 20   # default bbox padding