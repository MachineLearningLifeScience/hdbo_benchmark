from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

SQUARE_FIG_SIZE = (8, 8)
HORIZONTAL_FIG_SIZE = (16, 8)
FONT_SIZE = 14
ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
FIGURE_PATH = ROOT_DIR / "reports" / "figures"
FIGURE_PATH.mkdir(exist_ok=True, parents=True)

COLORS_FOR_BINARY_MAPS = (
    (198 / 255, 210 / 255, 210 / 255, 1.0),
    (69 / 255, 71 / 255, 74 / 255, 1.0),
)
CMAP_BINARY = LinearSegmentedColormap.from_list(
    "Custom", COLORS_FOR_BINARY_MAPS, len(COLORS_FOR_BINARY_MAPS)
)

LINE_WIDTH_FOR_SMALL_SCREEN = 3.0
POINT_SIZE_FOR_SMALL_SCREEN = 20
FONT_SCALE_FOR_SMALL_SCREEN = 2.0
