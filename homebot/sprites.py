"""Chunky flat-palette pixel-art sprites built from character grids.

Each sprite is a list of equal-length strings; each character maps to a color
in PALETTE ('.' = transparent). The grid is rendered 1 char = 1 pixel, then
scaled up nearest-neighbor so the pixels stay crisp and blocky (no smoothing,
no gradients, no outlines) — matching the low-res reference style.

To tweak a sprite, edit its grid below. To add one, define a new grid and a
PALETTE entry for any new color. Keep the palette small — the chunky look comes
from using few flat colors.
"""
import pygame

PALETTE = {
    ".": None,             # transparent
    # recliner + person
    "R": (158, 48, 26),    # recliner red
    "r": (112, 30, 16),    # recliner red, darker (base lip / shadow)
    "K": (40, 42, 58),     # shirt (dark slate)
    "J": (84, 78, 184),    # jeans (periwinkle)
    "S": (54, 40, 28),     # hair / shoes (brown)
    "L": (226, 174, 130),  # skin (warm)
    # CRT television
    "W": (134, 88, 50),    # tv wood
    "w": (98, 62, 36),     # tv wood, darker
    "B": (38, 36, 44),     # tv bezel / dark plastic
    "G": (150, 164, 168),  # tv screen
    "H": (205, 214, 214),  # tv screen glare
    "N": (70, 70, 78),     # metal (antenna / knobs)
    # fridge
    "F": (208, 228, 234),  # fridge body
    "f": (170, 198, 208),  # fridge shadow / seam
    "D": (120, 130, 138),  # fridge handle
    # counter / sink
    "P": (218, 202, 168),  # countertop surface (warm cream)
    "p": (168, 148, 118),  # counter front edge
    "A": (138, 148, 158),  # sink basin rim (steel)
    "a": (88, 98, 108),    # sink basin interior
    "M": (185, 195, 205),  # faucet chrome
    # package
    "C": (198, 152, 98),   # cardboard
    "c": (150, 112, 66),   # cardboard, darker
    "T": (216, 198, 152),  # packing tape
    # bottle
    "V": (72, 132, 72),    # bottle green glass
    "v": (44, 84, 44),     # bottle dark glass / interior
    # can
    "X": (188, 192, 198),  # can aluminum
    "x": (128, 132, 140),  # can rim / shadow
    "Y": (94, 98, 106),    # pull tab
}

# Top-down man reclined in a recliner. Head = south (closest to camera);
# feet point north toward the TV. Knees visibly wider than lower legs.
RECLINER = [
    "....SS....SS....",   # feet (north)
    "....JJ....JJ....",   # lower legs
    "...JJJR..RJJJ...",  # knees (one row, 3px wide — subtle bump)
    ".RRRJJRRRRJJRRR.",  # thighs / chair arms
    ".RRRKKKKKKKKRRR.",  # lower torso
    ".RRLKKKKKKKKLRR.",  # torso / skin arms
    ".RRLLKKKKKKLLRR.",  # body
    ".RRRLKKSSKKLRRR.",  # shoulders
    "..RRRKSLLSKRRR..",  # face
    "...RRRRLLRRRR...",  # chin / headrest (south)
]

# Tiny wooden side table — sits to the man's left (west of the recliner).
TABLE = [
    ".WWWW.",
    ".WWWW.",
    ".WWWW.",
    ".w..w.",
]

# Old CRT television, true top-down. Cabinet top is the dominant surface;
# thin bezel+screen strip at south edge (screen faces the recliner);
# stand legs signal it's upright. Narrower than the original.
TV = [
    ".WWWWWWWW.",
    ".WWWWWWWW.",
    ".WWWWWWWW.",
    ".WWWWWWWW.",
    ".WWWWWWWW.",
    ".wBBBBBBw.",
    ".wBGHGGBw.",
    "..ww..ww..",
]

# Two-door fridge, top-down. Cabinet top dominates; front face (south edge)
# shows a dark door gasket outline + handles so it clearly reads as a door.
FRIDGE = [
    "FFFFFFFFFF",
    "FFFFFFFFFF",
    "FFFFFFFFFF",
    "FFFFFFFFFF",
    "fBBBBBBBBf",
    "fDFFFFFDFf",
    "fBBBBBBBBf",
]

# Kitchen counter (plain work surface along the north wall, top-down).
COUNTER = [
    "PPPPPPPPPP",
    "PPPPPPPPPP",
    "PPPPPPPPPP",
    "pppppppppp",
]

# Kitchen sink (top-down: counter surround + basin + faucet).
SINK = [
    "PPPPPPPPPP",
    "PAAAAAAAAP",
    "PAaaaaaaAP",
    "PAaaaMaaAP",
    "PAaaaaaaAP",
    "PAAAAAAAAP",
    "pppppppppp",
]

# Cardboard package with taped cross, top-down.
PACKAGE = [
    ".CCCCC.",
    "CCCTCCC",
    "CCCTCCC",
    "CTTTTTC",
    "CCCTCCC",
    "CCCTCCC",
    ".cCCCc.",
]

# Glass bottle, top-down (oval body, narrow neck at top).
BOTTLE = [
    "..VVV..",
    ".VVVVV.",
    ".VvvvV.",
    ".VvvvV.",
    ".VVVVV.",
    "..VVV..",
    "...V...",
]

# Aluminum can, top-down (round with pull tab visible).
CAN = [
    "..XXX..",
    ".XXXXX.",
    ".XxYxX.",
    ".XXXXX.",
    "..xxx..",
]

DEFAULT_SCALE = 6  # px per source pixel — keep consistent across sprites


def make_sprite(grid, scale: int = DEFAULT_SCALE, palette: dict = PALETTE) -> pygame.Surface:
    """Build a crisp, scaled pixel-art Surface (with alpha) from a character grid."""
    h = len(grid)
    w = max(len(row) for row in grid)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            color = palette.get(ch)
            if color is not None:
                surf.set_at((x, y), color)
    # scale() is nearest-neighbor — keeps hard pixel edges (smoothscale would blur them)
    return pygame.transform.scale(surf, (w * scale, h * scale))
