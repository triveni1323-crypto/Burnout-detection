"""
Design system — single source of truth for all colors, fonts, spacing.
Change here, entire app updates. NASA/SpaceX visual language.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Colors:
    background:     str = "#050816"
    surface:        str = "#0A1628"
    surface_glass:  str = "rgba(10, 22, 40, 0.85)"
    accent_cyan:    str = "#00E5FF"
    accent_purple:  str = "#7B61FF"
    accent_green:   str = "#00FFB3"
    text_primary:   str = "#E8F4FD"
    text_secondary: str = "#8BA3BC"
    danger:         str = "#FF4444"
    warning:        str = "#FFB800"
    success:        str = "#00FFB3"
    border:         str = "rgba(0, 229, 255, 0.15)"


@dataclass(frozen=True)
class Fonts:
    primary:   str = "'Orbitron', 'Space Grotesk', sans-serif"
    secondary: str = "'Inter', 'Rajdhani', sans-serif"
    mono:      str = "'JetBrains Mono', 'Courier New', monospace"


@dataclass(frozen=True)
class Spacing:
    xs:  str = "4px"
    sm:  str = "8px"
    md:  str = "16px"
    lg:  str = "24px"
    xl:  str = "32px"
    xxl: str = "48px"


# Singletons — import these everywhere
COLORS   = Colors()
FONTS    = Fonts()
SPACING  = Spacing()


def get_css() -> str:
    """
    Full CSS injected once in app.py.
    Glassmorphism + dark futuristic theme.
    """
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500&family=JetBrains+Mono&display=swap');

    /* ── Base reset ─────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS.background} !important;
        color: {COLORS.text_primary};
        font-family: {FONTS.secondary};
    }}

    /* ── Hide Streamlit chrome ──────────────── */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ── Sidebar ────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {COLORS.surface} !important;
        border-right: 1px solid {COLORS.border};
    }}

    /* ── Metric cards — glassmorphism ───────── */
    [data-testid="stMetric"] {{
        background: {COLORS.surface_glass};
        border: 1px solid {COLORS.border};
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
    }}

    /* ── Buttons ────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS.accent_cyan}, {COLORS.accent_purple});
        color: {COLORS.background};
        border: none;
        border-radius: 8px;
        font-family: {FONTS.primary};
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 229, 255, 0.3);
    }}

    /* ── Text inputs ────────────────────────── */
    .stTextInput > div > div > input {{
        background: {COLORS.surface} !important;
        border: 1px solid {COLORS.border} !important;
        color: {COLORS.text_primary} !important;
        border-radius: 8px;
    }}

    /* ── Headings ───────────────────────────── */
    h1, h2, h3 {{
        font-family: {FONTS.primary} !important;
        color: {COLORS.accent_cyan} !important;
        letter-spacing: 2px;
    }}

    /* ── Dividers ───────────────────────────── */
    hr {{
        border-color: {COLORS.border};
    }}

    /* ── Info / warning boxes ───────────────── */
    [data-testid="stAlert"] {{
        background: {COLORS.surface} !important;
        border-radius: 8px;
        border-left: 3px solid {COLORS.accent_cyan};
    }}
    </style>
    """