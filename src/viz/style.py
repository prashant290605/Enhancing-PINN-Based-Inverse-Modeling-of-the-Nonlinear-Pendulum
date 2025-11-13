"""Consistent matplotlib style settings for publication-quality figures."""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional


# Color palettes
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff9896",
    "info": "#9467bd",
    "dark": "#2f2f2f",
    "light": "#e0e0e0",
}

PALETTE_QUALITATIVE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

PALETTE_SEQUENTIAL = [
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
]


def set_publication_style(
    font_size: int = 12,
    font_family: str = "serif",
    use_latex: bool = False,
):
    """
    Set matplotlib style for publication-quality figures.

    Args:
        font_size: base font size
        font_family: font family ('serif', 'sans-serif', 'monospace')
        use_latex: whether to use LaTeX rendering
    """
    # Reset to defaults first
    mpl.rcdefaults()

    # Figure settings
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1
    plt.rcParams["savefig.format"] = "pdf"  # Default to PDF for publications

    # Font settings - use readable fonts
    plt.rcParams["font.size"] = font_size
    plt.rcParams["font.family"] = font_family
    
    # Enable LaTeX-style math rendering (without full LaTeX)
    plt.rcParams["mathtext.fontset"] = "stix"  # STIX fonts for math
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    if use_latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    # Axes settings
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.labelsize"] = font_size + 2
    plt.rcParams["axes.titlesize"] = font_size + 4
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE_QUALITATIVE)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # Grid settings
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.color"] = "gray"

    # Legend settings
    plt.rcParams["legend.fontsize"] = font_size - 1
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.9
    plt.rcParams["legend.edgecolor"] = "gray"
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.shadow"] = False
    plt.rcParams["legend.loc"] = "best"

    # Tick settings
    plt.rcParams["xtick.labelsize"] = font_size - 1
    plt.rcParams["ytick.labelsize"] = font_size - 1
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    # Line settings
    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["lines.markeredgewidth"] = 1.0

    # Error bar settings
    plt.rcParams["errorbar.capsize"] = 3
    
    # Image settings
    plt.rcParams["image.cmap"] = "viridis"
    plt.rcParams["image.interpolation"] = "nearest"


def set_presentation_style(font_size: int = 16):
    """
    Set matplotlib style for presentation slides.

    Args:
        font_size: base font size (larger for presentations)
    """
    set_publication_style(font_size=font_size, font_family="sans-serif", use_latex=False)

    # Larger figure size for presentations
    plt.rcParams["figure.figsize"] = (12, 7)

    # Thicker lines for visibility
    plt.rcParams["lines.linewidth"] = 3.0
    plt.rcParams["axes.linewidth"] = 2.0


def set_notebook_style(font_size: int = 11):
    """
    Set matplotlib style for Jupyter notebooks.

    Args:
        font_size: base font size
    """
    set_publication_style(font_size=font_size, font_family="sans-serif", use_latex=False)

    # Inline backend settings
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100


def get_color(name: str) -> str:
    """
    Get color from palette.

    Args:
        name: color name

    Returns:
        hex color code
    """
    return COLORS.get(name, COLORS["primary"])


def get_palette(name: str = "qualitative", n_colors: Optional[int] = None):
    """
    Get color palette.

    Args:
        name: palette name ('qualitative', 'sequential')
        n_colors: number of colors to return (None for all)

    Returns:
        list of hex color codes
    """
    if name == "qualitative":
        palette = PALETTE_QUALITATIVE
    elif name == "sequential":
        palette = PALETTE_SEQUENTIAL
    else:
        palette = PALETTE_QUALITATIVE

    if n_colors is not None:
        palette = palette[:n_colors]

    return palette


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[tuple] = None,
    **kwargs,
) -> tuple:
    """
    Create figure with subplots using current style.

    Args:
        nrows: number of rows
        ncols: number of columns
        figsize: figure size (width, height)
        **kwargs: additional arguments for plt.subplots

    Returns:
        fig, axes
    """
    if figsize is None:
        # Auto-size based on number of subplots
        width = 5 * ncols
        height = 4 * nrows
        figsize = (width, height)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    return fig, axes


def save_figure(
    fig: plt.Figure,
    path: str,
    formats: list = None,
    dpi: int = 300,
    **kwargs,
):
    """
    Save figure in multiple formats.

    Args:
        fig: matplotlib figure
        path: base path (without extension)
        formats: list of formats ['png', 'pdf', 'svg']
        dpi: resolution for raster formats
        **kwargs: additional arguments for savefig
    """
    if formats is None:
        formats = ["png"]

    from pathlib import Path

    base_path = Path(path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = base_path.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", **kwargs)


def setup_plot_style():
    """Setup plot style (alias for set_publication_style)."""
    set_publication_style()


def save_plot_dual_format(
    fig: plt.Figure,
    base_path: str,
    dpi: int = 300,
    formats: list = None,
    **kwargs
):
    """
    Save plot in both PNG and PDF formats for publications.
    
    Args:
        fig: matplotlib figure
        base_path: base path without extension (e.g., 'outputs/fig')
        dpi: resolution for raster formats
        formats: list of formats (default: ['png', 'pdf'])
        **kwargs: additional arguments for savefig
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    from pathlib import Path
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        output_path = base_path.with_suffix(f'.{fmt}')
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', **kwargs)


def generate_plot_name(
    plot_type: str,
    theta0_deg: float = None,
    damping: float = None,
    noise: float = None,
    n_sparse: int = None,
    use_passivity: bool = None,
    n_models: int = None,
    extra: str = None,
) -> str:
    """
    Generate consistent plot filename.
    
    Args:
        plot_type: type of plot (e.g., 'theta_vs_time', 'param_hist', 'energy_drift')
        theta0_deg: initial angle in degrees
        damping: damping coefficient
        noise: noise level
        n_sparse: number of sparse points
        use_passivity: whether passivity is used
        n_models: number of ensemble models
        extra: extra identifier
        
    Returns:
        filename string (without extension)
        
    Examples:
        'theta_vs_time_amp30_c005_passivity_on'
        'param_hist_g_amp30_c005_ens7'
        'energy_drift_comparison'
    """
    parts = [plot_type]
    
    if theta0_deg is not None:
        parts.append(f'amp{int(theta0_deg)}')
    
    if damping is not None:
        # Format: c005 for 0.05, c002 for 0.02
        parts.append(f'c{int(damping*1000):03d}')
    
    if noise is not None:
        # Format: n001 for 0.01, n005 for 0.05
        parts.append(f'n{int(noise*1000):03d}')
    
    if n_sparse is not None:
        parts.append(f's{n_sparse}')
    
    if use_passivity is not None:
        parts.append('passivity_on' if use_passivity else 'passivity_off')
    
    if n_models is not None:
        parts.append(f'ens{n_models}')
    
    if extra is not None:
        parts.append(extra)
    
    return '_'.join(parts)


def get_latex_labels():
    """
    Get LaTeX-style labels for common variables.
    
    Returns:
        dict of variable names to LaTeX strings
    """
    return {
        'theta': r'$\theta$',
        'theta_dot': r'$\dot{\theta}$',
        'theta_ddot': r'$\ddot{\theta}$',
        'omega': r'$\omega$',
        'time': r'$t$',
        'g': r'$g$',
        'L': r'$L$',
        'c': r'$c$',
        'damping': r'$c$',
        'energy': r'$H$',
        'H': r'$H(t)$',
        'theta_0': r'$\theta_0$',
        'rad': 'rad',
        'rad_s': 'rad/s',
        'rad_s2': 'rad/s²',
        'm_s2': 'm/s²',
        'm': 'm',
        's': 's',
        'J': 'J',
    }


def format_axis_label(variable: str, unit: str = None) -> str:
    """
    Format axis label with LaTeX-style variable and optional unit.
    
    Args:
        variable: variable name (e.g., 'theta', 'g', 'energy')
        unit: unit string (e.g., 'rad', 'm_s2')
        
    Returns:
        formatted label string
        
    Examples:
        format_axis_label('theta', 'rad') -> '$\\theta$ (rad)'
        format_axis_label('g', 'm_s2') -> '$g$ (m/s²)'
    """
    labels = get_latex_labels()
    var_label = labels.get(variable, variable)
    
    if unit:
        unit_label = labels.get(unit, unit)
        return f'{var_label} ({unit_label})'
    
    return var_label


# Initialize with publication style by default
set_publication_style()

