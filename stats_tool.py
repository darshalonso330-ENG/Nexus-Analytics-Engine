"""
╔══════════════════════════════════════════════════════════════════╗
║           Statistical Analysis Tool  —  Professional Edition    ║
║           Built with tkinter · pandas · matplotlib · seaborn    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  THEME PALETTE
# ─────────────────────────────────────────────────────────────────
DARK_BG     = "#0f1117"
PANEL_BG    = "#1a1d2e"
CARD_BG     = "#22253a"
ACCENT      = "#6c63ff"
ACCENT2     = "#00d4aa"
ACCENT3     = "#ff6584"
TEXT_MAIN   = "#e8eaf6"
TEXT_MUTED  = "#8892b0"
BORDER      = "#2e3250"
SUCCESS     = "#00d4aa"
WARNING     = "#ffd166"
ERROR       = "#ff6584"

FONT_H1     = ("Helvetica Neue", 20, "bold")
FONT_H2     = ("Helvetica Neue", 13, "bold")
FONT_BODY   = ("Helvetica Neue", 10)
FONT_MONO   = ("Courier New", 10)
FONT_SMALL  = ("Helvetica Neue", 9)
FONT_LABEL  = ("Helvetica Neue", 10, "bold")


# ─────────────────────────────────────────────────────────────────
#  SAMPLE DATASET GENERATOR
# ─────────────────────────────────────────────────────────────────
def generate_sample_dataset() -> pd.DataFrame:
    """Create a rich synthetic dataset that exercises all features."""
    np.random.seed(42)
    n = 200
    departments = np.random.choice(["Engineering", "Marketing", "Finance", "HR", "Design"], n)
    experience  = np.random.randint(1, 16, n)
    base_salary = 40_000 + experience * 3_500 + np.random.normal(0, 5_000, n)
    base_salary += np.where(departments == "Engineering", 8_000, 0)
    base_salary += np.where(departments == "Finance",     5_000, 0)
    age          = 22 + experience + np.random.randint(0, 8, n)
    performance  = np.clip(experience * 0.4 + np.random.normal(5, 1.5, n), 1, 10).round(1)
    projects     = np.random.poisson(experience * 0.8 + 1, n)

    return pd.DataFrame({
        "Age":          age.astype(int),
        "Experience":   experience,
        "Salary":       base_salary.round(2),
        "Performance":  performance,
        "Projects":     projects,
        "Department":   departments,
    })


# ─────────────────────────────────────────────────────────────────
#  STATISTICS ENGINE
# ─────────────────────────────────────────────────────────────────
class StatisticsEngine:
    """Pure-logic layer — no GUI dependencies."""

    OPERATIONS = ["Mean", "Median", "Mode", "Range",
                  "Variance", "Std Deviation",
                  "Min", "Max", "Sum", "Count",
                  "Skewness", "Kurtosis",
                  "25th Percentile", "75th Percentile", "IQR",
                  "Full Summary"]

    @staticmethod
    def compute(series: pd.Series, operation: str) -> str:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return "No numeric data in this column."
        op = operation.lower()
        if   op == "mean":             return f"{s.mean():.4f}"
        elif op == "median":           return f"{s.median():.4f}"
        elif op == "mode":
            m = s.mode()
            return ", ".join(f"{v:.4f}" for v in m[:5]) + (" …" if len(m) > 5 else "")
        elif op == "range":            return f"{s.max() - s.min():.4f}"
        elif op == "variance":         return f"{s.var():.4f}"
        elif op == "std deviation":    return f"{s.std():.4f}"
        elif op == "min":              return f"{s.min():.4f}"
        elif op == "max":              return f"{s.max():.4f}"
        elif op == "sum":              return f"{s.sum():.4f}"
        elif op == "count":            return str(int(s.count()))
        elif op == "skewness":         return f"{s.skew():.4f}"
        elif op == "kurtosis":         return f"{s.kurt():.4f}"
        elif op == "25th percentile":  return f"{s.quantile(0.25):.4f}"
        elif op == "75th percentile":  return f"{s.quantile(0.75):.4f}"
        elif op == "iqr":              return f"{s.quantile(0.75) - s.quantile(0.25):.4f}"
        elif op == "full summary":
            desc = s.describe()
            lines = [f"  {k:<8} {v:.4f}" for k, v in desc.items()]
            lines.append(f"  {'skew':<8} {s.skew():.4f}")
            lines.append(f"  {'kurt':<8} {s.kurt():.4f}")
            return "\n".join(lines)
        return "Unknown operation."


# ─────────────────────────────────────────────────────────────────
#  VISUALIZATION ENGINE
# ─────────────────────────────────────────────────────────────────
class VisualizationEngine:
    """Handles all matplotlib / seaborn chart generation."""

    PLOT_TYPES = ["Histogram", "Box Plot", "Bar Plot",
                  "Scatter Plot", "Pie Chart",
                  "Violin Plot", "KDE Plot",
                  "Pair Plot", "Heatmap (Correlation)"]

    @staticmethod
    def _apply_theme():
        sns.set_theme(style="darkgrid", palette="muted")
        plt.rcParams.update({
            "figure.facecolor":  "#1a1d2e",
            "axes.facecolor":    "#22253a",
            "axes.edgecolor":    "#2e3250",
            "axes.labelcolor":   "#e8eaf6",
            "xtick.color":       "#8892b0",
            "ytick.color":       "#8892b0",
            "text.color":        "#e8eaf6",
            "grid.color":        "#2e3250",
            "grid.alpha":        0.6,
            "font.family":       "sans-serif",
        })

    @classmethod
    def plot(cls, df: pd.DataFrame, cols: list[str], plot_type: str):
        cls._apply_theme()
        ptype = plot_type.lower()

        # ── helpers ──────────────────────────────────────────────
        PALETTE = [ACCENT, ACCENT2, ACCENT3, "#ffd166", "#06d6a0", "#118ab2"]

        def numeric_cols():
            return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

        def first_numeric():
            nc = numeric_cols()
            if not nc:
                raise ValueError("Please select at least one numeric column.")
            return nc[0]

        # ── dispatch ──────────────────────────────────────────────
        if ptype == "histogram":
            col = first_numeric()
            fig, ax = plt.subplots(figsize=(8, 5))
            n_val = df[col].dropna()
            ax.hist(n_val, bins=30, color=ACCENT, edgecolor=DARK_BG, alpha=0.85)
            ax.axvline(n_val.mean(),   color=ACCENT2, lw=2, ls="--", label=f"Mean  {n_val.mean():.2f}")
            ax.axvline(n_val.median(), color=ACCENT3, lw=2, ls=":",  label=f"Median {n_val.median():.2f}")
            ax.set_title(f"Histogram — {col}", fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel(col); ax.set_ylabel("Frequency")
            ax.legend(fontsize=9)
            fig.tight_layout()

        elif ptype == "box plot":
            nc = numeric_cols()
            if not nc: raise ValueError("Select at least one numeric column.")
            fig, ax = plt.subplots(figsize=(max(6, len(nc)*2), 5))
            data_list = [df[c].dropna() for c in nc]
            bp = ax.boxplot(data_list, patch_artist=True, notch=True,
                            medianprops=dict(color=ACCENT2, lw=2))
            for patch, color in zip(bp["boxes"], PALETTE * 10):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax.set_xticklabels(nc, rotation=15, ha="right")
            ax.set_title("Box Plot", fontsize=14, fontweight="bold", pad=12)
            fig.tight_layout()

        elif ptype == "bar plot":
            col = cols[0]
            fig, ax = plt.subplots(figsize=(8, 5))
            vc = df[col].value_counts().head(15)
            bars = ax.bar(vc.index.astype(str), vc.values, color=PALETTE[:len(vc)],
                          edgecolor=DARK_BG, alpha=0.85)
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        str(val), ha="center", va="bottom", fontsize=8, color=TEXT_MUTED)
            ax.set_title(f"Bar Plot — {col}", fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel(col); ax.set_ylabel("Count")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()

        elif ptype == "scatter plot":
            nc = numeric_cols()
            if len(nc) < 2: raise ValueError("Select at least 2 numeric columns for Scatter Plot.")
            x_col, y_col = nc[0], nc[1]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df[x_col], df[y_col], color=ACCENT, alpha=0.6,
                       edgecolors=PANEL_BG, s=55)
            # regression line
            try:
                m, b = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                xr = np.linspace(df[x_col].min(), df[x_col].max(), 200)
                ax.plot(xr, m*xr+b, color=ACCENT2, lw=2, label=f"y={m:.2f}x+{b:.2f}")
                ax.legend(fontsize=9)
            except Exception:
                pass
            ax.set_title(f"Scatter — {x_col} vs {y_col}", fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel(x_col); ax.set_ylabel(y_col)
            fig.tight_layout()

        elif ptype == "pie chart":
            col = cols[0]
            fig, ax = plt.subplots(figsize=(7, 6))
            vc = df[col].value_counts().head(10)
            wedges, texts, autotexts = ax.pie(
                vc.values, labels=vc.index.astype(str),
                autopct="%1.1f%%", startangle=140,
                colors=sns.color_palette("husl", len(vc)),
                wedgeprops=dict(edgecolor=DARK_BG, linewidth=1.5))
            for at in autotexts: at.set_color(DARK_BG); at.set_fontsize(9)
            ax.set_title(f"Pie Chart — {col}", fontsize=14, fontweight="bold", pad=12)
            fig.tight_layout()

        elif ptype == "violin plot":
            nc = numeric_cols()
            if not nc: raise ValueError("Select at least one numeric column.")
            fig, ax = plt.subplots(figsize=(max(6, len(nc)*2), 5))
            data_list = [df[c].dropna().values for c in nc]
            parts = ax.violinplot(data_list, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(PALETTE[i % len(PALETTE)]); pc.set_alpha(0.75)
            ax.set_xticks(range(1, len(nc)+1)); ax.set_xticklabels(nc, rotation=15, ha="right")
            ax.set_title("Violin Plot", fontsize=14, fontweight="bold", pad=12)
            fig.tight_layout()

        elif ptype == "kde plot":
            nc = numeric_cols()
            if not nc: raise ValueError("Select at least one numeric column.")
            fig, ax = plt.subplots(figsize=(8, 5))
            for i, col in enumerate(nc):
                series = df[col].dropna()
                series.plot.kde(ax=ax, color=PALETTE[i % len(PALETTE)],
                                lw=2.5, label=col, alpha=0.85)
                ax.fill_between(ax.lines[-1].get_xdata(),
                                ax.lines[-1].get_ydata(),
                                alpha=0.12, color=PALETTE[i % len(PALETTE)])
            ax.set_title("KDE Plot", fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel("Value"); ax.set_ylabel("Density")
            ax.legend(fontsize=9)
            fig.tight_layout()

        elif ptype == "pair plot":
            nc = numeric_cols()[:5]  # cap at 5 for readability
            if len(nc) < 2: raise ValueError("Select at least 2 numeric columns for Pair Plot.")
            pg = sns.pairplot(df[nc].dropna(), diag_kind="kde",
                              plot_kws=dict(alpha=0.5, color=ACCENT),
                              diag_kws=dict(color=ACCENT2, fill=True))
            pg.figure.suptitle("Pair Plot", y=1.02, fontsize=14, fontweight="bold",
                                color=TEXT_MAIN)
            fig = pg.figure

        elif ptype == "heatmap (correlation)":
            nc = numeric_cols()
            if len(nc) < 2: raise ValueError("Select at least 2 numeric columns for Heatmap.")
            corr = df[nc].corr()
            fig, ax = plt.subplots(figsize=(max(6, len(nc)), max(5, len(nc)-1)))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
                        cmap="coolwarm", center=0, ax=ax,
                        cbar_kws={"shrink": 0.75})
            ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
            fig.tight_layout()
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        return fig


# ─────────────────────────────────────────────────────────────────
#  MAIN APPLICATION CLASS
# ─────────────────────────────────────────────────────────────────
class StatApp(tk.Tk):
    """Root window — orchestrates all panels."""

    def __init__(self):
        super().__init__()
        self.title("Statistical Analysis Tool")
        self.geometry("1150x760")
        self.minsize(900, 640)
        self.configure(bg=DARK_BG)
        self.resizable(True, True)

        self.df: pd.DataFrame | None = None
        self._stat_engine = StatisticsEngine()
        self._viz_engine   = VisualizationEngine()

        self._build_ui()
        self._load_sample()   # start with demo data

    # ─── UI CONSTRUCTION ──────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        # two-column body
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.columnconfigure(0, weight=2, minsize=320)
        body.columnconfigure(1, weight=3, minsize=420)
        body.rowconfigure(0, weight=1)

        left  = tk.Frame(body, bg=DARK_BG)
        right = tk.Frame(body, bg=DARK_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        self._build_data_panel(left)
        self._build_stats_panel(left)
        self._build_viz_panel(right)
        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL_BG, height=62)
        hdr.pack(fill="x", padx=0, pady=0)
        hdr.pack_propagate(False)

        # coloured accent strip
        strip = tk.Frame(hdr, bg=ACCENT, width=5)
        strip.pack(side="left", fill="y")

        tk.Label(hdr, text="📊", font=("Helvetica Neue", 22),
                 bg=PANEL_BG, fg=ACCENT).pack(side="left", padx=(16, 8))
        tk.Label(hdr, text="Statistical Analysis Tool",
                 font=FONT_H1, bg=PANEL_BG, fg=TEXT_MAIN).pack(side="left")
        tk.Label(hdr, text="Professional Edition",
                 font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_MUTED).pack(side="left", padx=(10, 0), pady=(8, 0))

        # right-side info
        info = tk.Frame(hdr, bg=PANEL_BG)
        info.pack(side="right", padx=20)
        self._row_label = tk.Label(info, text="Rows: —", font=FONT_SMALL,
                                   bg=PANEL_BG, fg=TEXT_MUTED)
        self._row_label.pack(anchor="e")
        self._col_label = tk.Label(info, text="Cols: —", font=FONT_SMALL,
                                   bg=PANEL_BG, fg=TEXT_MUTED)
        self._col_label.pack(anchor="e")

    # ── DATA PANEL ─────────────────────────────────────────────────
    def _build_data_panel(self, parent):
        card = self._card(parent, "📁  Dataset")
        card.pack(fill="x", pady=(0, 10))

        btn_row = tk.Frame(card, bg=CARD_BG)
        btn_row.pack(fill="x", padx=12, pady=(4, 8))

        self._btn(btn_row, "Load CSV File", self._load_file,
                  color=ACCENT).pack(side="left", padx=(0, 8))
        self._btn(btn_row, "Sample Data", self._load_sample,
                  color=ACCENT2).pack(side="left")
        self._btn(btn_row, "Export Stats", self._export_stats,
                  color=TEXT_MUTED).pack(side="right")

        # mini data preview table
        tbl_frame = tk.Frame(card, bg=CARD_BG)
        tbl_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        cols = ("Col1", "Col2", "Col3", "Col4", "Col5")
        self._preview_tree = ttk.Treeview(tbl_frame, columns=cols,
                                          show="headings", height=6)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=CARD_BG, foreground=TEXT_MAIN,
                        fieldbackground=CARD_BG, borderwidth=0,
                        rowheight=22, font=FONT_MONO)
        style.configure("Treeview.Heading",
                        background=PANEL_BG, foreground=ACCENT,
                        relief="flat", font=FONT_LABEL)
        style.map("Treeview", background=[("selected", ACCENT)])

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",
                            command=self._preview_tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal",
                            command=self._preview_tree.xview)
        self._preview_tree.configure(yscrollcommand=vsb.set,
                                     xscrollcommand=hsb.set)
        self._preview_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

    # ── STATISTICS PANEL ───────────────────────────────────────────
    def _build_stats_panel(self, parent):
        card = self._card(parent, "🔢  Statistics")
        card.pack(fill="x", pady=(0, 10))

        grid = tk.Frame(card, bg=CARD_BG)
        grid.pack(fill="x", padx=12, pady=8)
        grid.columnconfigure(1, weight=1)

        # column selector
        tk.Label(grid, text="Column", font=FONT_LABEL,
                 bg=CARD_BG, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w", pady=4)
        self._stat_col_var = tk.StringVar()
        self._stat_col_cb  = ttk.Combobox(grid, textvariable=self._stat_col_var,
                                           state="readonly", font=FONT_BODY)
        self._stat_col_cb.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=4)

        # operation selector
        tk.Label(grid, text="Operation", font=FONT_LABEL,
                 bg=CARD_BG, fg=TEXT_MUTED).grid(row=1, column=0, sticky="w", pady=4)
        self._op_var = tk.StringVar(value="Mean")
        self._op_cb  = ttk.Combobox(grid, textvariable=self._op_var,
                                     values=StatisticsEngine.OPERATIONS,
                                     state="readonly", font=FONT_BODY)
        self._op_cb.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)

        self._btn(card, "▶  Calculate", self._calculate,
                  color=ACCENT).pack(pady=(2, 8), padx=12, anchor="w")

        # result display
        res_frame = tk.Frame(card, bg=PANEL_BG, bd=0, relief="flat")
        res_frame.pack(fill="x", padx=12, pady=(0, 12))

        tk.Label(res_frame, text="RESULT", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_MUTED).pack(anchor="w", padx=10, pady=(6, 0))

        self._result_var = tk.StringVar(value="—")
        self._result_lbl = tk.Label(res_frame, textvariable=self._result_var,
                                     font=("Courier New", 13, "bold"),
                                     bg=PANEL_BG, fg=ACCENT2,
                                     wraplength=300, justify="left",
                                     pady=6, padx=10)
        self._result_lbl.pack(fill="x")

        # thin accent bottom line
        tk.Frame(res_frame, bg=ACCENT2, height=2).pack(fill="x")

    # ── VISUALIZATION PANEL ────────────────────────────────────────
    def _build_viz_panel(self, parent):
        card = self._card(parent, "📈  Visualization")
        card.pack(fill="both", expand=True)

        top = tk.Frame(card, bg=CARD_BG)
        top.pack(fill="x", padx=12, pady=8)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        # column multi-select
        tk.Label(top, text="Columns", font=FONT_LABEL,
                 bg=CARD_BG, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w", pady=4)
        self._viz_col_lb_frame = tk.Frame(top, bg=CARD_BG)
        self._viz_col_lb_frame.grid(row=0, column=1, sticky="ew", padx=(8, 16), pady=4)
        self._viz_col_lb = tk.Listbox(self._viz_col_lb_frame,
                                       selectmode="multiple", height=4,
                                       bg=PANEL_BG, fg=TEXT_MAIN,
                                       selectbackground=ACCENT,
                                       selectforeground=TEXT_MAIN,
                                       activestyle="none",
                                       font=FONT_BODY, bd=0,
                                       highlightthickness=0)
        self._viz_col_lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(self._viz_col_lb_frame, command=self._viz_col_lb.yview,
                          bg=PANEL_BG, troughcolor=PANEL_BG)
        sb.pack(side="right", fill="y")
        self._viz_col_lb.config(yscrollcommand=sb.set)

        # plot type
        tk.Label(top, text="Plot Type", font=FONT_LABEL,
                 bg=CARD_BG, fg=TEXT_MUTED).grid(row=0, column=2, sticky="w", pady=4)
        self._plot_var = tk.StringVar(value="Histogram")
        self._plot_cb  = ttk.Combobox(top, textvariable=self._plot_var,
                                       values=VisualizationEngine.PLOT_TYPES,
                                       state="readonly", font=FONT_BODY, width=24)
        self._plot_cb.grid(row=0, column=3, sticky="ew", padx=(8, 0), pady=4)

        btn_row = tk.Frame(card, bg=CARD_BG)
        btn_row.pack(fill="x", padx=12, pady=(0, 10))
        self._btn(btn_row, "▶  Generate Plot", self._plot_in_panel,
                  color=ACCENT3).pack(side="left", padx=(0, 8))
        self._btn(btn_row, "⊞  Pop-out Window", self._plot_popup,
                  color=ACCENT).pack(side="left")
        self._btn(btn_row, "💾  Save Plot", self._save_plot,
                  color=TEXT_MUTED).pack(side="right")

        # embedded canvas area
        self._canvas_frame = tk.Frame(card, bg=PANEL_BG,
                                       highlightbackground=BORDER,
                                       highlightthickness=1)
        self._canvas_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._placeholder_lbl = tk.Label(
            self._canvas_frame,
            text="✦  Select columns and a plot type, then click  ▶  Generate Plot",
            font=FONT_BODY, bg=PANEL_BG, fg=TEXT_MUTED)
        self._placeholder_lbl.place(relx=0.5, rely=0.5, anchor="center")

        self._embedded_canvas = None
        self._current_fig     = None

    # ── STATUS BAR ────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = tk.Frame(self, bg=PANEL_BG, height=26)
        bar.pack(fill="x", side="bottom")
        tk.Frame(bar, bg=ACCENT, width=4).pack(side="left", fill="y")
        self._status_var = tk.StringVar(value="Ready — load a CSV or use sample data.")
        tk.Label(bar, textvariable=self._status_var,
                 font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_MUTED,
                 anchor="w").pack(side="left", padx=10)

    # ─── WIDGET HELPERS ──────────────────────────────────────────
    def _card(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=CARD_BG, bd=0,
                         highlightbackground=BORDER, highlightthickness=1)
        tk.Frame(outer, bg=ACCENT, height=3).pack(fill="x")
        tk.Label(outer, text=title, font=FONT_H2,
                 bg=CARD_BG, fg=TEXT_MAIN, pady=6, padx=12,
                 anchor="w").pack(fill="x")
        return outer

    @staticmethod
    def _btn(parent, text, cmd, color=ACCENT) -> tk.Button:
        b = tk.Button(parent, text=text, command=cmd,
                      font=FONT_LABEL, bg=color, fg=DARK_BG,
                      activebackground=color, activeforeground=DARK_BG,
                      relief="flat", bd=0, padx=14, pady=5,
                      cursor="hand2")
        # hover effect
        b.bind("<Enter>", lambda e: b.config(bg=_lighten(color)))
        b.bind("<Leave>", lambda e: b.config(bg=color))
        return b

    def _status(self, msg: str, kind: str = "info"):
        colour_map = {"info": TEXT_MUTED, "ok": SUCCESS, "err": ERROR, "warn": WARNING}
        self._status_var.set(msg)
        icon = {"info": "ℹ", "ok": "✔", "err": "✖", "warn": "⚠"}.get(kind, "")
        self._status_var.set(f"{icon}  {msg}")

    # ─── DATA LOGIC ──────────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            self._status("No file selected.", "warn"); return
        try:
            df = pd.read_csv(path)
            self._apply_dataset(df, f"Loaded: {path.split('/')[-1]}")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            self._status(f"Error loading file: {exc}", "err")

    def _load_sample(self):
        df = generate_sample_dataset()
        self._apply_dataset(df, "Sample dataset loaded (200 employee records).")

    def _apply_dataset(self, df: pd.DataFrame, msg: str):
        self.df = df
        self._refresh_preview()
        self._refresh_dropdowns()
        self._row_label.config(text=f"Rows: {len(df):,}")
        self._col_label.config(text=f"Cols: {len(df.columns)}")
        self._status(msg, "ok")

    def _refresh_preview(self):
        tree = self._preview_tree
        # clear
        tree.delete(*tree.get_children())
        cols = list(self.df.columns)
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(80, len(c)*9), minwidth=60, anchor="center")
        for _, row in self.df.head(8).iterrows():
            tree.insert("", "end", values=[str(v)[:18] for v in row])

    def _refresh_dropdowns(self):
        cols = list(self.df.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]

        self._stat_col_cb["values"] = cols
        if cols: self._stat_col_var.set(num_cols[0] if num_cols else cols[0])

        self._viz_col_lb.delete(0, "end")
        for c in cols:
            self._viz_col_lb.insert("end", c)
        # pre-select first numeric
        if num_cols:
            idx = cols.index(num_cols[0])
            self._viz_col_lb.selection_set(idx)

    # ─── STATISTICS LOGIC ────────────────────────────────────────
    def _calculate(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load a dataset first."); return
        col = self._stat_col_var.get()
        op  = self._op_var.get()
        if col not in self.df.columns:
            self._status("Invalid column selected.", "err"); return
        try:
            result = StatisticsEngine.compute(self.df[col], op)
            self._result_var.set(result)
            self._status(f"{op} of '{col}' → {result.split(chr(10))[0]}", "ok")
        except Exception as exc:
            self._result_var.set(f"Error: {exc}")
            self._status(str(exc), "err")

    # ─── VISUALIZATION LOGIC ────────────────────────────────────
    def _selected_columns(self) -> list[str]:
        if self.df is None:
            raise ValueError("No dataset loaded.")
        idxs = self._viz_col_lb.curselection()
        if not idxs:
            raise ValueError("Select at least one column from the list.")
        all_cols = list(self.df.columns)
        return [all_cols[i] for i in idxs]

    def _generate_figure(self) -> plt.Figure:
        cols     = self._selected_columns()
        plot_type = self._plot_var.get()
        return VisualizationEngine.plot(self.df, cols, plot_type)

    def _plot_in_panel(self):
        try:
            fig = self._generate_figure()
            self._current_fig = fig
            self._embed_figure(fig)
            self._status(f"{self._plot_var.get()} generated.", "ok")
        except Exception as exc:
            messagebox.showerror("Plot Error", str(exc))
            self._status(str(exc), "err")

    def _embed_figure(self, fig):
        # destroy old canvas
        if self._embedded_canvas:
            self._embedded_canvas.get_tk_widget().destroy()
            self._embedded_canvas = None
        if self._placeholder_lbl.winfo_ismapped():
            self._placeholder_lbl.place_forget()

        canvas = FigureCanvasTkAgg(fig, master=self._canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._embedded_canvas = canvas

    def _plot_popup(self):
        try:
            fig = self._generate_figure()
            self._current_fig = fig
            # open in a new toplevel
            win = tk.Toplevel(self)
            win.title(f"{self._plot_var.get()} — Pop-out")
            win.configure(bg=PANEL_BG)
            win.geometry("850x560")
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(canvas, win)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._status(f"{self._plot_var.get()} opened in pop-out window.", "ok")
        except Exception as exc:
            messagebox.showerror("Plot Error", str(exc))
            self._status(str(exc), "err")

    def _save_plot(self):
        if self._current_fig is None:
            messagebox.showinfo("No Plot", "Generate a plot first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("JPEG", "*.jpg")])
        if not path: return
        try:
            self._current_fig.savefig(path, dpi=180, bbox_inches="tight",
                                      facecolor=PANEL_BG)
            self._status(f"Plot saved → {path.split('/')[-1]}", "ok")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _export_stats(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load a dataset first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")])
        if not path: return
        try:
            summary = self.df.describe(include="all").transpose()
            if path.endswith(".xlsx"):
                summary.to_excel(path)
            else:
                summary.to_csv(path)
            self._status(f"Statistics exported → {path.split('/')[-1]}", "ok")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))


# ─────────────────────────────────────────────────────────────────
#  COLOUR HELPER
# ─────────────────────────────────────────────────────────────────
def _lighten(hex_color: str, amount: float = 0.15) -> str:
    """Return a slightly lighter version of a hex colour."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        r = min(255, int(r + (255-r)*amount))
        g = min(255, int(g + (255-g)*amount))
        b = min(255, int(b + (255-b)*amount))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = StatApp()
    app.mainloop()