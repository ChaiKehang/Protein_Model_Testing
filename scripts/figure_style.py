"""Shared web-report figure styling; preserves all measurement values."""
from functools import wraps
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

INK = "#223747"
MUTED = "#617586"
TEAL = "#087F76"
BLUE = "#365D9A"
AMBER = "#A35E20"
PURPLE = "#79548E"
SLATE = "#75869D"
COLORS = [TEAL, BLUE, AMBER, PURPLE]
MARKERS = ["o", "s", "D", "^"]
LINES = ["-", "--", "-.", ":"]

FIGURES = {
    "length_scaling": (1, "输入变长，计算成本怎样增长", "BF16 · 单样本 · 每项 30 次稳态测量"),
    "batch_scaling": (2, "Batch 的收益与显存边界", "真实并行 batch · 吞吐按累计实测时间计算"),
    "task_boundaries": (3, "同一个任务，三种计时口径", "泛素 · 首次调用 / 稳态 / 热交付分别展示"),
    "structure_quality": (4, "泛素预测结构与实验结构对照", "1UBQ A · 76 个 Cα · Kabsch 刚体对齐"),
    "crambin_quality": (5, "第二个真实蛋白：逐残基结构误差", "1CRN A · 46 个 Cα · 生成时仅提供序列"),
    "precision": (6, "精度取舍：速度与显存同时看", "同一 checkpoint · 76 残基 · 主干精度受控"),
    "hotspots": (7, "GPU 时间主要花在哪些算子上", "独立 profiler 调用 · 仅聚合 ATen self CUDA 时间"),
    "microbench": (8, "把真实形状提取成算子微基准", "固定合成输入 · 10 次预热 / 50 次 GPU Event 测量"),
    "gpu_monitor": (9, "整个套件运行期间的 GPU 状态", "500 ms 采样 · 包括模型加载、推理与进程间隙"),
    "repeatability": (11, "重复进程的波动也应保留", "3 个独立进程 · 每个进程 30 次原始测量 · 不平滑"),
    "nsys_hotspots": (12, "Nsight：不同输入的主要 CUDA kernel", "BF16 · B1 · 5 次捕获 · 编号对应完整名称表"),
    "nsys_activity": (13, "Nsight：GPU 工作与时间线空档", "GPU 活动区间取并集 · 包含工具开销 · 不等同于 SM 占用率"),
    "nsys_timeline": (14, "Nsight：把热点放回执行时间线", "预热后第一次捕获调用 · 全部 kernel · 各面板同一时间尺度"),
}


def report_style(function):
    @wraps(function)
    def styled(*args, **kwargs):
        path = Path(__file__).parent / "assets/wqy-microhei.ttc"
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
        with plt.rc_context({
            "font.family": [family, "DejaVu Sans"], "font.size": 12,
            "axes.titlesize": 14, "axes.titleweight": "normal", "axes.titlepad": 38,
            "axes.labelsize": 12, "axes.labelpad": 10, "axes.labelcolor": MUTED,
            "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
            "xtick.labelsize": 11, "ytick.labelsize": 11,
            "xtick.major.size": 0, "ytick.major.size": 0,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.spines.left": False, "axes.edgecolor": "#D4DEE5",
            "axes.facecolor": "#FBFCFD", "figure.facecolor": "white",
            "axes.grid": False, "grid.color": "#E2E9EE", "grid.linewidth": .7,
            "grid.alpha": .85, "legend.frameon": False, "legend.fontsize": 10,
            "lines.linewidth": 2.2, "lines.markersize": 6,
            "axes.unicode_minus": False, "svg.fonttype": "path",
            "pdf.fonttype": 42, "ps.fonttype": 42,
            "figure.constrained_layout.h_pad": .18, "figure.constrained_layout.w_pad": .16,
            "figure.constrained_layout.wspace": .12,
        }):
            return function(*args, **kwargs)
    return styled


def panel(ax, title, *, grid="y", zero=True):
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis=grid)
    ax.set_axisbelow(True)
    if ax.yaxis.get_units() is None:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    if zero:
        ax.set_ylim(bottom=0)
    ax.margins(x=.07)


def legend(ax, *, ncol=2):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=ncol,
              borderaxespad=0, handlelength=2.2, columnspacing=1.4)


def bar_labels(ax, bars, *, suffix="", digits=1, color=INK, fontsize=11):
    labels = [f"{p.get_height():.{digits}f}{suffix}" for p in bars]
    ax.bar_label(bars, labels=labels, padding=6, fontsize=fontsize, color=color)
    ax.set_ylim(top=max(ax.get_ylim()[1], max(p.get_height() for p in bars)*1.2))


def peak_label(ax, x, y, text, *, offset=(-20, 18), color=INK):
    ax.annotate(text, xy=(x,y), xytext=offset, textcoords="offset points",
                ha="right" if offset[0]<0 else "left", va="bottom", fontsize=11,
                fontweight="normal", color=color,
                bbox={"boxstyle":"round,pad=.35", "fc":"white", "ec":"#DCE5EA", "alpha":.97},
                arrowprops={"arrowstyle":"-", "color":"#96A6B1", "lw":1})
