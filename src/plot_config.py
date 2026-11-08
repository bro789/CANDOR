# 文件名: plot_config.py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# 1. 专属色卡库 (Dictionaries & Lists)
# =========================================================

# 【色卡 A】(用于：热力图、双模型对比、消融实验柱状图)
RDBU_8 = [
    '#B2182B',  # [0] 深绯红 (固定代表 Ours!)
    '#D6604D',  # [1] 砖红   (Ours 的变体或次重点)
    '#F4A582',  # [2] 蜜桃橙
    '#FDDBC7',  # [3] 浅杏粉
    '#D1E5F0',  # [4] 雾蓝灰
    '#92C5DE',  # [5] 天青蓝
    '#4393C3',  # [6] 湖蓝
    '#2166AC'   # [7] 深海蓝 (固定代表最强 Baseline!)
]

# 【色卡 B】CS多模型对比色卡 (用于：5~7条线的折线图，解决图1颜色不够分的问题)
CS_MODELS = [
    '#B2182B',  # 1. Ours (深绯红，最醒目)
    '#2166AC',  # 2. Baseline 1 (深海蓝)
    '#F78B44',  # 3. Baseline 2 (活力橙)
    '#31A354',  # 4. Baseline 3 (森林绿)
    '#756BB1',  # 5. Baseline 4 (学术紫)
    '#4393C3',  # 6. Baseline 5 (湖蓝)
    '#7F7F7F'   # 7. 早期经典方法 (中性灰，用来垫底)
]

# 【色卡 C】图4原版：AAAI 浅色粉彩 (用于：箱线图内部填充、带黑框的柱状图)
AAAI_PASTEL = [
    '#FBC7C7', '#FBD7BB', '#EDF4E1', '#CAE7BC', 
    '#E1F4F9', '#92CDDC', '#90AADB', '#CFC0FB', '#B79FEB'
]

# 自动生成图1的热力图渐变色带 (可以直接传给 cmap 参数)
CMAP_RDBU = LinearSegmentedColormap.from_list("My_RdBu", RDBU_8[::-1], N=256)

# 固定标记符号顺序 (画折线图时，不同模型不仅颜色不同，形状也要不同，防止黑白打印分不清)
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*']

# 同一篇论文的框架图与统计图使用同一套有衬线字体。
# Times New Roman 是个人默认风格；会议或期刊有明确要求时，以其模板为准。
SERIF_FONTS = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif']
FONT_SIZES = {
    'body': 9,
    'title': 10,
    'note': 8,
    'tick': 8,
    'legend': 8,
}

# 框架图使用低饱和填充色；统计图继续使用上方已有的三套色卡。
# 颜色只承担图中实际定义的语义，不为新论文自动赋予正负或优劣含义。
DIAGRAM_COLORS = {
    'text': '#222222',
    'secondary': '#555555',
    'arrow': '#555555',
    'neutral_fill': '#F4F7FA',
    'neutral_edge': '#9BA5B0',
    'warm_fill': '#F8E8D5',
    'warm_edge': '#B88955',
    'cool_fill': '#E7EFF7',
    'cool_edge': '#7899B8',
    'history_fill': '#EEE9F5',
    'history_edge': '#9B86B7',
    'model_fill': '#F1F4F7',
    'model_edge': '#7899B8',
    'note_fill': '#FAFBFD',
    'note_edge': '#9BA5B0',
}

DIAGRAM_NODE_LINEWIDTH = 0.6
DIAGRAM_TARGET_LINEWIDTH = 0.9
DIAGRAM_ARROW_LINEWIDTH = 0.6
DIAGRAM_ARROW_SCALE = 8


def diagram_node_style(role='neutral', *, highlight=False):
    """供 FancyBboxPatch 使用的直角节点样式，宽高由调用者设置。"""
    return {
        'boxstyle': 'square,pad=0',
        'facecolor': DIAGRAM_COLORS[f'{role}_fill'],
        'edgecolor': DIAGRAM_COLORS['text'] if highlight else DIAGRAM_COLORS[f'{role}_edge'],
        'linewidth': DIAGRAM_TARGET_LINEWIDTH if highlight else DIAGRAM_NODE_LINEWIDTH,
        'joinstyle': 'miter',
    }


def diagram_arrow_style(*, dependency=False):
    """供 FancyArrowPatch 或 annotate(arrowprops=...) 使用；实线表示主流程。"""
    return {
        'arrowstyle': '->',
        'linewidth': DIAGRAM_ARROW_LINEWIDTH,
        'mutation_scale': DIAGRAM_ARROW_SCALE,
        'color': DIAGRAM_COLORS['arrow'],
        'linestyle': '--' if dependency else '-',
        'shrinkA': 0,
        'shrinkB': 0,
        'joinstyle': 'miter',
        'capstyle': 'butt',
    }

# =========================================================
# 2. 论文标准画布尺寸 (单位：英寸 inches)
# =========================================================
FIG_SIZE = {
    'single': (3.5, 2.6),       # 常用单栏尺寸；最终按论文版心调整
    'single_square': (3.5, 3.2),# 单栏偏方图 (适合热力图、混淆矩阵、t-SNE)
    'double': (7.2, 2.8),       # 常用双栏尺寸 (适合并排 2~3 个子图)
    'diagram_double': (180 / 25.4, 116.25 / 25.4),  # 已选 Fig.1 的物理尺寸
}

# =========================================================
# 3. 全局样式一键初始化函数
# =========================================================
def init_style(mode='line'):
    """
    统计图调用 init_style()；框架图调用 init_style(mode='diagram')。
    字号以最终排版尺寸为准；这里的 8–10 pt 是个人默认值，不是会议通用规定。
    """
    plt.rcParams.update({
        # 默认颜色循环
        'axes.prop_cycle': mpl.cycler(color=CS_MODELS),
        
        # 框架图、直方图、柱状图、折线图、热力图统一使用有衬线字体。
        'font.family': 'serif',
        'font.serif': SERIF_FONTS,
        'mathtext.fontset': 'stix',  # 与 Times 风格协调的衬线数学字体
        'font.size': FONT_SIZES['body'],
        'text.color': DIAGRAM_COLORS['text'],
        'axes.titlesize': FONT_SIZES['title'],
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_SIZES['body'],
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'],
        'legend.fancybox': False,
        'legend.framealpha': 1.0,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        
        # 线条与边框
        'lines.linewidth': 1.5,
        'lines.markersize': 4.5,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',  # 边框用深灰，比纯黑更有质感
        'axes.labelcolor': DIAGRAM_COLORS['text'],
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.direction': 'in',      # 刻度线朝内 (学术风标配)
        'ytick.direction': 'in',
        
        # 网格线 (默认开启浅灰虚线网格)
        'axes.grid': mode != 'diagram',
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        
        # 高清导出与字体嵌入 (极其重要！防止投稿时 PDF 字体报错)
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',  # SVG 保留可编辑文字，接收方需要安装相应字体
    })
