"""All paper figures for CANDOR, following plot_standard/ (Times New Roman, 9/10/8 pt, square nodes,
0.6 pt dark-gray arrows, RDBU_8 / CS_MODELS, Ours = #B2182B, strongest baseline = #2166AC).
Every number is read from run outputs; nothing is typed in by hand except the worked-example text in F1,
which is copied from runs/main_r20_s3/omni7b_ours_v4{.json,_pairs.json} (question 184) and the dirty DB.

usage: python paper_figs.py [out_dir]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as PlotPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_config as cfg  # noqa: E402

for f in glob.glob(os.path.join(os.environ.get('CANDOR_FONTS', 'fonts'), '*.ttf')):
    fm.fontManager.addfont(f)
assert 'times' in fm.findfont('Times New Roman').lower(), 'Times New Roman not found'

R = os.environ.get('CANDOR_RUNS', 'runs')
SCALE = os.environ.get('CANDOR_SCALE', 'runs/scale')
OUT = sys.argv[1] if len(sys.argv) > 1 else 'figs'
FS = cfg.FONT_SIZES
C = cfg.DIAGRAM_COLORS
OURS, BASE = cfg.RDBU_8[0], cfg.RDBU_8[7]
SINGLE_W = 3.5
DOUBLE_W = 180 / 25.4


def save(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        # 'standard' keeps the exact figsize (init_style sets savefig.bbox='tight', which trims the canvas)
        with plt.rc_context({'savefig.bbox': 'standard'}):
            fig.savefig(f'{OUT}/{name}.{ext}', pad_inches=0)
    plt.close(fig)
    print('wrote', name)


def summary(path):
    return json.load(open(path))['summary']


# ------------------------------------------------------------------ F1 overview (double column)
BOXED = []  # (text artist, x0, x1) for the fit check


def node(ax, x, y, w, h, title, lines, role='neutral', highlight=False):
    """title top-aligned at the same offset in every node; detail lines below it"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, zorder=2, **cfg.diagram_node_style(role, highlight=highlight)))
    ty = y + h - 0.16
    BOXED.append((ax.text(x + w / 2, ty, title, ha='center', va='center', fontsize=FS['body'], fontweight='bold',
                          zorder=3), x, x + w))
    for k, s in enumerate(lines):
        BOXED.append((ax.text(x + w / 2, ty - 0.185 - 0.135 * k, s, ha='center', va='center', fontsize=FS['note'],
                              color=C['secondary'], zorder=3), x, x + w))


def arrow(ax, pts, dependency=False):
    path = PlotPath(pts, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(pts) - 1))
    ax.add_patch(FancyArrowPatch(path=path, zorder=1, **cfg.diagram_arrow_style(dependency=dependency)))


def example(ax, x, y, w, h, lines):
    """trace block, left-aligned. Line styles: h = block title (9 pt bold), s = sub-heading (8 pt italic),
    a = answer line (8 pt, dark), n = detail (8 pt, gray)"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, zorder=2, **cfg.diagram_node_style('note')))
    sty = {'h': dict(fontsize=FS['body'], fontweight='bold', color=C['text']),
           's': dict(fontsize=FS['note'], fontstyle='italic', color=C['text']),
           'a': dict(fontsize=FS['note'], color=C['text']),
           'n': dict(fontsize=FS['note'], color=C['secondary'])}
    for k, (s, st) in enumerate(lines):
        yy = y + h - 0.13 - 0.138 * k - (0.02 if k else 0)
        BOXED.append((ax.text(x + 0.06, yy, s, ha='left', va='center', zorder=3, **sty[st]), x, x + w))


def dep_arrow(ax, x, y0, y1):
    """dependency arrow: dashed shaft, solid head of the standard size"""
    head = 0.07
    ax.plot([x, x], [y0, y1 - head], color=C['arrow'], lw=cfg.DIAGRAM_ARROW_LINEWIDTH, ls=(0, (3, 2)), zorder=1,
            solid_capstyle='butt', dash_capstyle='butt')
    arrow(ax, [(x, y1 - head), (x, y1)])


def fit_check(fig, ax):
    """every boxed text must stay 0.04 in inside its box"""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = 0
    for t, x0, x1 in BOXED:
        bb = t.get_window_extent()
        (a, _), (b, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if a < x0 + 0.04 or b > x1 - 0.04:
            print(f'  OVERFLOW {t.get_text()!r}: text {a:.2f}-{b:.2f} box {x0:.2f}-{x1:.2f}')
            bad += 1
    print('fit check:', 'ok' if not bad else f'{bad} overflow(s)')


def fig_overview():
    # verify the worked example against the run and the databases before drawing it
    import sqlite3
    run = {r['i']: r for r in json.load(open(f'{R}/main_r20_s3/omni7b_ours_v4.json'))}[184]
    ev = {r['i']: r for r in json.load(open(f'{R}/main_r20_s3/omni7b_ours_v4.eval.json'))['records']}[184]
    ev0 = {r['i']: r for r in json.load(open(f'{R}/main_r20_s3/omni7b_none.eval.json'))['records']}[184]
    pairs = [p for p in json.load(open(f'{R}/main_r20_s3/omni7b_ours_v4_pairs.json')) if p['i'] == 184]
    q = json.load(open(os.environ.get('CANDOR_DIRTY', 'data/bird_dirty') + '/main_r20_s3/dev.json'))[184]
    assert ev['correct'] and not ev0['correct'] and ev['affected']
    assert "IN ('F', 'FEMALE')" in run['pred_sql'] and "IN ('Junior', 'junior')" in run['pred_sql']
    assert "c.gender = 'F'" in run['orig_sql'] and "ca.type = 'Junior'" in run['orig_sql']
    pf = {(p['b'], p['a']): p['p'] for p in pairs}  # key (stored value asked about, option offered); see run_varfix.py pair_log
    assert len(pairs) == 4 and set(pf) == {('FEMALE', 'F'), ('MALE', 'F'), ('classic', 'Junior'), ('F', 'FEMALE')}
    assert pf[('FEMALE', 'F')] > 0.99 and pf[('F', 'FEMALE')] > 0.99 and pf[('MALE', 'F')] < 0.01
    assert pf[('classic', 'Junior')] < 0.01 and run['stats']['llm_calls'] == 4 and run['stats']['stat_rejects'] == 2
    assert run['stats'].get('group_sites', 0) == 0 and run['stats']['eq_sites'] == 2
    dirty = sqlite3.connect(os.environ.get('CANDOR_DIRTY', 'data/bird_dirty') + '/main_r20_s3/dev_databases/financial/financial.sqlite')
    clean = sqlite3.connect(os.environ.get('CANDOR_BIRD_DEV', 'data/bird/dev_20240627') + '/dev_databases/financial/financial.sqlite')
    a_orig = dirty.execute(run['orig_sql']).fetchall()[0][0]
    a_new = dirty.execute(run['pred_sql']).fetchall()[0][0]
    truth = clean.execute(q['SQL']).fetchall()[0][0]
    dg = dict(dirty.execute('select gender, count(*) from client group by 1').fetchall())
    dt = dict(dirty.execute('select type, count(*) from card group by 1').fetchall())
    assert (a_orig, a_new, truth) == (14, 71, 71) and len(dg) == 5 and len(dt) == 6
    assert sorted(dg) == ['F', 'FEMALE', 'M', 'MALE', 'm'] and 'gold' in dt and 'gold ' in dt
    V = __import__('varfix')
    assert V.short_codes('F', 'M') and V.short_codes('F', 'm')
    V.PRIOR_SIM[0] = 85
    assert V.freq_distinct(dt['Junior'], dt['gold'], a='Junior', b='gold')

    cfg.init_style(mode='diagram')
    W, H = DOUBLE_W, 3.96
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    gap = 0.13
    pad = gap / 2  # dashed CANDOR frame runs through the middle of the first gap
    widths = [0.95, 0.95, 0.95, 1.50, 0.95, 0.95]  # same-level nodes equal; the key node is wider
    span = sum(widths) + gap * (len(widths) - 1) + pad
    x0 = (W - span) / 2
    xs = [x0 + sum(widths[:k]) + gap * k for k in range(len(widths))]
    xend = xs[-1] + widths[-1]

    # vertical layout, bottom up
    ey, eh = 0.05, 0.13 + 0.138 * 7 + 0.13          # (b) trace blocks, up to 8 lines
    note_y = ey + eh + 0.17
    title_b = note_y + 0.17
    sep = title_b + 0.15
    dby, dbh = sep + 0.16, 0.27                      # database node
    ny, nh = dby + dbh + pad + 0.17, 0.86            # pipeline nodes
    top = ny + nh + 0.25                             # top of the CANDOR frame
    assert top < H - 0.29 - 0.09, (top, H)

    # (a) pipeline
    ax.text(x0, H - 0.12, '(a)  Pipeline', fontsize=FS['title'], fontweight='bold', va='center')
    ax.text(x0, H - 0.29, 'CANDOR rewrites the SQL of any text-to-SQL model. It reads the current database at query '
            'time and calls one instruction-tuned LLM.', fontsize=FS['note'], color=C['secondary'], va='center')
    specs = [
        ('Text-to-SQL', ['question → SQL,', 'any model'], 'neutral', False),
        ('1  Site analysis', ['selection sites,', 'group sites'], 'cool', False),
        ('2  Blocking', ['norm. keys, fuzzy,', 'small domains,', 'LLM expansion'], 'cool', False),
        ('3  Statistics + verifier', ['same class: auto-accepted,', 'statistics rules,', 'LLM selection with abstention,', 'bidirectional test'],
         'warm', True),
        ('4  Group sites', ['restricted domain,', 'own blocking,', 'leader clustering,', 'stage-3 checks'], 'cool', False),
        ('5  Rewrite', ['IN / NOT IN lists,', 'CASE for groups;', 'SQL runs as usual'], 'cool', False),
    ]
    for (t, ls, role, hl), x, w in zip(specs, xs, widths):
        node(ax, x, ny, w, nh, t, ls, role, hl)
    for k in range(len(xs) - 1):
        arrow(ax, [(xs[k] + widths[k], ny + nh / 2), (xs[k + 1], ny + nh / 2)])
    # the database: stages 2-5 read domains / frequencies (site analysis uses only the schema)
    dbx0, dbx1 = xs[2], xend
    ax.add_patch(FancyBboxPatch((dbx0, dby), dbx1 - dbx0, dbh, zorder=2, **cfg.diagram_node_style('history')))
    BOXED.append((ax.text((dbx0 + dbx1) / 2, dby + dbh / 2, 'Dirty database, read at query time: column domains '
                          'with value frequencies', ha='center', va='center', fontsize=FS['note'], zorder=3),
                  dbx0, dbx1))
    reads = [xs[k] + widths[k] / 2 for k in (2, 3, 4, 5)]
    for cx in reads:
        dep_arrow(ax, cx, dby + dbh, ny)
    # dashed CANDOR frame, with a gap where each read arrow passes its bottom edge
    fl, fr, fb = xs[1] - pad, xend + pad, ny - pad
    dash = dict(color=C['neutral_edge'], lw=cfg.DIAGRAM_NODE_LINEWIDTH, ls=(0, (3, 3)), zorder=1)
    ax.plot([fl, fl, fr, fr], [fb, top, top, fb], **dash)
    xs_b = [fl] + [v for cx in reads for v in (cx - 0.045, cx + 0.045)] + [fr]
    for a_, b_ in zip(xs_b[::2], xs_b[1::2]):
        ax.plot([a_, b_], [fb, fb], **dash)
    ax.text(fl + 0.06, ny + nh + 0.115, 'CANDOR', fontsize=FS['body'], fontweight='bold', va='center')

    ax.plot([x0, fr], [sep, sep], color='#D0D5DB', lw=0.6)

    # (b) worked example
    ax.text(x0, title_b, '(b)  Worked example', fontsize=FS['title'], fontweight='bold', va='center')
    note_b = ax.text(x0, note_y, 'Held-out seed 3, question 184: “How many female customers have a junior credit '
                     'card?” SQL from OmniSQL-7B; 4 calls to the Qwen2.5-7B verifier.',
                     fontsize=FS['note'], color=C['secondary'], va='center')
    BOXED.append((note_b, x0 - 0.04, W - 0.04))
    blocks = [
        [('Generated SQL', 'h'), ('… WHERE', 'n'), ("c.gender = 'F'", 'n'), ('AND', 'n'),
         ("ca.type = 'Junior'", 'n'), (f'Answer: {a_orig}', 'a'), (f'(truth: {truth})', 'n')],
        [('Sites', 'h'), ("client.gender = 'F'", 'n'), ("card.type = 'Junior'", 'n'), ('(aliases c, ca)', 'n')],
        [('Candidates', 'h'), ('client.gender', 's'), (f'{len(dg)} values, {len({V.norm(v) for v in dg})} classes', 'n'), ('card.type', 's'),
         (f'{len(dt)} values, {len({V.norm(v) for v in dt})} classes', 'n'), ('(small domains:', 'n'), ('all values)', 'n')],
        [('Decisions', 'h'), ("'junior': same class, accepted", 'n'), ("'M', 'm': short codes, rejected", 'n'),
         ("'gold': frequency prior, rejected", 'n'), ('LLM verifier', 's'),
         (f"'FEMALE' is 'F'?  $p$ = {pf[('FEMALE', 'F')]:.4f}", 'n'),
         (f"'F' is 'FEMALE'?  $p$ = {pf[('F', 'FEMALE')]:.4f}", 'n'), ("'MALE', 'classic': rejected", 'n')],
        [('Group sites', 'h'), ('none (skipped)', 'n')],
        [('Rewritten SQL', 'h'), ('… WHERE', 'n'), ('c.gender IN', 'n'), ("('F', 'FEMALE')", 'n'),
         ('AND', 'n'), ('ca.type IN', 'n'), ("('Junior', 'junior')", 'n'), (f'Answer: {a_new}', 'a')],
    ]
    assert max(len(b) for b in blocks) <= 8
    for b, x, w in zip(blocks, xs, widths):
        example(ax, x, ey, w, eh, b)
    for k in range(len(xs) - 1):
        arrow(ax, [(xs[k] + widths[k], ey + eh / 2), (xs[k + 1], ey + eh / 2)])
    fit_check(fig, ax)
    save(fig, 'fig_overview')






# ------------------------------------------------------------------ F2 cost
def fig_cost():
    runs = json.load(open(f'{R}/main_r20_s3/omni7b_ours_v4.json'))
    calls = np.array([r['stats'].get('llm_calls', 0) for r in runs])
    E = json.load(open(os.environ.get('CANDOR_DIRTY', 'data/bird_dirty') + '/eager_main_r20_s3_v4/eager_meta.json'))['stats']['llm_calls']
    n = len(calls)
    cum = np.cumsum(calls)
    rate = cum[-1] / n
    be = E / rate
    print(f'cost: n={n} total={cum[-1]} rate={rate:.2f} eager={E} break-even={be:.0f}')

    # log-log axes: the measured range (1..1,534 questions) gets most of the width; on linear axes it was ~13%
    cfg.init_style()
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6), layout='constrained')
    xmin, xmax = 10, 30000
    k0 = xmin - 1
    ax.plot(np.arange(xmin, n + 1), cum[k0:], color=OURS, lw=1.8, ls='-',
            label=f'CANDOR, measured ({n:,} held-out questions)', zorder=4)
    xe = np.array([n, xmax])
    ax.plot(xe, rate * xe, color=OURS, lw=1.2, ls=(0, (1.5, 1.5)),
            label=f'CANDOR, extrapolated at {rate:.2f} calls/question', zorder=3)
    ax.plot([xmin, xmax], [E, E], color=BASE, lw=1.4, ls='--', zorder=3,
            label=f'Eager cleaning, one-off ({E:,} calls)')
    ax.plot([n, n], [10, 2e5], color='#7F7F7F', lw=0.6, ls=':', zorder=2)
    ax.text(n * 0.9, 300, 'end of\nmeasurement', fontsize=FS['note'], color='#555555', va='center', ha='right',
            linespacing=0.95)
    ax.plot([be], [E], marker='o', ms=5, mfc='white', mec='#333333', mew=0.8, zorder=5, ls='none')
    ax.annotate(f'extrapolated break-even\n≈ {be / 1e3:.1f}k questions', xy=(be, E),
                xytext=(be, 2500), fontsize=FS['note'], va='center', ha='center', linespacing=0.95,
                arrowprops=dict(cfg.diagram_arrow_style(), shrinkA=2, shrinkB=3))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(10, 2e5)
    # plain 8-pt tick labels (10, 100, 1k, ...); mathtext 10^n renders its exponent at 5.6 pt, below the standard
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    kfmt = FuncFormatter(lambda v, _: f'{v / 1e3:g}k' if v >= 1e3 else f'{v:g}')
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(LogLocator(base=10))
        a_.set_major_formatter(kfmt)
        a_.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Questions answered (log scale)')
    ax.set_ylabel('Cumulative LLM calls (log scale)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, handlelength=2.4, borderaxespad=0.1,
              labelspacing=0.25)
    save(fig, 'fig_cost')


# ------------------------------------------------------------------ F3 dirty-row rate
def fig_rate():
    # held-out seed 4 at every rate (method frozen before seed 4 was built); gold SQL
    roots = [('rate_r05_s4', 5), ('rate_r10_s4', 10), ('main_r20_s4', 20), ('rate_r30_s4', 30)]
    meth = [  # key, label, color, marker, linestyle
        ('ours_v4', 'CANDOR', OURS, 'o', '-'),
        ('fuzzy', 'Fuzzy', BASE, 's', '-'),
        ('norm', 'Norm', cfg.CS_MODELS[2], '^', '--'),
        ('like', 'LIKE', cfg.CS_MODELS[3], 'v', '-.'),
        ('oracle', 'Oracle variant sets', cfg.CS_MODELS[6], 'D', ':'),
    ]
    cfg.init_style()
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.3), layout='constrained')
    xs = [x for _, x in roots]
    handles = {}
    for key, lab, col, mk, ls in meth:
        ys = [summary(f'{R}/gold_{rt}/gold_{key}.eval.json')['EX_aff'] for rt, _ in roots]
        print('rate', key, ys)
        # Norm: larger hollow triangle under the Fuzzy square, so a tie (r = 5%) shows both markers
        hollow = key in ('oracle', 'norm')
        handles[key], = ax.plot(xs, ys, color=col, marker=mk, ls=ls, lw=1.8 if key == 'ours_v4' else 1.2,
                                ms=4.5 if key == 'ours_v4' else (5.5 if key == 'norm' else 4),
                                mfc='white' if hollow else col, mew=1.0,
                                label=lab, zorder={'ours_v4': 5, 'fuzzy': 4}.get(key, 3))
    naff = [summary(f'{R}/gold_{rt}/gold_ours_v4.eval.json')['n_aff'] for rt, _ in roots]
    assert all(summary(f'{R}/gold_{rt}/gold_{m}.eval.json')['n_aff'] == n for rt, n in zip([r for r, _ in roots], naff)
               for m, *_ in meth)
    print('rate n_aff', naff)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x}%' for x in xs])
    ax.set_xlim(2.5, 32.5)
    ax.set_ylim(40, 100)
    ax.set_xlabel('Dirty-row rate $r$ per eligible attribute')
    ax.set_ylabel('EX on affected questions (%)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    order = ['ours_v4', 'norm', 'fuzzy', 'like', 'oracle']  # column-major: (CANDOR, Norm) (Fuzzy, LIKE) (Oracle)
    lg = ax.legend([handles[k] for k in order], [handles[k].get_label() for k in order], loc='lower left',
                   bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, handlelength=2.2, columnspacing=0.8,
                   borderaxespad=0.1, alignment='left',
                   title=f'Gold SQL, held-out seed 4; n = {min(naff)}–{max(naff)} per rate',
                   title_fontsize=FS['note'])
    lg.get_title().set_color('#555555')
    save(fig, 'fig_rate')


# ------------------------------------------------------------------ F4 scalability
def fig_scale():
    tags = [('main_r20_s3', 'x1'), ('scale_x10', 'x10'), ('scale_x100', 'x100')]
    meta = {t: json.load(open(f'{SCALE}/{t}_meta.json')) for t, _ in tags[1:]}
    tot_before = sum(a['distinct_before'] for d in meta['scale_x10']['dbs'].values() for a in d.values())
    tot = [tot_before] + [sum(a['distinct_after'] for d in meta[t]['dbs'].values() for a in d.values())
                          for t, _ in tags[1:]]
    nattr = sum(len(d) for d in meta['scale_x10']['dbs'].values())
    print('scale distinct totals', tot, 'attributes', nattr)

    def load(t, cold):
        return json.load(open(f'{SCALE}/{t}_cold{cold}.json'))
    def pct(t, c, q):  # same order statistic as the reported summaries (run_varfix_online.py)
        ss = sorted(r['sec'] for r in load(t, c))
        return ss[len(ss) // 2] if q == 50 else ss[int(.9 * len(ss))]
    p50 = {c: [pct(t, c, 50) for t, _ in tags] for c in (0, 1)}
    p90 = {c: [pct(t, c, 90) for t, _ in tags] for c in (0, 1)}
    warm = [load(t, 0) for t, _ in tags]
    llm = [np.mean([r['llm_s'] for r in w]) for w in warm]
    db = [np.mean([r['db_s'] for r in w]) for w in warm]
    other = [np.mean([r['sec'] - r['llm_s'] - r['db_s'] for r in w]) for w in warm]
    calls = [np.mean([r['calls'] for r in w]) for w in warm]
    nq = len(warm[0])
    print('p50', p50, 'p90', p90, 'llm', llm, 'db', db, 'other', other, 'calls', calls, 'n', nq)

    def fmt(v):
        return f'{v / 1e6:.2f}M' if v >= 1e6 else f'{v / 1e3:.1f}k' if v < 1e5 else f'{v / 1e3:.0f}k'
    # * = capped at 300k added values per attribute (explained in the caption)
    xl = ['×' + lab[1:] + ('*' if lab == 'x100' else '') + '\n' + fmt(v) for (_, lab), v in zip(tags, tot)]

    cfg.init_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(SINGLE_W, 2.45), layout='constrained')
    x = np.arange(3)
    # cold series drawn first and offset slightly in x, so the warm markers stay visible where they coincide
    off = 0.07
    a.plot(x + off, p90[1], color=OURS, marker='s', ls=':', lw=1.2, mfc='white', label='p90, cold', zorder=3)
    a.plot(x + off, p50[1], color=cfg.RDBU_8[1], marker='o', ls=':', lw=1.2, mfc='white', label='p50, cold', zorder=3)
    a.plot(x, p90[0], color=OURS, marker='s', ls='-', lw=1.4, label='p90, warm', zorder=4)
    a.plot(x, p50[0], color=cfg.RDBU_8[1], marker='o', ls='--', lw=1.4, label='p50, warm', zorder=4)
    hs, ls_ = a.get_legend_handles_labels()
    lorder = [2, 0, 3, 1]  # p90 warm, p90 cold, p50 warm, p50 cold
    a.set_ylim(0, 2.6)
    a.set_ylabel('Added latency per question (s)')
    a.set_title('(a) Latency', loc='left', pad=4)
    a.legend([hs[i] for i in lorder], [ls_[i] for i in lorder], loc='upper left', frameon=False, handlelength=2.0,
             borderaxespad=0.2, labelspacing=0.25)

    # db_s times only the domain reads; everything else CANDOR does (parsing, blocking, restriction queries) is "other"
    parts = [(llm, 'LLM calls', cfg.AAAI_PASTEL[0], '//'), (db, 'Domain reads', cfg.AAAI_PASTEL[6], '..'),
             (other, 'Other: parsing,\nblocking, restriction', cfg.AAAI_PASTEL[3], '\\\\')]
    bottom = np.zeros(3)
    for vals, lab, col, h in parts:
        b.bar(x, vals, 0.55, bottom=bottom, color=col, edgecolor='#333333', lw=0.6, hatch=h, label=lab, zorder=3)
        bottom += np.array(vals)
    for xi, top, c in zip(x, bottom, calls):
        b.text(xi, top + 0.03, f'{c:.1f}\ncalls', ha='center', va='bottom', fontsize=FS['note'], linespacing=0.95)
    b.set_ylim(0, 1.42)
    b.set_ylabel('Mean added time per question (s)')
    b.set_title('(b) Breakdown', loc='left', pad=4)
    hb, lb = b.get_legend_handles_labels()
    b.legend(hb[::-1], lb[::-1], loc='upper left', frameon=False, handlelength=1.4, borderaxespad=0.2,
             labelspacing=0.25, title='Warm runs', title_fontsize=FS['note'], alignment='left')  # same top-to-bottom order as the stack
    b.grid(False, axis='x')
    for s in (a, b):
        s.set_xticks(x)
        s.set_xticklabels(xl)
        s.set_xlim(-0.5, 2.5)
        s.spines[['top', 'right']].set_visible(False)
        s.set_axisbelow(True)
    fig.supxlabel(f'Domain scale and distinct values ({nattr} attributes, {nq} questions)', fontsize=FS['body'])
    save(fig, 'fig_scale')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[2:] or ['overview', 'cost', 'rate', 'scale']
    for w in which:
        globals()[f'fig_{w}']()
