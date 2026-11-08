"""Render the paper's tables directly from run outputs (no hand-typed numbers).
Writes paper/tables/{main,robust,ablation}.tex. CANDOR numbers are v4 (final method); Spider-Dirty is excluded. Missing cells are rendered as '--'."""
import json, os, glob, statistics
R = os.environ.get('CANDOR_RUNS', 'runs')
P = os.environ.get('CANDOR_TABLES', 'paper/tables')
os.makedirs(P, exist_ok=True)


def exact(d):
    """summary recomputed from the per-question records at full precision (the stored summary is rounded to 2 decimals,
    and rounding that again to 1 decimal can go the wrong way, e.g. 46.154 -> 46.15 -> 46.1)"""
    s = dict(d['summary'])
    recs = d.get('records')
    if recs:
        aff = [r for r in recs if r.get('affected')]
        un = [r for r in recs if not r.get('affected')]
        s['EX'] = 100.0 * sum(r['correct'] for r in recs) / len(recs)
        if aff:
            s['EX_aff'] = 100.0 * sum(r['correct'] for r in aff) / len(aff)
        if un:
            s['EX_unaff'] = 100.0 * sum(r['correct'] for r in un) / len(un)
    return s


def summ(path):
    f = path + '.eval.json'
    return exact(json.load(open(f))) if os.path.exists(f) else None


def meta(path):
    f = path + '.meta.json'
    return json.load(open(f)) if os.path.exists(f) else None


def runs_json(path):
    f = path + '.json'
    return json.load(open(f)) if os.path.exists(f) else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def fmt(v, bold=False):
    if v is None:
        return '--'
    s = f'{v:.1f}'
    return f'\\textbf{{{s}}}' if bold else s


def cell_triplet(roots, tag, name, prefix=R):
    """mean over roots of (EX, aff, unaff) + count of seeds found"""
    ex, af, un = [], [], []
    for rt in roots:
        s = summ(f'{prefix}/{rt}/{tag}_{name}')
        if s:
            ex.append(s['EX']); af.append(s['EX_aff']); un.append(s['EX_unaff'])
    return mean(ex), mean(af), mean(un)


def calls_per_q(roots, tag, name):
    vals = []
    for rt in roots:
        j = runs_json(f'{R}/{rt}/{tag}_{name}')
        if j:
            vals.append(sum(r['stats'].get('llm_calls', 0) for r in j) / len(j))
    return mean(vals)[0]


METHODS = [('No rewrite', 'none'), ('LIKE substring', 'like'), ('Norm', 'norm'), ('Fuzzy', 'fuzzy'),
           ('LLM-Repair (7B)', 'repair_q7'), ('LLM-Repair (32B)', 'repair_q32'), ('Eager cleaning (7B)', 'eagerv4_q7'),
           ('\\quad $+$ query-time Norm', 'eagerv4_norm_q7'),
           ('\\sys{} (7B verifier)', 'ours_v4'), ('\\sys{} (32B verifier)', 'ours_v4_q32'),
           ('Oracle variant sets', 'oracle')]
TAGNAME = {'omni7b': 'OmniSQL-7B', 'qwen7b': 'Qwen2.5-7B', 'qwen32': 'Qwen2.5-32B', 'omni14b': 'OmniSQL-14B'}


def main_table(roots, tags, clean_root, out, caption, label, remap=None, star=True):
    remap = remap or {}
    rows = []
    best = {}
    data = {}
    for lab, m in METHODS:
        for t in tags:
            (ex, n), (af, _), (un, _) = cell_triplet(roots, t, remap.get(m, m))
            data[(m, t)] = (ex, af, un, n)
    for t in tags:
        for k in range(3):
            vals = [data[(m, t)][k] for _, m in METHODS if m != 'oracle' and data[(m, t)][k] is not None]
            best[(t, k)] = max(vals) if vals else None
    for lab, m in METHODS:
        cells = []
        note = ''
        for t in tags:
            ex, af, un, n = data[(m, t)]
            if n and n < len(roots):
                note = '$^\\dagger$'
            for k, v in enumerate((ex, af, un)):
                cells.append(fmt(v, bold=(m != 'oracle' and v is not None and best[(t, k)] is not None and abs(v - best[(t, k)]) < 1e-9)))
        for t in tags:
            cs = summ(f'{R}/{clean_root}/{t}_{remap.get(m, m)}') if m != 'oracle' and not m.startswith('eager') else None
            cells.append(fmt(cs['EX'] if cs else None))
        if all(c == '--' for c in cells):
            continue
        rows.append(f'{lab}{note} & ' + ' & '.join(cells) + ' \\\\')
        if m in ('fuzzy', 'eagerv4_norm_q7'):
            rows.append('\\midrule')
    tg = TAGNAME
    k = len(tags)
    head1 = ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{tg[t]} (dirty)}}' for t in tags) + f' & \\multicolumn{{{k}}}{{c}}{{Clean EX}}\\\\'
    cm = ' '.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(k)) + f' \\cmidrule(lr){{{2 + 3 * k}-{1 + 4 * k}}}'
    head2 = 'Method & ' + ' & '.join(['EX & Aff. & Unaff.'] * k) + ' & ' + ' & '.join(tg[t].replace('Qwen2.5-', 'Q-').replace('OmniSQL-', 'O-') for t in tags) + '\\\\'
    env = 'table*' if star else 'table'
    tex = [f'\\begin{{{env}}}[t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\small',
           '\\setlength{\\tabcolsep}{3.5pt}' if star else '\\setlength{\\tabcolsep}{2.5pt}',
           '' if star else '\\resizebox{\\columnwidth}{!}{%',
           '\\begin{tabular}{l' + 'rrr' * k + 'r' * k + '}', '\\toprule', head1, cm, head2, '\\midrule'] + rows + \
          ['\\bottomrule', '\\end{tabular}' + ('' if star else '}'), f'\\end{{{env}}}']
    open(f'{P}/{out}', 'w').write('\n'.join(tex) + '\n')
    return data


def ablation_table(root='main_r20_s1', tag='omni7b'):
    """leave-one-out from the final method (v4) on the validation seed; each row changes one thing"""
    rows = [('\\sys{} (full)', 'v4'),
            ('\\multicolumn{7}{l}{\\emph{(a) verifier formulation}}', None),
            ('Pairwise yes/no', 'v4abl_pair'),
            ('Selection w/o bidirectional test', 'v4abl_nomutual'),
            ('\\sys{} $+$ in-domain distractors', 'v4abl_dist'),
            ('\\multicolumn{7}{l}{\\emph{(b) components removed}}', None),
            ('$-$ statistics rules', 'v4abl_nostats'),
            ('$-$ similarity exception of the frequency prior', 'ours_v3'),
            ('$-$ LLM query expansion', 'v4abl_noexpand'),
            ('$-$ wrapped sites', 'v4abl_nowrapped'),
            ('$-$ small-domain enumeration', 'v4abl_smalldom0'),
            ('$-$ query-driven restriction', 'v4abl_norestrict'),
            ('\\multicolumn{7}{l}{\\emph{(c) sensitivity}}', None),
            ('$\\tau=0.3$ / $0.7$ / $0.9$', 'TAU'),
            ('Verifier Qwen3-4B', 'v4abl_q4'),
            ('Verifier Qwen2.5-32B-AWQ', 'v4abl_q32')]
    out = []
    for lab, n in rows:
        if n is None:
            out.append(lab + ' \\\\')
            continue
        if n == 'TAU':
            ss = [summ(f'{R}/{root}/{tag}_v4abl_tau{t}') for t in ('03', '07', '09')]
            if all(ss):
                out.append(f"{lab} & " + ' / '.join(f"{x['EX']:.1f}" for x in ss) + ' & & & & & \\\\')
            continue
        s = summ(f'{R}/{root}/{tag}_{n}')
        mt = meta(f'{R}/{root}/{tag}_{n}')
        if not s:
            continue
        tot = mt['totals'] if mt else {}
        tp, fp, fn = tot.get('var_tp'), tot.get('var_fp'), tot.get('var_fn')
        pr = tp / (tp + fp) if tp is not None and tp + fp else None
        rc = tp / (tp + fn) if tp is not None and tp + fn else None
        cq = tot.get('llm_calls', 0) / 1534 if mt else None
        gd = tot.get('group_dom') if mt else None
        out.append(f"{lab} & {fmt(s['EX'])} & {fmt(s['EX_aff'])} & {fmt(s['EX_unaff'])} & "
                   f"{'--' if pr is None else f'{pr:.2f}'} & {'--' if rc is None else f'{rc:.2f}'} & "
                   f"{'--' if cq is None else f'{cq:.1f}'} \\\\")
    tex = ['\\begin{table}[t]', '\\centering', '\\caption{Component analysis on the validation seed of \\bench{} (OmniSQL-7B, '
           'Qwen2.5-7B verifier unless stated). P/R: precision and recall of the variants accepted at selection sites against '
           'the ground-truth classes. Calls: LLM calls per question.}',
           '\\label{tab:ablation}', '\\small', '\\setlength{\\tabcolsep}{3.2pt}', '\\resizebox{\\columnwidth}{!}{%',
           '\\begin{tabular}{lrrrrrr}', '\\toprule',
           'Variant & EX & Aff. & Unaff. & P & R & Calls\\\\', '\\midrule'] + out + ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    open(f'{P}/ablation.tex', 'w').write('\n'.join(tex) + '\n')


def robust_table():
    """gold-SQL (generator-free) robustness: affected-subset EX per variant type and per dirty-row rate"""
    # held-out seed 4 throughout (method frozen before seed 4 was built); the dirty-row-rate sweep is Fig. fig_rate.
    # The seed-0 (development) version of this table is kept in STATUS.md for the record only.
    cfg = [('\\multicolumn{9}{l}{\\emph{one variant type} ($r=0.2$)}', None),
           ('Case', 'type_case_s4'), ('Punctuation / spacing', 'type_punct_s4'), ('Typo', 'type_typo_s4'),
           ('Abbreviation', 'type_abbrev_s4'), ('Code / alias', 'type_alias_s4'),
           ('\\multicolumn{9}{l}{\\emph{all types} ($r=0.2$)}', None),
           ('All types', 'main_r20_s4')]
    ms = ['norm', 'like', 'fuzzy', 'ours_v4', 'oracle']
    out = []
    for lab, rt in cfg:
        if rt is None:
            out.append(lab + ' \\\\')
            continue
        ss = {m: summ(f'{R}/gold_{rt}/gold_{m}') for m in ms}
        if not ss['norm']:
            continue
        n = ss['norm']['n_aff']
        vals = [ss[m]['EX_aff'] if ss[m] else None for m in ms]
        best = max(v for m, v in zip(ms, vals) if m != 'oracle' and v is not None)
        cells = [fmt(v, bold=(m != 'oracle' and v is not None and abs(v - best) < 1e-9)) for m, v in zip(ms, vals)]
        un = [fmt(ss[m]['EX_unaff']) if ss[m] else '--' for m in ('fuzzy', 'ours_v4')]
        out.append(f'{lab} & {n} & ' + ' & '.join(cells) + ' & ' + ' & '.join(un) + ' \\\\')
    tex = ['\\begin{table}[t]', '\\centering',
           '\\caption{Robustness with gold SQL (no generator errors) on seed 4: EX (\\%) on the affected '
           'questions per variant type, and EX on the unaffected questions for Fuzzy and \\sys{}.}',
           '\\label{tab:robust}', '\\small', '\\setlength{\\tabcolsep}{2.6pt}', '\\resizebox{\\columnwidth}{!}{%',
           '\\begin{tabular}{lrrrrrrrr}', '\\toprule',
           ' & & \\multicolumn{5}{c}{Affected EX} & \\multicolumn{2}{c}{Unaff.\\ EX}\\\\',
           '\\cmidrule(lr){3-7} \\cmidrule(lr){8-9}',
           'Configuration & \\#Aff & Norm & LIKE & Fuzzy & \\sys{} & Oracle & Fuzzy & \\sys{}\\\\', '\\midrule'] + out + \
          ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    open(f'{P}/robust.tex', 'w').write('\n'.join(tex) + '\n')


def multiseed_table(roots=('main_r20_s2', 'main_r20_s3', 'main_r20_s4')):
    """mean +- sd of EX over the held-out seeds, every generator"""
    tags = ['omni7b', 'qwen7b', 'qwen32', 'omni14b', 'llama8']
    ms = [('No rewrite', 'none'), ('LIKE substring', 'like'), ('Norm', 'norm'), ('Fuzzy', 'fuzzy'),
          ('\\sys{}', 'ours_v4'), ('Oracle variant sets', 'oracle')]
    out = []
    for lab, m in ms:
        cells = []
        for t in tags:
            xs = [summ(f'{R}/{rt}/{t}_{m}') for rt in roots]
            xs = [x['EX'] for x in xs if x]
            if len(xs) == len(roots):
                cells.append(f'{statistics.mean(xs):.1f}$\\pm${statistics.stdev(xs):.1f}')
            else:
                cells.append('--')
        out.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
        if m in ('fuzzy', 'ours_v4'):
            out.append('\\midrule')
    tex = ['\\begin{table*}[t]', '\\centering',
           '\\caption{EX (\\%) on \\bench{} ($r=0.2$), mean $\\pm$ standard deviation over seeds 2, 3 and 4.}',
           '\\label{tab:seeds}', '\\small', '\\begin{tabular}{lrrrrr}', '\\toprule',
           'Method & ' + ' & '.join(TAGNAME.get(t, 'Llama-3.1-8B') for t in tags) + '\\\\', '\\midrule'] + out + \
          ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    open(f'{P}/seeds.tex', 'w').write('\n'.join(tex) + '\n')


if __name__ == '__main__':
    multiseed_table()
    robust_table()
    main_table(['main_r20_s3'], ['omni7b', 'qwen7b', 'qwen32'], 'clean', 'main.tex',
               'Execution accuracy (\\%) on \\bench{} (seed 3, $r=0.2$) under clean-answer semantics: all 1{,}534 '
               'questions (EX), the 426 affected questions (Aff.) and the unaffected ones (Unaff.). Clean EX: the same method '
               'on the original BIRD databases. Bold: best non-oracle. --: not run.',
               'tab:main')
    ablation_table()
    print('tables written to', P)
