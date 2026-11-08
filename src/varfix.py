"""Query-time variant-aware SQL rewriting for text-to-SQL over dirty databases.

Given a generated SQL query q over database D:
  1. analyze q: find value-sensitive sites
       EQ   : col = 'c' | col != 'c' | col IN ('a','b') | col LIKE 'c' (no wildcard)
       GROUP: GROUP BY col, SELECT DISTINCT col, COUNT(DISTINCT col)
  2. for each EQ site: candidate variants of c from dom(col) (blocking), then LLM verification
  3. for each GROUP site: block the (query-restricted) domain, LLM partitions blocks into same-value groups
  4. rewrite: col = 'c' -> col IN (variants);  col -> CASE WHEN col IN (...) THEN canon ... ELSE col END

Modes (matcher):
  ours   : blocking + LLM verification (auto-accept normalization-identical variants if auto_norm)
  norm   : variants = values with identical normalization key (lower, strip non-alnum); no LLM
  fuzzy  : variants = all blocking candidates with WRatio >= fuzzy_tau; no LLM (direct fuzzy-matching transplant)
  oracle : variants from the ground-truth canonical map (upper bound for rewriting a given SQL)
"""
import json, re, sqlite3, collections, os, time
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from rapidfuzz import fuzz, process

ABBR = {
    'street': 'st', 'avenue': 'ave', 'road': 'rd', 'boulevard': 'blvd', 'drive': 'dr', 'county': 'co',
    'company': 'co', 'corporation': 'corp', 'incorporated': 'inc', 'limited': 'ltd', 'saint': 'st',
    'mount': 'mt', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'university': 'univ',
    'department': 'dept', 'international': 'intl', 'and': '&', 'elementary': 'elem', 'school': 'sch',
    'district': 'dist', 'junior': 'jr', 'senior': 'sr', 'center': 'ctr', 'centre': 'ctr', 'association': 'assn',
    'national': 'natl', 'number': 'no', 'fort': 'ft', 'academy': 'acad', 'community': 'comm',
    'technology': 'tech', 'professional': 'prof', 'management': 'mgmt', 'government': 'govt',
    'administration': 'admin', 'committee': 'cmte', 'meeting': 'mtg', 'general': 'gen', 'special': 'spec',
    'education': 'ed', 'services': 'svcs', 'charter': 'chtr', 'unified': 'unif', 'circuit': 'circ',
}


def norm(s):
    return re.sub(r'[^0-9a-zÀ-￿]', '', s.lower())


def abbr_norm(s):
    toks = re.findall(r'[0-9a-zÀ-￿&]+', s.lower())
    return ''.join(ABBR.get(t, t) for t in toks)


class Schema:
    def __init__(self, db_info):
        self.cols = collections.defaultdict(dict)  # table_lower -> col_lower -> (table, col, type)
        tn = db_info['table_names_original']
        for (ti, c), ty in zip(db_info['column_names_original'], db_info['column_types']):
            if ti >= 0:
                self.cols[tn[ti].lower()][c.lower()] = (tn[ti], c, ty)
        self.tables = {t.lower(): t for t in tn}

    def find(self, table, col):
        return self.cols.get(table.lower(), {}).get(col.lower())


DB_T = [0.0]  # cumulative seconds spent in domain scans (cache misses); used only for timing reports


class DB:
    """domain access with caching"""

    def __init__(self, path):
        self.path = path
        self.con = sqlite3.connect(f'file:{path}?mode=ro', uri=True, check_same_thread=False)
        self.con.text_factory = lambda b: b.decode(errors='ignore')
        self.cache = {}

    def domain(self, table, col):
        k = (table, col)
        if k not in self.cache:
            t_ = time.time()
            try:
                rows = self.con.execute(
                    f'SELECT `{col}`, COUNT(*) FROM `{table}` WHERE typeof(`{col}`)="text" GROUP BY `{col}`').fetchall()
            except Exception:
                rows = []
            self.cache[k] = rows
            DB_T[0] += time.time() - t_
        return self.cache[k]

    def norm_index(self, table, col):
        k = ('__norm', table, col)
        if k not in self.cache:
            idx = collections.defaultdict(list)
            for v, _ in self.domain(table, col):
                idx[norm(v) or v].append(v)
            self.cache[k] = idx
        return self.cache[k]

    def query(self, sql, timeout=20):
        t0 = time.time()
        self.con.set_progress_handler(lambda: 1 if time.time() - t0 > timeout else 0, 10000)
        try:
            return self.con.execute(sql).fetchall()
        except Exception:
            return None
        finally:
            self.con.set_progress_handler(None, 0)


# ---------------------------------------------------------------- analysis
def _resolve(col, scope, schema):
    """map a Column node to (table, column) in the schema, or None"""
    name = col.name
    q = col.table
    if q:
        src = scope.sources.get(q)
        if isinstance(src, exp.Table):
            f = schema.find(src.name, name)
            return (f[0], f[1]) if f else None
        if q.lower() in schema.tables:
            f = schema.find(q, name)
            return (f[0], f[1]) if f else None
        return None
    hits = []
    for alias, src in scope.sources.items():
        if isinstance(src, exp.Table):
            f = schema.find(src.name, name)
            if f:
                hits.append((f[0], f[1]))
    hits = list(dict.fromkeys(hits))
    return hits[0] if len(hits) == 1 else None


def _str_lit(node, scope, schema):
    """string literal value of node, also accepting double-quoted identifiers that are not columns"""
    if isinstance(node, exp.Literal) and node.is_string:
        return node.this
    if isinstance(node, exp.Column) and not node.table and node.this.quoted and _resolve(node, scope, schema) is None:
        return node.name
    return None


WRAPPED = [False]  # set by the driver (--wrapped)


def analyze(sql, schema):
    """returns (tree, eq_sites, group_sites) or (None, [], []) if unparsable"""
    try:
        tree = sqlglot.parse_one(sql, read='sqlite')
    except Exception:
        return None, [], []
    eq_sites, group_sites = [], []
    try:
        scopes = list(traverse_scope(tree))
    except Exception:
        return None, [], []
    for scope in scopes:
        sel = scope.expression
        if not isinstance(sel, exp.Select):
            continue
        for node in list(sel.find_all(exp.EQ, exp.NEQ, exp.In, exp.Like)):
            # only nodes belonging to this scope (not nested subqueries)
            if node.find_ancestor(exp.Select) is not sel:
                continue
            if isinstance(node, exp.In):
                colnode = node.this
                if not isinstance(colnode, exp.Column) or node.args.get('query'):
                    continue
                consts = [_str_lit(e, scope, schema) for e in node.expressions]
                if not consts or any(c is None for c in consts):
                    continue
                kind = 'in'
            else:
                l, r = node.left, node.right
                if WRAPPED[0]:  # v3: LOWER(A) / UPPER(A) / TRIM(A) compared with a literal -> site on A
                    if isinstance(l, (exp.Lower, exp.Upper, exp.Trim)) and isinstance(l.this, exp.Column):
                        l = l.this
                    if isinstance(r, (exp.Lower, exp.Upper, exp.Trim)) and isinstance(r.this, exp.Column):
                        r = r.this
                if isinstance(l, exp.Column) and _str_lit(r, scope, schema) is not None and _resolve(l, scope, schema):
                    colnode, c = l, _str_lit(r, scope, schema)
                elif isinstance(r, exp.Column) and _str_lit(l, scope, schema) is not None and _resolve(r, scope, schema):
                    colnode, c = r, _str_lit(l, scope, schema)
                else:
                    continue
                if isinstance(node, exp.Like):
                    if '%' in c or '_' in c:
                        continue
                    kind = 'like'
                else:
                    kind = 'eq' if isinstance(node, exp.EQ) else 'neq'
                consts = [c]
            tc = _resolve(colnode, scope, schema)
            if not tc:
                continue
            eq_sites.append({'kind': kind, 'node': node, 'col': colnode, 'tc': tc, 'consts': consts, 'scope': scope})
        # selects feeding a membership / comparison predicate keep raw values (outer side is not canonicalized)
        if not _under(sel, None, (exp.In, exp.Exists, exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            gcols = []
            g = sel.args.get('group')
            if g:
                gcols += [e for e in g.expressions if isinstance(e, exp.Column)]
            if sel.args.get('distinct'):
                gcols += [e.unalias() if isinstance(e, exp.Alias) else e for e in sel.expressions]
                gcols = [e for e in gcols if isinstance(e, exp.Column)]
            for cnt in sel.find_all(exp.Count):
                d = cnt.this
                if isinstance(d, exp.Distinct) and cnt.find_ancestor(exp.Select) is sel:
                    gcols += [e for e in d.expressions if isinstance(e, exp.Column)]
            seen = set()
            for gc in gcols:
                tc = _resolve(gc, scope, schema)
                if tc and tc not in seen:
                    seen.add(tc)
                    group_sites.append({'kind': 'group', 'col': gc, 'tc': tc, 'scope': scope, 'select': sel})
    return tree, eq_sites, group_sites


# ---------------------------------------------------------------- candidates
def eq_candidates(c, dom, K=20, tau=70):
    """blocking: normalization / abbreviation identical values + fuzzy (WRatio) neighbours. returns [(v, score, tier)]"""
    nc, ac = norm(c), abbr_norm(c)
    out = {}
    for v, cnt in dom:
        if v == c:
            continue
        if norm(v) == nc and nc:
            out[v] = (100.0, 'norm')
        elif abbr_norm(v) == ac and ac:
            out[v] = (99.0, 'abbr')
    choices = [v for v, _ in dom if v != c and v not in out]
    if choices:
        for v, s, _ in process.extract(c, choices, scorer=fuzz.WRatio, limit=K, score_cutoff=tau, processor=str.lower):
            out[v] = (s, 'fuzzy')
    res = sorted(((v, s, t) for v, (s, t) in out.items()), key=lambda x: -x[1])
    return res[:K + 10]


def group_blocks(vals, tau=88, max_pairs_n=6000):
    """partition values into candidate blocks (union-find over norm/abbr equality and fuzzy pairs)."""
    parent = {v: v for v in vals}

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def u(a, b):
        ra, rb = f(a), f(b)
        if ra != rb:
            parent[ra] = rb

    by = collections.defaultdict(list)
    for v in vals:
        by[('n', norm(v))].append(v)
        by[('a', abbr_norm(v))].append(v)
    for k, vs in by.items():
        if k[1]:
            for x in vs[1:]:
                u(vs[0], x)
    if 1 < len(vals) <= max_pairs_n:
        low = [v.lower() for v in vals]
        import numpy as np
        from rapidfuzz.process import cdist
        M = cdist(low, low, scorer=fuzz.ratio, score_cutoff=tau, workers=16, dtype=np.uint8)
        ii, jj = np.nonzero(M)
        for i, j in zip(ii, jj):
            if i < j:
                u(vals[i], vals[j])
    comp = collections.defaultdict(list)
    for v in vals:
        comp[f(v)].append(v)
    return [sorted(c) for c in comp.values() if len(c) > 1]


# ---------------------------------------------------------------- prompts
def top_values(dom, k=10):
    return [v for v, _ in sorted(dom, key=lambda x: -x[1])[:k]]


IDLIKE = re.compile(r'^(?=.*\d)[A-Za-z0-9_\-:./]+$')


def is_key_or_idlike(schema_info, tc, dom):
    """columns we never canonicalize: declared PK/FK, or identifier-like value sets (no spaces, contain digits)"""
    if tc in schema_info:
        return True
    if not dom:
        return False
    vals = [v for v, _ in dom[:2000]]
    return sum(1 for v in vals if IDLIKE.match(v)) >= 0.8 * len(vals)


def key_columns(db_info):
    tn = db_info['table_names_original']
    cols = db_info['column_names_original']
    idx = set()
    for pk in db_info['primary_keys']:
        idx.update(pk if isinstance(pk, list) else [pk])
    for a, b in db_info['foreign_keys']:
        idx.update([a, b])
    return {(tn[cols[i][0]], cols[i][1]) for i in idx if cols[i][0] >= 0}


LET = 'ABCDEFGH'


def distractors(v, target, dom, exclude=(), m=3):
    """in-domain competitors for the select prompt: values most similar to v that are not v, the target,
    normalization-identical to the target, or already known variants (exclude)."""
    nt, nv = norm(target), norm(v)
    ex = set(exclude) | {v, target}
    cnt = dict(dom)
    twins = sorted((x for x, _ in dom if x not in ex and norm(x) == nv and nv != nt), key=lambda x: -cnt.get(x, 0))
    pool = [x for x, _ in dom if x not in ex and norm(x) != nt and norm(x) != nv]
    out = twins[:1]
    if pool:
        sims = process.extract(v, pool, scorer=fuzz.WRatio, limit=12, processor=str.lower)
        sims = sorted(sims, key=lambda x: (-x[1], -cnt.get(x[0], 0)))
        out += [x[0] for x in sims[:m - len(out)]]
    return out


def has_competing_twin(v, target, dom):
    """v is normalization-identical to another stored value that is not a variant of the target -> v denotes that one"""
    nt, nv = norm(target), norm(v)
    return bool(nv) and nv != nt and any(x != v and norm(x) == nv for x, _ in dom)


def select_prompt(db, tc, v, opts, dom):
    """multiple choice: which option does stored value v denote? last letter = none of the above"""
    ex = ', '.join(json.dumps(x, ensure_ascii=False) for x in top_values(dom))
    lines = '\n'.join(f'{LET[i]}. {json.dumps(o, ensure_ascii=False)}' for i, o in enumerate(opts))
    none = LET[len(opts)]
    return (f"Column `{tc[1]}` of table `{tc[0]}` in database `{db}`. Frequent values: {ex}.\n"
            f"The same value may be recorded in different ways (letter case, spacing, punctuation, typos, abbreviations, "
            f"standard codes). Which value below does the stored value {json.dumps(v, ensure_ascii=False)} denote?\n{lines}\n"
            f"{none}. none of the above (it is a different value)\n"
            f"Choose an option only if the stored value is clearly another way of writing it; values that are merely "
            f"similar but denote different things (e.g., different names, codes or categories) are different values.\n"
            f"Answer with the letter only.")


# ---------------------------------------------------------------- v3: database-statistics evidence
def short_codes(a, b):
    """two distinct codes of <= 3 characters are never variants of each other (a typo of a short code is another code)"""
    return len(a.strip()) <= 3 and len(b.strip()) <= 3 and norm(a) != norm(b)


PRIOR_SIM = [0]  # v4: set by the driver (--prior_sim); 0 = v3 behaviour


def freq_distinct(fa, fb, fmin=5, rho=0.3, a=None, b=None):
    """two values that are both frequent and of comparable frequency are distinct categories, not a value and its variant.
    v4: frequency is evidence of distinctness only for pairs that are not near-identical strings; near-identical pairs
    (character similarity >= PRIOR_SIM) go to the LLM verifier instead (a repeated misspelling can be frequent)."""
    lo, hi = min(fa, fb), max(fa, fb)
    if not (lo >= fmin and hi > 0 and lo / hi >= rho):
        return False
    if PRIOR_SIM[0] and a is not None and b is not None and fuzz.ratio(a.lower(), b.lower()) >= PRIOR_SIM[0]:
        return False
    return True


def typo_gate(c, v, fc, fv, sim=90, asym=0.5):
    """a rare, near-identical spelling of a frequent value: skip the reverse check (a canonical form does not 'denote' its typo)"""
    return len(c.strip()) >= 5 and fuzz.ratio(c.lower(), v.lower()) >= sim and fv <= asym * max(1, fc)


def expand_prompt(db, tc, c, dom):
    ex = ', '.join(json.dumps(v, ensure_ascii=False) for v in top_values(dom))
    return (f"Column `{tc[1]}` of table `{tc[0]}` in database `{db}`. Frequent values: {ex}.\n"
            f"List up to 5 other ways the value {json.dumps(c, ensure_ascii=False)} might be recorded in this column by a "
            f"different data source: a standard code or abbreviation, the full name, or a common alternative spelling. "
            f"Answer with a JSON list of strings only.")


def parse_strlist(txt):
    m = re.search(r'\[.*?\]', txt, re.S)
    if not m:
        return []
    try:
        xs = json.loads(m.group(0))
    except Exception:
        return []
    return [x for x in xs if isinstance(x, str) and 0 < len(x) <= 80][:5]


def pair_prompt(db, tc, a, b, dom):
    ex = ', '.join(json.dumps(v, ensure_ascii=False) for v in top_values(dom))
    return (f"Column `{tc[1]}` of table `{tc[0]}` in database `{db}`. Frequent values in this column: {ex}.\n"
            f"Value A: {json.dumps(a, ensure_ascii=False)}\nValue B: {json.dumps(b, ensure_ascii=False)}\n"
            f"Are A and B two ways of recording the SAME value of this column (they may differ only in letter case, spacing, "
            f"punctuation, a typo, an abbreviation, or a standard code/synonym), rather than two different values? "
            f"Answer Yes or No.")


def eq_prompt(db, tc, c, cands, dom, question):
    ex = ', '.join(json.dumps(v, ensure_ascii=False) for v in top_values(dom))
    lines = '\n'.join(f'{i + 1}. {json.dumps(v, ensure_ascii=False)}' for i, (v, _, _) in enumerate(cands))
    return (f"Database `{db}`, column `{tc[1]}` of table `{tc[0]}`. Frequent values in this column: {ex}.\n"
            f"A SQL query for the question \"{question}\" filters this column with the value {json.dumps(c, ensure_ascii=False)}.\n"
            f"The column may store the same real-world value in different ways (letter case, spacing, punctuation, typos, "
            f"abbreviations, synonyms). Which of the following stored values denote exactly the same real-world value as "
            f"{json.dumps(c, ensure_ascii=False)}? Do NOT select values that denote a different entity or category, even if they look "
            f"similar (e.g., 'Baltimore County' differs from 'Baltimore City').\nCandidates:\n{lines}\n"
            f"Answer with a JSON list of the matching candidate numbers only (e.g., [1, 3] or [])." )


def group_prompt(db, tc, block, question):
    lines = '\n'.join(f'{i + 1}. {json.dumps(v, ensure_ascii=False)}' for i, v in enumerate(block))
    return (f"Database `{db}`, column `{tc[1]}` of table `{tc[0]}` (used in a query for the question \"{question}\").\n"
            f"The column may store the same real-world value in different ways (letter case, spacing, punctuation, typos, "
            f"abbreviations, synonyms). Group the following values so that each group contains values denoting exactly the "
            f"same real-world value. Values that denote different entities must be in different groups.\nValues:\n{lines}\n"
            f"Answer with a JSON list of groups, each a list of value numbers, covering every number exactly once "
            f"(e.g., [[1, 2], [3]]).")


def parse_numlist(txt, n):
    m = re.search(r'\[[\d,\s]*\]', txt)
    if not m:
        return []
    try:
        xs = json.loads(m.group(0))
    except Exception:
        return []
    return sorted(set(int(x) - 1 for x in xs if isinstance(x, int) and 1 <= x <= n))


def parse_groups(txt, n):
    m = re.search(r'\[\s*\[.*\]\s*\]', txt, re.S)
    if not m:
        return [[i] for i in range(n)]
    try:
        gs = json.loads(m.group(0))
    except Exception:
        return [[i] for i in range(n)]
    seen, out = set(), []
    for g in gs:
        if not isinstance(g, list):
            continue
        gg = [int(x) - 1 for x in g if isinstance(x, int) and 1 <= x <= n and int(x) - 1 not in seen]
        seen.update(gg)
        if gg:
            out.append(gg)
    out += [[i] for i in range(n) if i not in seen]
    return out


# ---------------------------------------------------------------- rewriting
def lit(v):
    return exp.Literal.string(v)


def apply_eq(site, varset):
    """varset: ordered list of strings to match (includes original constants)"""
    node, col = site['node'], site['col']
    new = exp.In(this=col.copy(), expressions=[lit(v) for v in varset])
    if site['kind'] == 'neq':
        new = exp.Not(this=new)
    node.replace(new)


def case_expr(col, groups):
    """groups: list of (canonical, [variants]) with len(variants)>=1 excluding canonical"""
    ifs = []
    for canon, vs in groups:
        ifs.append(exp.If(this=exp.In(this=col.copy(), expressions=[lit(v) for v in vs]), true=lit(canon)))
    return exp.Case(ifs=ifs, default=col.copy())


def _under(node, top, types):
    """True if some node strictly between `node` and `top` is an instance of `types`"""
    p = node.parent
    while p is not None and p is not top:
        if isinstance(p, types):
            return True
        p = p.parent
    return False


def apply_group(site, groups, schema):
    """replace occurrences of the group column in projections / GROUP BY / ORDER BY / HAVING of its select"""
    if not groups:
        return 0
    sel, tc, scope = site['select'], site['tc'], site['scope']
    targets = [c for c in sel.find_all(exp.Column)
               if c.find_ancestor(exp.Select) is sel and not _under(c, sel, (exp.Where, exp.Join, exp.From))
               and _resolve(c, scope, schema) == tc]
    for c in targets:
        if c.parent is sel and c.arg_key == 'expressions':
            c.replace(exp.alias_(case_expr(c, groups), c.name, quoted=True))  # keep output column name
        else:
            c.replace(case_expr(c, groups))
    return len(targets)


def restricted_domain_sql(site):
    """SELECT DISTINCT col FROM <from/joins> WHERE <where>  for the group site's select (query-driven restriction)."""
    sel = site['select']
    q = exp.Select(expressions=[site['col'].copy()]).distinct()
    for part in ('from', 'joins', 'where', 'with'):
        if sel.args.get(part) is not None:
            v = sel.args[part]
            q.set(part, [x.copy() for x in v] if isinstance(v, list) else v.copy())
    return q.sql(dialect='sqlite')
