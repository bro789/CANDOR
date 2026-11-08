"""BIRD-Dirty generator: inject value-level representation variants into text columns of BIRD dev DBs.

Each eligible distinct value v gets 1-2 variants (same real-world value, different surface form).
Each row holding v is replaced, with probability `rate`, by one of v's variants. The original value
stays the canonical form; the ground-truth map variant -> canonical is written next to the DBs.

Eligible column: declared/stored text, not PK/FK, not used as a join key in any gold SQL,
avg length <= 40, mostly alphabetic (not dates / numbers / urls).
Variant types: case, punct (whitespace/punctuation/hyphenation), typo (one edit), abbrev (token dictionary),
alias (standard code lists: ISO 3166 countries, US states, chemical elements, months, gender/yes-no codes;
data/aliases_kb.json, built by gen_aliases_kb.py).
"""
import argparse, json, os, random, re, shutil, sqlite3, collections, string

ABBR = {
    'street': 'St', 'avenue': 'Ave', 'road': 'Rd', 'boulevard': 'Blvd', 'drive': 'Dr', 'county': 'Co.',
    'company': 'Co.', 'corporation': 'Corp.', 'incorporated': 'Inc.', 'limited': 'Ltd.', 'saint': 'St.',
    'mount': 'Mt.', 'north': 'N.', 'south': 'S.', 'east': 'E.', 'west': 'W.', 'university': 'Univ.',
    'department': 'Dept.', 'international': 'Intl.', 'and': '&', 'elementary': 'Elem.', 'school': 'Sch.',
    'district': 'Dist.', 'junior': 'Jr.', 'senior': 'Sr.', 'high': 'Hi', 'unified': 'Unif.',
    'center': 'Ctr.', 'centre': 'Ctr.', 'association': 'Assn.', 'national': 'Natl.', 'grand prix': 'GP',
    'circuit': 'Circ.', 'united states': 'US', 'united kingdom': 'UK', 'number': 'No.', 'fort': 'Ft.',
    'charter': 'Chtr.', 'academy': 'Acad.', 'community': 'Comm.', 'technology': 'Tech.', 'professional': 'Prof.',
    'management': 'Mgmt.', 'government': 'Govt.', 'administration': 'Admin.', 'committee': 'Cmte.',
    'meeting': 'Mtg.', 'general': 'Gen.', 'special': 'Spec.', 'education': 'Ed.', 'services': 'Svcs.',
}
KEYB = {c: n for c, n in zip('qwertyuiopasdfghjklzxcvbnm', 'wqretyuiopsadfghjklzxcvbnm')}
NEIGH = {'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr', 'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
         'k': 'jl', 'l': 'k', 'm': 'n', 'n': 'bm', 'o': 'ip', 'p': 'o', 'q': 'w', 'r': 'et', 's': 'ad', 't': 'ry',
         'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu', 'z': 'x'}

DATE_RE = re.compile(r'^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}')
NUMLIKE = re.compile(r'^[\d\W_]+$')


def gen_case(v, rng):
    # sorted: set iteration order depends on per-process string hashing (PYTHONHASHSEED). The released BIRD-Dirty
    # roots were generated before this fix and are distributed as cell-level deltas (bench_delta.py).
    opts = sorted(x for x in {v.lower(), v.upper(), v.title()} if x != v)
    return rng.choice(opts) if opts else None


def gen_punct(v, rng):
    opts = set()
    if ' ' in v:
        opts.update({v.replace(' ', '-', 1), v.replace(' ', '', 1), v.replace(' ', '  ', 1)})
    if '-' in v:
        opts.update({v.replace('-', ' '), v.replace('-', '')})
    if '.' in v:
        opts.add(v.replace('.', ''))
    if ',' in v:
        opts.add(v.replace(',', ''))
    opts.add(v + ' ')
    opts.discard(v)
    opts = sorted(opts)
    return rng.choice(opts) if opts else None


def gen_typo(v, rng):
    idx = [i for i, ch in enumerate(v) if ch.isalpha()]
    if len(idx) < 5:
        return None
    i = rng.choice(idx[1:])
    op = rng.choice(['del', 'swap', 'sub', 'dup'])
    if op == 'del':
        out = v[:i] + v[i + 1:]
    elif op == 'swap' and i + 1 < len(v):
        out = v[:i] + v[i + 1] + v[i] + v[i + 2:]
    elif op == 'sub':
        ch = v[i]
        nb = NEIGH.get(ch.lower(), 'e')
        r = rng.choice(nb)
        out = v[:i] + (r.upper() if ch.isupper() else r) + v[i + 1:]
    else:
        out = v[:i] + v[i] + v[i:]
    return out if out != v else None


def gen_abbrev(v, rng):
    low = v.lower()
    hits = [k for k in ABBR if re.search(r'\b' + re.escape(k) + r'\b', low)]
    rev = [k for k, a in ABBR.items() if re.search(r'(?<!\w)' + re.escape(a.lower()) + r'(?!\w)', low)]
    if hits:
        k = rng.choice(hits)
        return re.sub(r'\b' + re.escape(k) + r'\b', ABBR[k], v, count=1, flags=re.I)
    if rev:
        k = rng.choice(rev)
        return re.sub(r'(?<!\w)' + re.escape(ABBR[k]) + r'(?!\w)', k.title(), v, count=1, flags=re.I)
    return None


GENS = {'case': gen_case, 'punct': gen_punct, 'typo': gen_typo, 'abbrev': gen_abbrev}


def gold_join_columns(dev):
    cols = collections.defaultdict(set)
    for ex in dev:
        s = ex['SQL']
        for m in re.finditer(r'\bON\b(.*?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\)|$)', s, re.I | re.S):
            for c in re.findall(r'[`"]?(\w+)[`"]?\s*\.\s*[`"]?([\w ]+?)[`"]?\s*(?:=|$)', m.group(1)):
                cols[ex['db_id']].add(c[1].strip().lower())
            for c in re.findall(r'=\s*[`"]?\w+[`"]?\s*\.\s*[`"]?(\w+)[`"]?', m.group(1)):
                cols[ex['db_id']].add(c.strip().lower())
    return cols


def eligible_columns(con, db_info, join_cols):
    cur = con.cursor()
    tnames = db_info['table_names_original']
    keyidx = set()
    for pk in db_info['primary_keys']:
        keyidx.update(pk if isinstance(pk, list) else [pk])
    for a, b in db_info['foreign_keys']:
        keyidx.update([a, b])
    out = []
    for ci, (ti, cname) in enumerate(db_info['column_names_original']):
        if ti < 0 or ci in keyidx or cname.lower() in join_cols:
            continue
        t = tnames[ti]
        rows = cur.execute(f'SELECT `{cname}`, COUNT(*) FROM `{t}` WHERE typeof(`{cname}`)="text" AND `{cname}`!="" GROUP BY `{cname}`').fetchall()
        if len(rows) < 2:
            continue
        vals = [r[0] for r in rows]
        avg = sum(len(x) for x in vals) / len(vals)
        if avg > 40:
            continue
        alpha = sum(1 for x in vals if any(ch.isalpha() for ch in x) and not DATE_RE.match(x) and not x.startswith('http'))
        if alpha / len(vals) < 0.5:
            continue
        out.append((t, cname, rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bird', default=os.environ.get('CANDOR_BIRD_DEV', 'data/bird/dev_20240627'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--rate', type=float, default=0.2)
    ap.add_argument('--types', default='case,punct,typo,abbrev,alias')
    ap.add_argument('--aliases', default=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'), 'aliases_kb.json'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dbs', default='')
    a = ap.parse_args()
    types = a.types.split(',')
    dev = json.load(open(f'{a.bird}/dev.json'))
    tables = {d['db_id']: d for d in json.load(open(f'{a.bird}/dev_tables.json'))}
    jc = gold_join_columns(dev)
    aliases = json.load(open(a.aliases)) if ('alias' in types and os.path.exists(a.aliases)) else {}
    dbs = a.dbs.split(',') if a.dbs else sorted(tables)
    os.makedirs(a.out, exist_ok=True)
    canon = {}
    stats = {}
    for db in dbs:
        rng = random.Random(f'{a.seed}-{db}')
        src = f'{a.bird}/dev_databases/{db}/{db}.sqlite'
        dst_dir = f'{a.out}/dev_databases/{db}'
        os.makedirs(dst_dir, exist_ok=True)
        dst = f'{dst_dir}/{db}.sqlite'
        shutil.copyfile(src, dst)
        for extra in os.listdir(f'{a.bird}/dev_databases/{db}'):
            if extra.endswith('.sqlite'):
                continue
            p = f'{a.bird}/dev_databases/{db}/{extra}'
            if os.path.isdir(p):
                shutil.copytree(p, f'{dst_dir}/{extra}', dirs_exist_ok=True)
        con = sqlite3.connect(dst)
        con.text_factory = lambda b: b.decode(errors='ignore')  # some Spider DBs hold invalid UTF-8
        cur = con.cursor()
        cols = eligible_columns(con, tables[db], jc[db])
        dbmap = {}
        st = collections.Counter()
        for t, c, rows in cols:
            key = f'{t}.{c}'
            dom = set(r[0] for r in rows)
            domlow = set(x.lower().strip() for x in dom)
            al = aliases.get(db, {}).get(key, {})
            used = set()
            vmap = {}
            for v, cnt in rows:
                if not any(ch.isalpha() for ch in v) or DATE_RE.match(v):
                    continue
                k = 1 if rng.random() < 0.7 else 2
                vs = []
                order = types[:]
                rng.shuffle(order)
                for ty in order:
                    if len(vs) >= k:
                        break
                    if ty == 'alias':
                        w = al.get(v)
                    else:
                        w = GENS[ty](v, rng)
                    if not w or w == v or w in dom or w in used:
                        continue
                    if ty in ('typo', 'abbrev', 'alias') and w.lower().strip() in domlow:
                        continue
                    vs.append((w, ty))
                    used.add(w)
                if vs:
                    vmap[v] = vs
            if not vmap:
                continue
            ids = cur.execute(f'SELECT rowid, `{c}` FROM `{t}` WHERE typeof(`{c}`)="text"').fetchall()
            upd = []
            for rid, v in ids:
                if v in vmap and rng.random() < a.rate:
                    w, ty = rng.choice(vmap[v])
                    upd.append((w, rid))
                    dbmap.setdefault(key, {})[w] = v
                    st[ty] += 1
            if upd:
                cur.executemany(f'UPDATE `{t}` SET `{c}`=? WHERE rowid=?', upd)
                st['cols'] += 1
                st['cells'] += len(upd)
        con.commit()
        con.close()
        canon[db] = dbmap
        stats[db] = dict(st, eligible_cols=len(cols))
        print(db, dict(st), 'eligible', len(cols), flush=True)
    json.dump(canon, open(f'{a.out}/canon_map.json', 'w'), ensure_ascii=False)
    json.dump({'args': vars(a), 'stats': stats}, open(f'{a.out}/inject_stats.json', 'w'), indent=1)
    for f in ['dev.json', 'dev_tables.json']:
        shutil.copyfile(f'{a.bird}/{f}', f'{a.out}/{f}')


if __name__ == '__main__':
    main()
