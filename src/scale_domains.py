"""Scalability study: grow the value domains of the attributes that seed-3 queries compare or group on, in a few BIRD
databases, by a factor K with synthetic distinct values, and write a new root (other databases hard-linked).
python scale_domains.py SRC_ROOT DST_ROOT K DB [DB...]
Each target attribute with d distinct text values gets (K-1)*d new rows (one per new value, capped at --cap), copied from
random existing rows with only that attribute replaced. Synthetic values are drawn from the attribute's own token
vocabulary and length distribution (so blocking sees realistic near neighbours) and never collide, after normalization,
with an existing value. Tables are rebuilt without constraints/indexes (only domain scans matter here).
Answers to queries over these attributes change, so this root is for latency / call counts only, not for EX."""
import json, os, sys, random, shutil, sqlite3, collections, re
import varfix as V

src, dst, K = sys.argv[1].rstrip('/'), sys.argv[2].rstrip('/'), int(sys.argv[3])
DBS = sys.argv[4:]
CAP = int(os.environ.get('SCALE_CAP', '300000'))
CLEAN = os.environ.get('CANDOR_BIRD_DEV', 'data/bird/dev_20240627')
tables = {d['db_id']: d for d in json.load(open(f'{CLEAN}/dev_tables.json'))}
schemas = {db: V.Schema(info) for db, info in tables.items()}
dev = json.load(open(f'{src}/dev.json'))
from run_varfix import load_preds
sqls = load_preds(f'{src}/pred_omni7b.json')
targets = collections.defaultdict(set)
for e, sql in zip(dev, sqls):
    if e['db_id'] not in DBS or not sql:
        continue
    tree, eqs, groups = V.analyze(sql, schemas[e['db_id']])
    for s in eqs + groups:
        targets[e['db_id']].add(tuple(s['tc']))

os.makedirs(dst, exist_ok=True)
for x in os.listdir(src):
    if x == 'dev_databases' or os.path.exists(f'{dst}/{x}'):
        continue
    if os.path.isdir(f'{src}/{x}'):
        shutil.copytree(f'{src}/{x}', f'{dst}/{x}', copy_function=os.link)
    else:
        os.link(f'{src}/{x}', f'{dst}/{x}')
os.makedirs(f'{dst}/dev_databases', exist_ok=True)
for db in sorted(os.listdir(f'{src}/dev_databases')):
    if db not in DBS and not os.path.exists(f'{dst}/dev_databases/{db}'):
        shutil.copytree(f'{src}/dev_databases/{db}', f'{dst}/dev_databases/{db}', copy_function=os.link)

rep = {'K': K, 'cap': CAP, 'dbs': {}}
rng = random.Random(K)
for db in DBS:
    d = f'{dst}/dev_databases/{db}'
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(f'{src}/dev_databases/{db}', d)          # real copy: this one is modified
    con = sqlite3.connect(f'{d}/{db}.sqlite')
    con.text_factory = lambda b: b.decode('utf-8', errors='ignore')
    rep['dbs'][db] = {}
    rebuilt, orig_rows = set(), {}
    for t, c in sorted(targets[db]):
        vals = con.execute(f'SELECT `{c}`, COUNT(*) FROM `{t}` WHERE typeof(`{c}`)="text" GROUP BY `{c}`').fetchall()
        if len(vals) < 2:
            continue
        n_new = min(CAP, (K - 1) * len(vals))
        keys = {V.norm(v) or v for v, _ in vals}
        toks = [w for v, _ in vals for w in re.findall(r'\S+', v)]
        lens = [max(1, len(re.findall(r'\S+', v))) for v, _ in vals]
        chars = sorted(set(''.join(v for v, _ in vals if v)) - set(' ')) or list('abcdefghijklmnopqrstuvwxyz')
        new = []
        fails, ctr = 0, 0
        while len(new) < n_new:
            L = rng.choice(lens)
            ctr += 1
            if fails >= 20:   # small vocabularies (e.g. 'M'/'F'): suffix a fresh counter so the loop always terminates
                s = f'{rng.choice(vals)[0] or "x"} {ctr}'
            elif L == 1 or len(set(toks)) < 20:
                base = rng.choice(vals)[0] or 'x'
                s = ''.join(rng.choice(chars) for _ in range(max(2, len(base))))
            else:
                s = ' '.join(rng.choice(toks) for _ in range(L))
            k = V.norm(s) or s
            if k in keys:
                fails += 1
                continue
            fails = 0
            keys.add(k)
            new.append(s)
        cols = [r[1] for r in con.execute(f'PRAGMA table_info(`{t}`)').fetchall()]
        ci = cols.index(c)
        if t not in rebuilt:
            rebuilt.add(t)
            orig_rows[t] = con.execute(f'SELECT * FROM `{t}` ORDER BY RANDOM() LIMIT 5000').fetchall()
            con.execute(f'CREATE TABLE `__scale_tmp` AS SELECT * FROM `{t}`')
            con.execute(f'DROP TABLE `{t}`')
            con.execute(f'ALTER TABLE `__scale_tmp` RENAME TO `{t}`')
        rows = orig_rows[t]
        batch = []
        for s in new:
            r = list(rng.choice(rows))
            r[ci] = s
            batch.append(r)
        con.executemany(f'INSERT INTO `{t}` VALUES ({",".join("?" * len(cols))})', batch)
        con.commit()
        rep['dbs'][db][f'{t}.{c}'] = {'distinct_before': len(vals), 'added': len(new), 'distinct_after': len(vals) + len(new)}
        print(db, t, c, rep['dbs'][db][f'{t}.{c}'], flush=True)
    con.close()
json.dump(rep, open(f'{dst}/scale_meta.json', 'w'), indent=1)
print('done', dst)
