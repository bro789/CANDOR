"""Release / transfer format for BIRD-Dirty roots: cell-level deltas against the original BIRD databases.
  export: python bench_delta.py export DIRTY_ROOT OUT.jsonl.gz      (needs the clean BIRD root, CANDOR_BIRD_DEV)
  apply : python bench_delta.py apply  DELTA.jsonl.gz NEW_ROOT       (copies the clean DBs, applies the deltas)
  hash  : python bench_delta.py hash   ROOT                          (content hash of all injected columns)
The canonical map, dev.json and dev_tables.json are copied alongside by the caller."""
import gzip, hashlib, json, os, shutil, sqlite3, sys
CLEAN = os.environ.get('CANDOR_BIRD_DEV', 'data/bird/dev_20240627')


def cols_of(root):
    cm = json.load(open(f'{root}/canon_map.json'))
    return {db: [k.split('.', 1) for k in cols] for db, cols in cm.items()}


def export(root, out):
    n = 0
    with gzip.open(out, 'wt') as f:
        for db, cols in sorted(cols_of(root).items()):
            dcon = sqlite3.connect(f'{root}/dev_databases/{db}/{db}.sqlite')
            dcon.execute(f"ATTACH DATABASE '{CLEAN}/dev_databases/{db}/{db}.sqlite' AS c")
            for t, c in cols:
                q = (f'SELECT d.rowid, d.`{c}` FROM main.`{t}` d JOIN c.`{t}` o ON d.rowid = o.rowid '
                     f'WHERE d.`{c}` IS NOT o.`{c}`')
                for rid, v in dcon.execute(q):
                    f.write(json.dumps([db, t, c, rid, v], ensure_ascii=False) + '\n')
                    n += 1
            dcon.close()
    print('exported', n, 'cells')


def apply(delta, root):
    by = {}
    with gzip.open(delta, 'rt') as f:
        for line in f:
            db, t, c, rid, v = json.loads(line)
            by.setdefault(db, []).append((t, c, rid, v))
    for db in sorted(os.listdir(f'{CLEAN}/dev_databases')):
        if not os.path.exists(f'{CLEAN}/dev_databases/{db}/{db}.sqlite'):
            continue
        os.makedirs(f'{root}/dev_databases/{db}', exist_ok=True)
        dst = f'{root}/dev_databases/{db}/{db}.sqlite'
        shutil.copyfile(f'{CLEAN}/dev_databases/{db}/{db}.sqlite', dst)
        con = sqlite3.connect(dst)
        for t, c, rid, v in by.get(db, []):
            con.execute(f'UPDATE `{t}` SET `{c}`=? WHERE rowid=?', (v, rid))
        con.commit()
        con.close()
    print('applied', sum(len(x) for x in by.values()), 'cells')


def chash(root):
    h = hashlib.md5()
    for db, cols in sorted(cols_of(root).items()):
        con = sqlite3.connect(f'{root}/dev_databases/{db}/{db}.sqlite')
        con.text_factory = lambda b: b.decode(errors='replace')
        for t, c in sorted(cols):
            for rid, v in con.execute(f'SELECT rowid, `{c}` FROM `{t}` ORDER BY rowid'):
                h.update(f'{db}|{t}|{c}|{rid}|{v}\n'.encode())
        con.close()
    print('content-md5', h.hexdigest())


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'export':
        export(sys.argv[2], sys.argv[3])
    elif cmd == 'apply':
        apply(sys.argv[2], sys.argv[3])
    else:
        chash(sys.argv[2])
