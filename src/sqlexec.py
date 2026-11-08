"""SQLite execution helpers with a hard time limit (progress handler) and canonical-map result comparison."""
import json, os, sqlite3, time
from concurrent.futures import ProcessPoolExecutor


def run_sql(db_file, sql, timeout=30.0, max_rows=100000):
    con = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True, check_same_thread=False)
    con.text_factory = lambda b: b.decode(errors='ignore')
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > timeout else 0, 10000)
    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        return True, rows
    except Exception as e:
        return False, str(e)[:200]
    finally:
        con.close()


def _job(args):
    db_file, sql, timeout = args
    if not sql or not sql.strip():
        return False, 'empty'
    return run_sql(db_file, sql, timeout)


def run_many(jobs, timeout=30.0, workers=48):
    """jobs: list of (db_file, sql). returns list of (ok, rows|err)"""
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_job, [(d, s, timeout) for d, s in jobs], chunksize=4))


def load_global_map(root):
    """variant -> canonical per db (union over columns; conflicting variants dropped)."""
    p = f'{root}/canon_map.json'
    if not os.path.exists(p):
        return {}
    cm = json.load(open(p))
    g = {}
    for db, cols in cm.items():
        m, bad = {}, set()
        for col, vm in cols.items():
            for v, c in vm.items():
                if v in m and m[v] != c:
                    bad.add(v)
                m[v] = c
        for b in bad:
            m.pop(b, None)
        g[db] = m
    return g


def canon_rows(rows, m):
    if not m:
        return set(rows)
    return set(tuple(m.get(x, x) if isinstance(x, str) else x for x in r) for r in rows)
