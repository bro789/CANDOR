"""Build OmniSQL inputs (BM25 value index + prompts) for a BIRD-format root (clean or dirty).
Faithful to OmniSQL's build_contents_index.py / process_dataset.py, but with per-root temp dirs so
several roots can be processed in parallel.
Usage: python prep_omnisql.py ROOT   (ROOT contains dev.json, dev_tables.json, dev_databases/)
"""
import json, os, sys, subprocess, sqlite3, shutil, tempfile
OMNI = os.environ.get('OMNISQL_DIR', 'third_party/OmniSQL/train_and_evaluate')


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_content_index(db_file, index_path, tmp):
    con = sqlite3.connect(db_file, check_same_thread=False)
    con.text_factory = lambda b: b.decode(errors='ignore')
    cur = con.cursor()
    tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';")]
    docs = []
    for t in tabs:
        if t == 'sqlite_sequence':
            continue
        cols = [r[0] for r in cur.execute(f"SELECT name FROM PRAGMA_TABLE_INFO('{t}')")]
        for c in cols:
            try:
                vals = [r[0] for r in cur.execute(f'SELECT DISTINCT `{c}` FROM `{t}` WHERE `{c}` IS NOT NULL;')]
            except Exception as e:
                print(e)
                continue
            vals = [v for v in vals if isinstance(v, str) and not is_number(v)]
            for cid, v in enumerate(vals):
                if 0 < len(v) <= 40:
                    docs.append({'id': f'{t}-**-{c}-**-{cid}', 'contents': v})
    con.close()
    os.makedirs(tmp, exist_ok=True)
    with open(f'{tmp}/contents.json', 'w') as f:
        f.write(json.dumps(docs, indent=2, ensure_ascii=True))
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    cmd = [sys.executable, '-m', 'pyserini.index.lucene', '--collection', 'JsonCollection', '--input', tmp,
           '--index', index_path, '--generator', 'DefaultLuceneDocumentGenerator', '--threads', '16',
           '--storePositions', '--storeDocvectors', '--storeRaw']
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    os.remove(f'{tmp}/contents.json')


def main():
    root = sys.argv[1].rstrip('/')
    dbs = sorted(d for d in os.listdir(f'{root}/dev_databases')
                 if os.path.exists(f'{root}/dev_databases/{d}/{d}.sqlite'))
    idx_root = f'{root}/db_contents_index'
    tmp = tempfile.mkdtemp(prefix='omniidx_')
    for db in dbs:
        if os.path.exists(f'{idx_root}/{db}/segments_1') or (os.path.isdir(f'{idx_root}/{db}') and any(x.startswith('segments') for x in os.listdir(f'{idx_root}/{db}'))):
            continue
        build_content_index(f'{root}/dev_databases/{db}/{db}.sqlite', f'{idx_root}/{db}', tmp)
        print('indexed', db, flush=True)
    out = f'{root}/dev_bird.json'
    cmd = [sys.executable, 'process_dataset.py', '--input_data_file', f'{root}/dev.json', '--output_data_file', out,
           '--db_path', f'{root}/dev_databases/', '--tables', f'{root}/dev_tables.json', '--source', 'bird',
           '--mode', 'dev', '--value_limit_num', os.environ.get('CANDOR_VALUE_LIMIT', '2'), '--db_content_index_path', idx_root]
    subprocess.run(cmd, check=True, cwd=OMNI)
    print('wrote', out)


if __name__ == '__main__':
    main()
