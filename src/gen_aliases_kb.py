"""Deterministic 'knowledge alias' variants from standard code lists (no LLM in the generator):
ISO 3166 country name <-> alpha-2/alpha-3 code, US state name <-> postal code, chemical element symbol <-> name,
gender codes, yes/no codes, month names <-> abbreviations. Output format = aliases.json for inject.py.
"""
import json, os, sys, sqlite3, random
sys.path.insert(0, os.path.dirname(__file__))
import pycountry
from inject import eligible_columns, gold_join_columns

STATES = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
          'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
          'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
          'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
          'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
          'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
          'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
          'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
          'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
          'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC', 'Puerto Rico': 'PR'}
ELEM = ('h he li be b c n o f ne na mg al si p s cl ar k ca sc ti v cr mn fe co ni cu zn ga ge as se br kr rb sr y zr nb mo '
        'tc ru rh pd ag cd in sn sb te i xe cs ba la ce pr nd pm sm eu gd tb dy ho er tm yb lu hf ta w re os ir pt au hg tl pb bi').split()
ELEM_NAMES = ('hydrogen helium lithium beryllium boron carbon nitrogen oxygen fluorine neon sodium magnesium aluminium silicon '
              'phosphorus sulfur chlorine argon potassium calcium scandium titanium vanadium chromium manganese iron cobalt nickel '
              'copper zinc gallium germanium arsenic selenium bromine krypton rubidium strontium yttrium zirconium niobium '
              'molybdenum technetium ruthenium rhodium palladium silver cadmium indium tin antimony tellurium iodine xenon caesium '
              'barium lanthanum cerium praseodymium neodymium promethium samarium europium gadolinium terbium dysprosium holmium '
              'erbium thulium ytterbium lutetium hafnium tantalum tungsten rhenium osmium iridium platinum gold mercury thallium '
              'lead bismuth').split()
assert len(ELEM) == len(ELEM_NAMES), (len(ELEM), len(ELEM_NAMES))
EL = dict(zip(ELEM, ELEM_NAMES))
GENDER = {'m': 'Male', 'f': 'Female', 'male': 'M', 'female': 'F'}
YESNO = {'yes': 'Y', 'no': 'N', 'y': 'Yes', 'n': 'No', 'true': 'T', 'false': 'F'}
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


def country_alias(v, rng):
    try:
        c = pycountry.countries.lookup(v)
    except LookupError:
        return None
    if v.upper() in (c.alpha_2, c.alpha_3):
        return getattr(c, 'common_name', None) or c.name
    if v.lower() in (c.name.lower(), getattr(c, 'common_name', '').lower(), getattr(c, 'official_name', '').lower()):
        return rng.choice([c.alpha_2, c.alpha_3])
    return None


def alias(v, key, rng, domain_low):
    """only for columns whose semantics make the code list applicable (decided from column name / value set)"""
    low = v.strip().lower()
    tab, col = key.split('.')[0].lower(), key.split('.')[-1].lower()
    if col == 'element' and low in EL:
        return EL[low]
    if col in ('gender', 'sex') and low in GENDER:
        w = GENDER[low]
        return w.upper() if v.isupper() else w
    if domain_low <= set(YESNO) and low in YESNO:
        return YESNO[low]
    if 'state' in col or col == 'location':
        if v in STATES:
            return STATES[v]
        inv = {b: a for a, b in STATES.items()}
        if v in inv and 'state' in col:
            return inv[v]
    if 'month' in col:
        for m in MONTHS:
            if low == m.lower():
                return m[:3]
    if 'countr' in col or (tab == 'country' and col == 'name') or col == 'location':
        w = country_alias(v, rng)
        if w and w.lower() != low:
            return w
    return None


def main():
    B = sys.argv[1] if len(sys.argv) > 1 else 'data/bird/dev_20240627'
    OUTF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'), 'aliases_kb.json')
    dev = json.load(open(f'{B}/dev.json'))
    tables = {d['db_id']: d for d in json.load(open(f'{B}/dev_tables.json'))}
    jc = gold_join_columns(dev)
    out, n = {}, 0
    for db in sorted(tables):
        rng = random.Random(db)
        con = sqlite3.connect(f'{B}/dev_databases/{db}/{db}.sqlite')
        con.text_factory = lambda b: b.decode(errors='ignore')
        for t, c, rows in eligible_columns(con, tables[db], jc[db]):
            key = f'{t}.{c}'
            dl = set(v.strip().lower() for v, _ in rows)
            for v, _ in rows:
                w = alias(v, key, rng, dl)
                if w and w != v:
                    out.setdefault(db, {}).setdefault(key, {})[v] = w
                    n += 1
        con.close()
    json.dump(out, open(OUTF, 'w'), ensure_ascii=False, indent=0)
    for db, cols in out.items():
        print(db, {k: (len(v), list(v.items())[:3]) for k, v in cols.items()})
    print('total', n)


if __name__ == '__main__':
    main()
