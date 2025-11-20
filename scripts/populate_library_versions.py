"""
Populate versions in table_4_1_core_libraries.csv using requirements.txt.
This script will look for matches between package names in CSV and names in requirements.
It updates the CSV with detected versions or 'Not pinned'.
"""

import re
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / 'requirements.txt'
CSV_IN = ROOT / 'thesis_assets' / 'tables' / 'table_4_1_core_libraries.csv'
CSV_OUT = ROOT / 'thesis_assets' / 'tables' / 'table_4_1_core_libraries_with_versions.csv'

# read requirements
reqs = {}
if REQ.exists():
    with open(REQ, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Examples: xgboost or langchain-core==0.2.9
            m = re.match(r'([^=<>!~]+)(==([^#\s]+))?', line)
            if m:
                pkg = m.group(1).strip()
                ver = m.group(3) if m.group(3) else None
                reqs[pkg.lower()] = ver

# read CSV and map versions where possible
rows = []
with open(CSV_IN, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    for row in reader:
        rows.append(row)

# helper cleanup mapping - some names differ
name_map = {
    'psycopg2-binary': 'psycopg2-binary',
    'python-dotenv': 'python-dotenv',
    'scikit-learn': 'scikit-learn',
    'xgboost': 'xgboost',
    'prophet': 'prophet',
    'plotly': 'plotly',
    'seaborn': 'seaborn',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'requests': 'requests',
    'streamlit': 'streamlit',
    'sqlalchemy': 'sqlalchemy',
    'matplotlib': 'matplotlib'
}

# Modify rows
for row in rows:
    lib = row['library'].strip().lower()
    mapped = name_map.get(lib, lib)
    ver = reqs.get(mapped)
    row['version'] = ver if ver else 'Not pinned'

# write out
with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {CSV_OUT}")
