"""
Merge all per-table CSVs into a single combined thesis_tables_all.csv file.
This script appends the rows with a first column 'table' to label which table they came from.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / 'thesis_assets' / 'tables'
OUT_CSV = TABLE_DIR / 'thesis_tables_all.csv'

files = [
    'table_3_1_database_schema.csv',
    'table_4_1_core_libraries.csv',
    'table_4_2_xgboost_features.csv',
    'table_5_1_test_checklist.csv',
    'table_5_2_backtesting_results.csv',
    'table_6_1_objectives_vs_achievements.csv'
]

rows = []
for fname in files:
    p = TABLE_DIR / fname
    if not p.exists():
        print(f"Skipping missing file: {p}")
        continue
    with p.open('r', encoding='utf-8') as f:
        # read as raw lines, prefix each with table name
        reader = csv.reader(f)
        header = next(reader, None)
        for r in reader:
            rows.append([fname] + r)

import os
# determine widest row length to write consistent CSV
maxlen = max(len(r) for r in rows) if rows else 0
header_row = ['table'] + [f'col{i}' for i in range(1, maxlen)]

# remove existing file if present and writable
if OUT_CSV.exists():
    try:
        OUT_CSV.unlink()
    except Exception as e:
        print(f"Could not remove existing {OUT_CSV}: {e}")
        # attempt to write to a temp file
        OUT_CSV = OUT_CSV.with_name('thesis_tables_all_tmp.csv')

with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header_row)
    for r in rows:
        # fill to maxlen
        r = r + [''] * (maxlen - len(r))
        writer.writerow(r)

print(f"Merged tables into {OUT_CSV}")
