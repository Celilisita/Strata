"""Bound physical lines in owned implementation files supplied by the caller."""
from pathlib import Path
import sys,json
suffixes={".cpp",".cc",".c",".cu",".hpp",".h",".cuh",".py",".cmake",".sh"}
files=set()
for value in sys.argv[1:]:
 p=Path(value)
 for q in (p.rglob("*") if p.is_dir() else [p]):
  if q.is_file() and q.suffix in suffixes and "__pycache__" not in q.parts:
   files.add(q)
rows=sorted(((str(p),len(p.read_bytes().splitlines())) for p in files),key=lambda row:row[1],reverse=True)
bad=[row for row in rows if row[1]>512]
print(json.dumps({"files":len(rows),"largest":rows[0] if rows else None,"violations":bad}))
sys.exit(bool(bad))
