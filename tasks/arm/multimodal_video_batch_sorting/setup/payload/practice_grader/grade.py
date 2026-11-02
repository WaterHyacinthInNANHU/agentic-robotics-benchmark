#!/usr/bin/env python3
"""Public schema/mechanics check; perception targets remain private."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,default=Path.home()/"submission/plan.json");a=p.parse_args()
 x=json.loads(a.plan.read_text());rows=x.get("scenes",[]) if isinstance(x,dict) else []
 ids=["scene_{:02d}".format(i) for i in range(24)];errors=[]
 if x.get("schema_version")!=1 or len(rows)!=24:errors.append("schema/scene count")
 seen=set()
 for row in rows:
  if not isinstance(row,dict) or set(row)!={"scene_id","actions"}:errors.append("bad row");continue
  sid=row["scene_id"];seen.add(sid);acts=row["actions"]
  if sid not in ids or not isinstance(acts,list) or len(acts)>28:errors.append(sid+": actions")
  for q in acts:
   if not isinstance(q,dict) or q.get("op") not in {"clear_guard","pick","place","verify"}:errors.append(sid+": action")
 if seen!=set(ids):errors.append("missing/duplicate scenes")
 print(json.dumps({"schema_mechanics_ok":not errors,"errors":errors[:20],"note":"Perception semantics are hidden."},indent=2))
 if errors:raise SystemExit(1)
if __name__=="__main__":main()
