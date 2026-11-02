"""Private replay for complete multi-part sorting missions."""
import math
TOOLS={"gear":"parallel_jaw","plate":"vacuum","coupler":"three_finger"};GRASPS={"gear":"rim","plate":"top","coupler":"body"}
def validate_plan(obj,scene_ids):
 if not isinstance(obj,dict) or obj.get("schema_version")!=1 or set(obj)!={"schema_version","scenes"}:raise ValueError("expected schema_version and scenes")
 rows=obj["scenes"]
 if not isinstance(rows,list) or len(rows)!=len(scene_ids):raise ValueError("one row per scene required")
 out={}
 for row in rows:
  if not isinstance(row,dict) or set(row)!={"scene_id","actions"}:raise ValueError("bad scene row")
  sid=row["scene_id"];acts=row["actions"]
  if sid in out or sid not in scene_ids or not isinstance(acts,list) or len(acts)>28:raise ValueError("bad actions")
  out[sid]=acts
 if set(out)!=set(scene_ids):raise ValueError("missing scene")
 return out
def evaluate_scene(s,actions):
 parts={p["label"]:p for p in s["parts"]};targets={q["label"]:q for q in s["targets"]};cleared=set();held=None;placed={};verified=set()
 for a in actions:
  if not isinstance(a,dict) or "op" not in a:return False,"malformed"
  op=a["op"]
  if op=="clear_guard":
   if set(a)!={"op","label","tool"} or a.get("label") not in targets or a.get("tool")!="parallel_jaw":return False,"bad_clear"
   if not targets[a["label"]]["guard"]:return False,"unneeded_clear"
   cleared.add(a["label"])
  elif op=="pick":
   if set(a)!={"op","label","tool","grasp","yaw_deg"} or held is not None:return False,"bad_pick"
   label=a.get("label")
   if label not in targets or label in placed:return False,"wrong_part"
   q=targets[label]
   if a.get("tool")!=TOOLS[q["kind"]] or a.get("grasp")!=GRASPS[q["kind"]]:return False,"wrong_grasp"
   if q["guard"] and label not in cleared:return False,"guard_collision"
   y=a.get("yaw_deg")
   if isinstance(y,bool) or not isinstance(y,(int,float)) or not math.isfinite(float(y)) or abs(((float(y)-q["yaw"]+180)%360)-180)>1e-6:return False,"wrong_orientation"
   held=label
  elif op=="place":
   if set(a)!={"op","label","bin","tool"} or held!=a.get("label"):return False,"bad_place"
   q=targets[held]
   if a.get("tool")!=TOOLS[q["kind"]] or a.get("bin")!=q["destination"]:return False,"wrong_destination"
   placed[held]=a["bin"];held=None
  elif op=="verify":
   if set(a)!={"op","label","bin"}:return False,"bad_verify"
   label=a.get("label")
   if label not in placed or a.get("bin")!=placed[label]:return False,"bad_verify"
   verified.add(label)
  else:return False,"unknown_op"
 ok=held is None and set(placed)==set(targets) and verified==set(targets)
 return ok,"ok" if ok else "incomplete_batch"
