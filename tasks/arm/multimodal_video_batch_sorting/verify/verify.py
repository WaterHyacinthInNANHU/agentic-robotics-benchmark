import json,os,stat,sys
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from grasp_model import validate_plan,evaluate_scene
from robotics_grader import ratio_rewards,read_anchor,stage_config,write_verdict,zero_verdict
from robotics_grader.stage import agent_home

def run_episodes(config,plans=None):
    scenes=json.loads((Path(__file__).parent/"target.json").read_text())["scenes"];ids=[s["scene_id"] for s in scenes]
    if plans is None:
        p=Path(agent_home())/"submission/plan.json";info=p.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size>262144:raise ValueError("plan.json must be regular and <=256 KiB")
        plans=validate_plan(json.loads(p.read_text()),ids)
    ok=0;reasons={}
    for s in scenes:
        good,reason=evaluate_scene(s,plans[s["scene_id"]]);ok+=int(good);reasons[reason]=reasons.get(reason,0)+1
    out={"success_rate":ok/len(scenes),"complete_successes":float(ok)}
    for k,v in reasons.items():out[k+"_rate"]=v/len(scenes)
    return out

def main():
    anchor,meta=read_anchor()
    try:r=run_episodes(stage_config())
    except (OSError,ValueError,TypeError,KeyError,OverflowError) as e:write_verdict(zero_verdict(reason="missing or invalid plan: "+str(e),anchored=anchor is not None));return
    m=r.pop("success_rate");write_verdict(ratio_rewards(m,anchor,cap=stage_config().get("scoring_cap",1.5),higher_is_better=True,metric_name="success_rate",extra=r,full_at=meta.get("full_at",1.0)))
if __name__=="__main__":main()
