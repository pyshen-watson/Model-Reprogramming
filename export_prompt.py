import pandas as pd
from pathlib import Path
from typing import Dict, List

def parse_hp(hp_path: Path):

    if not hp_path.exists():
        return None

    with open(hp_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        ds = [line for line in lines if "dataset:" in line][0].split()[-1]
        vp = [line for line in lines if "vp:" in line][0].split()[-1] == "true"
        fc = [line for line in lines if "fc:" in line][0].split()[-1] == "true"
        level = eval([line for line in lines if "level:" in line][0].split()[-1])
        group = eval([line for line in lines if "group:" in line][0].split()[-1])
        
        return { "ds": ds, "vp": vp, "fc": fc, "level": level, "group": group }

def parse_metrics(metrics_path: Path):
    
    if not metrics_path.exists():
        return None

    with open(metrics_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        attr = lines[-1].split(',')
        return { "acc": eval(attr[-2]), "loss": eval(attr[-1]) }

def parse_ckpt(ckpt_path: Path):
        
    if not ckpt_path.exists():
        return None
    
    attr = ckpt_path.stem.split('-')
    return { "loss": eval(attr[2].split('=')[-1]), "acc": eval(attr[3].split('=')[-1]) }

def export(records: List[Dict]):
    df = pd.DataFrame(records)
    df.sort_values(by=["ds", "name", "level", "group"], inplace=True)
    df.to_csv("records.csv", index=False)

def main():

    # Create the target directory list
    ROOT = Path(".record/VP/DNN/CF10-CE")
    all_dir = [d for d in ROOT.iterdir() if d.is_dir()]
    all_records = []

    for D in all_dir:
        for d in D.iterdir():
            hp_path = d / "hparams.yaml"
            hp = parse_hp(hp_path)
            
            if hp is None:
                continue
            
            vp = hp["vp"]
            fc = hp["fc"]
            
            if not vp and not fc: # Baseline
                mt = parse_metrics(d / "metrics.csv")
                if mt is None:
                    continue
                hp.update(mt)
            
            else:
                ckpt_path = list(d.glob("**/*.ckpt"))
                if len(ckpt_path) == 0:
                    continue
                ckpt_path = ckpt_path[0]
                mt = parse_ckpt(ckpt_path)
                if mt is None:
                    continue
                hp.update(mt)
            
            hp.pop("vp")
            hp.pop("fc")
            if not vp and not fc:
                hp.update({"name": "Baseline"})
            elif vp and not fc:
                hp.update({"name": "VP"})
            elif not vp and fc:
                hp.update({"name": "FC"})
            else:
                hp.update({"name": "MIX"})
                
            all_records.append(hp)
            
    export(all_records)
        
if __name__ == '__main__':
    main()
    