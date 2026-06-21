import argparse
import json
import os
import subprocess
from pathlib import Path


def read_last_json_line(path: Path):
    if not path.exists():
        return None
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def run_once(cmd: str, cwd: Path, env: dict):
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, shell=True)
    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="COCO_QA_one_obj")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--head-count", type=int, default=32)
    parser.add_argument("--ablate-weight", type=float, default=0.05)
    parser.add_argument("--ablate-layers", default="14-31")
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output-summary", default="outputs/head_screening_sample20.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    res_file = root / "outputs" / "res.json"
    out_summary = root / args.output_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEST_MODE"] = "True"
    env["TEST_SAMPLE_COUNT"] = str(args.sample_count)
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"

    base_cmd = (
        "python3 main_aro.py "
        f"--dataset={args.dataset} --model-name='llava1.5' --download "
        "--method adapt_vis --weight1 0.5 --weight2 1.2 --threshold 0.3 --option=four "
        "--adjust-method none "
        f"--region-config \"{args.region_config}\" "
        "--low-random-th -1"
    )

    results = []
    print("RUN baseline", flush=True)
    if not run_once(base_cmd, root, env):
        raise SystemExit("Baseline run failed")
    base_row = read_last_json_line(res_file)
    if base_row is None:
        raise SystemExit("Baseline result not found in outputs/res.json")

    base_acc = float(base_row.get("Individual accuracy", 0.0))
    results.append(
        {
            "kind": "baseline",
            "acc": base_acc,
            "correct_id": base_row.get("correct_id", []),
        }
    )
    print(f"BASE_ACC {base_acc}", flush=True)

    for head in range(args.head_count):
        cmd = (
            base_cmd
            + f" --ablate-head {head}"
            + f" --ablate-head-weight {args.ablate_weight}"
            + f" --ablate-head-layers {args.ablate_layers}"
        )
        print(f"RUN head={head}", flush=True)
        ok = run_once(cmd, root, env)
        if not ok:
            results.append({"kind": "head", "head": head, "status": "failed"})
            continue
        row = read_last_json_line(res_file)
        if row is None:
            results.append({"kind": "head", "head": head, "status": "failed_no_result"})
            continue
        acc = float(row.get("Individual accuracy", 0.0))
        results.append(
            {
                "kind": "head",
                "head": head,
                "status": "ok",
                "acc": acc,
                "delta_vs_base": base_acc - acc,
                "correct_id": row.get("correct_id", []),
            }
        )

    ok_heads = [r for r in results if r.get("kind") == "head" and r.get("status") == "ok"]
    ok_heads.sort(key=lambda x: x.get("delta_vs_base", -1e9), reverse=True)
    summary = {
        "dataset": args.dataset,
        "sample_count": args.sample_count,
        "ablate_weight": args.ablate_weight,
        "ablate_layers": args.ablate_layers,
        "region_config": args.region_config,
        "baseline_acc": base_acc,
        "top10_by_drop": ok_heads[:10],
        "all": results,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"SUMMARY_PATH {out_summary}", flush=True)


if __name__ == "__main__":
    main()
