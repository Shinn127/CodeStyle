"""Render the S4 acceptance JSON (plus manifests) into the archived REPORT.md.

  python analysis/vq_temporal_source/s4_report.py \
      --acceptance S4/acceptance_ep10.json [--secondary S4/acceptance_ep5.json] \
      --out S4/REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def pct(value):
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def gate_table(report: dict) -> str:
    lines = ["| 判据 | 结果 | 关键数字 |", "|---|---|---|"]
    order = ["g0_updates", "g1_d2r_down", "g2_d2e_not_up", "g3_jitter_drop",
             "g4_foot_sliding", "g5_quality", "g6_activity", "g7_style"]
    for name in order:
        g = report["gates"].get(name)
        if g is None:
            continue
        detail = g["detail"]
        if name == "g1_d2r_down":
            text = (f"mean {fmt(detail['d2r_local_mean']['b0_absolute'])}→"
                    f"{fmt(detail['d2r_local_mean']['b1_absolute'])}cm "
                    f"({pct(detail['d2r_local_mean']['relative_change'])}); "
                    f"p95 {fmt(detail['d2r_local_p95']['b0_absolute'])}→"
                    f"{fmt(detail['d2r_local_p95']['b1_absolute'])}cm "
                    f"({pct(detail['d2r_local_p95']['relative_change'])})")
        elif name == "g2_d2e_not_up":
            text = (f"‖Δ²e_q‖ {fmt(detail['b0_absolute'])}→{fmt(detail['b1_absolute'])}cm "
                    f"({pct(detail['relative_change'])})")
        elif name == "g3_jitter_drop":
            text = (f"jitter {fmt(detail['b0'])}→{fmt(detail['b1'])}cm "
                    f"({pct(detail['relative_change'])}; 要求 ≤ -15%)")
        elif name == "g4_foot_sliding":
            text = ("world " + fmt(detail['fs_world_q']['b0']) + "→" + fmt(detail['fs_world_q']['b1'])
                    + f"cm ({pct(detail['fs_world_q']['relative_change'])}); local "
                    + fmt(detail['fs_local_q']['b0']) + "→" + fmt(detail['fs_local_q']['b1'])
                    + f"cm ({pct(detail['fs_local_q']['relative_change'])})")
        elif name == "g5_quality":
            text = "; ".join(
                f"{key.replace('_q', '')} {fmt(v['b0'])}→{fmt(v['b1'])} ({pct(v['relative_change'])})"
                for key, v in detail.items())
        elif name == "g6_activity":
            worst = min(detail.items(), key=lambda kv: kv[1]["relative_change"])
            text = (f"最差项 {worst[0]} {fmt(worst[1]['b0'])}→{fmt(worst[1]['b1'])} "
                    f"({pct(worst[1]['relative_change'])})")
        elif name == "g7_style":
            text = f"accuracy {fmt(detail['b0'], 3)}→{fmt(detail['b1'], 3)} ({pct(detail['change'])})"
        elif name == "g0_updates":
            text = f"updates={fmt(detail.get('total_updates'))} acc={detail.get('acc_weight')}"
        else:
            text = ""
        lines.append(f"| {g['name']} | {g['result']} | {text} |")
    return "\n".join(lines)


def group_table(report: dict) -> str:
    lines = ["| 组 | n | Δ²r (B0→B1) | jitter (B0→B1) |", "|---|---:|---|---|"]
    for name, g in sorted(report["group_breakdown"].items()):
        d2r = g["d2r_local"]
        jit = g["jitter_q"]
        lines.append(
            f"| {name} | {g['n']} | {fmt(d2r['b0'])}→{fmt(d2r['b1'])} | "
            f"{fmt(jit['b0'])}→{fmt(jit['b1'])} |")
    return "\n".join(lines)


def continuous_table(report: dict) -> str:
    lines = ["| 连续路径指标 | B0 | B1 | 变化 |", "|---|---:|---:|---|"]
    for key, v in report["continuous_contrast"].items():
        lines.append(f"| {key} | {fmt(v['b0'])} | {fmt(v['b1'])} | {pct(v['relative_change'])} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--secondary", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report = json.loads(args.acceptance.read_text())
    secondary = json.loads(args.secondary.read_text()) if args.secondary else None

    parts = [
        "# S3/S4：acc 单变量全 VQ 配对 pilot 结果",
        "",
        "依据 [COMMON.md](COMMON.md)（预注册协议，S3 开跑前冻结）。两臂从 ep90 完整 VQ 状态冷启",
        "（init），都更新 E+Q(EMA)+D；唯一变量 = `ACC_WEIGHT`（B0=0，B1=4.9842，S2 标定）。",
        "",
        f"- 主分析 checkpoint：各臂 `epoch_0010.pt`（固定终点，零选择）；acceptance JSON：`{args.acceptance.name}`",
        f"- 结论：**{report['verdict']}**",
        "",
        "## 1. 预注册门结果（固定 val12，配对 bootstrap）",
        "",
        gate_table(report),
        "",
        "## 2. 活动性/防退化（门 6 明细）",
        "",
        "| 代理 | B0 | B1 | 变化 |",
        "|---|---:|---:|---|",
    ]
    for key, v in report["gates"]["g6_activity"]["detail"].items():
        parts.append(f"| {key} | {fmt(v['b0'])} | {fmt(v['b1'])} | {pct(v['relative_change'])} |")
    parts += [
        "",
        '## 3. 连续 AE 对照（区分"量化残差收益"与"两条路径共同变平滑"）',
        "",
        continuous_table(report),
        "",
        "## 4. 分组诊断（小样本，不作选型依据）",
        "",
        group_table(report),
        "",
        "## 5. GT 参照",
        "",
        f"- GT jitter = {fmt(report['gt_reference']['jitter'])}cm；GT world 足滑 = "
        f"{fmt(report['gt_reference']['fs_world'])}cm",
        "",
    ]
    if report.get("manifest_checks"):
        mc = report["manifest_checks"]
        parts += [
            "## 6. Manifest 交叉核验",
            "",
            f"- 同 updates：{mc['same_total_updates']}（{fmt(mc['total_updates'])}）",
            f"- acc_weight：{mc['acc_weight']}",
            f"- 同 DB 指纹：{mc['same_dataset_sha']}；同初始状态：{mc['same_init_state']}",
            "",
        ]
    if secondary:
        parts += [
            "## 7. 中途鲁棒性对照（epoch_0005.pt，非选型依据）",
            "",
            f"- 结论：**{secondary['verdict']}**",
            "",
            gate_table(secondary),
            "",
        ]
    parts += [
        "## 边界与后续",
        "",
        "- 本结果只覆盖表示层 val 诊断；量化向量已变，属新 VQ 身份，不得覆盖现役 controller 或旧 cache。",
        "- 150ep 重训、cache 重建、controller 训练/替换均需另行授权（COMMON.md §8）。",
        "",
    ]
    args.out.write_text("\n".join(parts))
    print("[Done]", args.out)


if __name__ == "__main__":
    main()
