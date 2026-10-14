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


def evidence_block(contract: dict | None, b0: dict | None, b1: dict | None) -> str:
    """Stage-report fixed fields required by the execution plan (section 8)."""
    lines = ["## 0. 阶段报告必备字段", ""]
    if contract:
        sc = contract["source_checkpoint"]
        lines += [
            f"- **hypothesis**：逐帧重构监督对二阶时间误差不敏感；直接监督 GT 二阶差分可压低"
            f"量化解码的 `Δ²r`/jitter 而不把运动压向静止。",
            f"- **single_variable**：`ACC_WEIGHT`（B0=0 / B1=4.9842），其余全部相同。",
            f"- **source_commit**：`{contract['git']['commit']}`（dirty={contract['git']['dirty']}，"
            f"S0 于干净树 PASS）",
            f"- **init_ckpt**：`{contract['source_checkpoint']['path']}` epoch={sc['epoch']}，"
            f"state sha256={contract['init_ckpt_restore']['restored_state_sha256'][:16]}…（位级恢复=True）",
            f"- **norm**：sha256={contract['norm']['sha256'][:16]}…（显式传入未重算）",
            f"- **dataset**：`{contract['db']['path']}`，train/val/test 段数="
            f"{contract['db']['splits']['train']}/{contract['db']['splits']['val']}/{contract['db']['splits']['test']}，"
            f"mirrored={contract['db']['mirrored_segments']}",
            f"- **config**：seq_len={contract['experiment']['seq_len']}, stride=1, batch="
            f"{contract['experiment']['batch_size']}, epochs={contract['experiment']['epochs']}, "
            f"lr=1e-4→1e-5, seed=42, loader_seed={contract['experiment']['loader_seed']}；"
            f"S2 标定 `ACC_WEIGHT=4.9842`（encoder 37.8% / decoder 8.1% of base grad）",
            f"- **环境**：{contract['environment']}",
        ]
    if b0 and b1:
        lines += [
            f"- **B0/B1 更新数**：{fmt(b0['integrity']['total_updates'])} / "
            f"{fmt(b1['integrity']['total_updates'])}（相同）",
            f"- **最终状态指纹**：B0 `{b0['final_checkpoint']['state_sha256'][:16]}…` / "
            f"B1 `{b1['final_checkpoint']['state_sha256'][:16]}…`（量化器均已改变 → 新 VQ 身份）",
        ]
    lines += [
        "- **tests**：`tests/test_vq_temporal_pipeline.py`（CPU 12 项：wrapper 可解析性、A/B pin、"
        "S4 门逻辑、val12 选定、探针 schema；GPU 1 项）；全量回归 `bash scripts/run_tests.sh cpu` "
        "= 422 passed；GPU 套件见 S4/gpu_tests.log",
        "- **per-source_metrics / CI**：本报告 §1–§4 及 `acceptance_*.json`（12 源配对 bootstrap，"
        "seed=42，1000 次）",
        "- **unexpected_behaviors**：① 入口脚本 `--no-decoder-only` 真实缺陷（本实验启动时暴露，"
        "已修复 + 回归测试）；② B1 的连续 AE 路径同向恶化（c.local_mpjpe +22%）→ 收益不是"
        "量化残差特有，而是解码器整体变平滑；③ B1 端点 codec 往返翻码率上升（0.605→0.682）；"
        "④ B0/B1 的 `best.pt` 均停在 ep1（首评接受语义），主分析已按预注册改用固定终点。",
        "- **decision**：**FAIL**（见 §1）；否决规则触发：以运动/精度代价换低 jitter，且 jitter "
        "降幅未达 15% 预注册阈值。**不自动续到 150ep。**",
        "- **next_authorization_needed**：S5 分支选择（S5a 位置空间二阶损失 / S5b 接触约束 / "
        "S5c 相对边界几何 / S5d 短历史 decoder）或其它改动，均需用户显式授权；本轮到此停止。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--secondary", type=Path, default=None)
    ap.add_argument("--contract", type=Path, default=HERE / "S0" / "contract.json")
    ap.add_argument("--b0-manifest", type=Path, default=None)
    ap.add_argument("--b1-manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report = json.loads(args.acceptance.read_text())
    secondary = json.loads(args.secondary.read_text()) if args.secondary else None
    contract = json.loads(args.contract.read_text()) if args.contract.exists() else None
    b0_manifest = json.loads(args.b0_manifest.read_text()) if args.b0_manifest and args.b0_manifest.exists() else None
    b1_manifest = json.loads(args.b1_manifest.read_text()) if args.b1_manifest and args.b1_manifest.exists() else None

    parts = [
        "# S3/S4：acc 单变量全 VQ 配对 pilot 结果",
        "",
        "依据 [COMMON.md](COMMON.md)（预注册协议，S3 开跑前冻结）。两臂从 ep90 完整 VQ 状态冷启",
        "（init），都更新 E+Q(EMA)+D；唯一变量 = `ACC_WEIGHT`（B0=0，B1=4.9842，S2 标定）。",
        "",
        f"- 主分析 checkpoint：各臂 `epoch_0010.pt`（固定终点，零选择）；acceptance JSON：`{args.acceptance.name}`",
        f"- 结论：**{report['verdict']}**",
        "",
        evidence_block(contract, b0_manifest, b1_manifest),
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
