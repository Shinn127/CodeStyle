# Donor transition pilot (v1) report

- run_id: 20261013-022258_seed42_bounded  seed=42
- status: complete_zero_accepted  complete=True
- Every sample is review_status=pending / train_eligible=false: machine-passed only,
  human naturalness review is still required before any training use.

## Funnel

- reference windows (calibration): 78
- geometric pairs selected: 4 (cap 4)
- duration attempts: 12 (cap 3 x max_pairs)
- accepted samples: 0 (cap 4)
- videos: 1 (cap 2)

### Rejection reasons (attempt level)

- ik_unreachable: 12
- ik_correction: 12
- raw_continuity_local_lin_jerk_mps3_max: 12
- raw_contact_slip_p95: 9
- ik_joint_delta: 9
- raw_continuity_local_pos_step_cm_max: 7
- raw_continuity_local_lin_vel_mps_max: 7
- raw_contact_slip_max: 6
- raw_contact_anchor_drift: 6
- b_suffix_mean_pose: 6
- raw_continuity_local_lin_acc_mps2_max: 6
- b_suffix_max_pose: 5
- raw_ground_penetration: 3
- bone_length: 3
- raw_continuity_local_pos_step_cm_mean: 3
- raw_continuity_local_lin_vel_mps_mean: 3
- raw_continuity_local_lin_jerk_mps3_mean: 3
- raw_continuity_world_lin_jerk_mps3_max: 3
- raw_continuity_local_lin_acc_mps2_mean: 3
- b_suffix_root_tilt: 2

### Near-miss attempts (fewest failing gates)

- ArmsAboveHead__ArmsBehindBack__BW__a4341_d3849 T=24: 10 failing gates (ik_unreachable, ik_correction, raw_continuity_local_pos_step_cm_max, raw_continuity_local_lin_vel_mps_max, raw_continuity_local_lin_jerk_mps3_max)
- ArmsAboveHead__ArmsBehindBack__BW__a4341_d3849 T=36: 10 failing gates (ik_unreachable, ik_correction, raw_continuity_local_lin_jerk_mps3_max, raw_contact_slip_p95, raw_contact_slip_max)
- ArmsAboveHead__ArmsBehindBack__BW__a4341_d3849 T=48: 10 failing gates (ik_unreachable, ik_correction, raw_continuity_local_lin_jerk_mps3_max, raw_contact_slip_p95, raw_contact_slip_max)
- ArmsAboveHead__ArmsBehindBack__SW__a2565_d2262 T=24: 10 failing gates (ik_unreachable, ik_correction, ik_joint_delta, bone_length, raw_continuity_local_pos_step_cm_max)
- ArmsAboveHead__ArmsBehindBack__SW__a2565_d2262 T=36: 10 failing gates (ik_unreachable, ik_correction, ik_joint_delta, bone_length, raw_continuity_local_pos_step_cm_max)
- With zero accepted samples these are the closest constructed candidates; they are still rejected and are not fabricated into accepted data.

### Accepted coverage (style pair / suffix)

- none

### Unavailable directed groups (train side missing or skeleton mismatch)

- Neutral->ArmsAboveHead:FW segments=[{'entry_id': 415, 'split': 'val'}, {'entry_id': 27, 'split': 'val'}] reject=split_or_missing
- Neutral->ArmsFolded:FW segments=[{'entry_id': 415, 'split': 'val'}, {'entry_id': 51, 'split': 'train'}] reject=split_or_missing
- Neutral->ArmsBehindBack:FW segments=[{'entry_id': 415, 'split': 'val'}, {'entry_id': 35, 'split': 'train'}] reject=split_or_missing
- ArmsAboveHead->Neutral:FW segments=[{'entry_id': 27, 'split': 'val'}, {'entry_id': 415, 'split': 'val'}] reject=split_or_missing
- ArmsAboveHead->ArmsFolded:FW segments=[{'entry_id': 27, 'split': 'val'}, {'entry_id': 51, 'split': 'train'}] reject=split_or_missing
- ArmsAboveHead->ArmsBehindBack:FW segments=[{'entry_id': 27, 'split': 'val'}, {'entry_id': 35, 'split': 'train'}] reject=split_or_missing
- ArmsFolded->Neutral:FW segments=[{'entry_id': 51, 'split': 'train'}, {'entry_id': 415, 'split': 'val'}] reject=split_or_missing
- ArmsFolded->ArmsAboveHead:FW segments=[{'entry_id': 51, 'split': 'train'}, {'entry_id': 27, 'split': 'val'}] reject=split_or_missing
- ArmsBehindBack->Neutral:FW segments=[{'entry_id': 35, 'split': 'train'}, {'entry_id': 415, 'split': 'val'}] reject=split_or_missing
- ArmsBehindBack->ArmsAboveHead:FW segments=[{'entry_id': 35, 'split': 'train'}, {'entry_id': 27, 'split': 'val'}] reject=split_or_missing
- Neutral->ArmsFolded:SW segments=[{'entry_id': 418, 'split': 'train'}, {'entry_id': 54, 'split': 'val'}] reject=split_or_missing
- ArmsAboveHead->ArmsFolded:SW segments=[{'entry_id': 30, 'split': 'train'}, {'entry_id': 54, 'split': 'val'}] reject=split_or_missing
- ArmsFolded->Neutral:SW segments=[{'entry_id': 54, 'split': 'val'}, {'entry_id': 418, 'split': 'train'}] reject=split_or_missing
- ArmsFolded->ArmsAboveHead:SW segments=[{'entry_id': 54, 'split': 'val'}, {'entry_id': 30, 'split': 'train'}] reject=split_or_missing
- ArmsFolded->ArmsBehindBack:SW segments=[{'entry_id': 54, 'split': 'val'}, {'entry_id': 38, 'split': 'train'}] reject=split_or_missing
- ArmsBehindBack->ArmsFolded:SW segments=[{'entry_id': 38, 'split': 'train'}, {'entry_id': 54, 'split': 'val'}] reject=split_or_missing

### Reference hard-gate findings (guide section 5.1)

- raw_bone_length_full_skeleton: limit=0.0001 m, reference failure fraction=0.4487 -> gate kept unchanged and enforced on every sample; the finding is delivered with the reference distribution, p50/p90/p99/max and per-joint offenders instead of lowering or skipping the gate
- bone-length reference distribution: windows=78 p50=8.27e-05 p90=2.65e-04 p99=6.08e-04 max=6.70e-04 m
- worst joints (reference max deviation): LeftForeArm=6.70e-04, RightForeArm=3.91e-04, LeftFoot=3.44e-04, LeftLeg=2.15e-04, Neck1=3.83e-05
- measured on the same real train reference windows as the other calibration; construction copies every per-frame joint translation verbatim, so this deviation is a source-data property and the construction adds exactly zero to it

### Calibration notes

- raw contact hard-gate failure fraction on references: 0.0000
- counts: {'windows': 78, 'slip_p95_fail': 0, 'slip_max_fail': 0, 'drift_fail': 0}
- contact mask rule: stable runs: height<=proxy+2cm and speed<=0.10 m/s for >=3 frames, bridged over <=2 failing frames below height<=proxy+4cm and speed<=0.20 m/s
- thresholds frozen at quality_thresholds.json sha256=9d3e95af38fefc94ed58d62b9070873e73fea82fff860922d54f0d43c17dc918
- caveat: continuity/codec reference windows are 240 rows while accepted samples span 301+T rows; the sample gates use P99 window maxima from the shorter references.

## Accepted samples (machine-pass; human review pending)

- none: every constructed attempt failed at least one machine gate; no sample was fabricated and no threshold was relaxed.

## Videos

- videos/reject_ArmsAboveHead__ArmsBehindBack__BW__a4341_d3849_T24.mp4 (rejected)

## Budget

- GPU seconds (frozen VQ encode/quantize/decode): 0.6
- wall seconds: 38

## Unresolved / honest caveats

- zero accepted samples: every constructed (pair, duration) attempt failed at least one frozen machine gate; no threshold was relaxed and no sample was fabricated
- dominant rejection drivers: ik_unreachable=12, ik_correction=12, raw_continuity_local_lin_jerk_mps3_max=12, raw_contact_slip_p95=9, ik_joint_delta=9, raw_continuity_local_pos_step_cm_max=7
- reference hard-gate finding recorded: raw_bone_length_full_skeleton failure_fraction=0.449 (gate unchanged)
