"""Bounded regression entry point; never uses real training data/checkpoints."""
import argparse
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parent
CORE=(
    'test_controller_repair.py','test_style_code_prior.py','test_recipe_training.py',
    'test_clip_recipe_revision.py','test_fp32_pipeline_efficiency.py',
    'test_training_pipeline.py','test_clip_recipe_v2.py','test_clip_recipe_v2_efficiency.py','test_interactive_control.py',
    'test_pose_postprocess.py',
    'test_motion_history_repair.py',
)


def main():
    parser=argparse.ArgumentParser(description='NewCode regression tests, CPU by default')
    parser.add_argument('suite',nargs='?',choices=('cpu','core','gpu','all'),default='cpu')
    parser.add_argument('--collect-only',action='store_true')
    parser.add_argument('--durations',type=int,default=10)
    args,extra=parser.parse_known_args()
    if args.suite in ('gpu','all') and not args.collect_only and not torch.cuda.is_available():
        parser.error('CUDA is not visible. Run this suite in the host GPU environment; use cpu/core for CPU checks.')
    paths=[str(ROOT/'tests'/name) for name in CORE] if args.suite=='core' else [str(ROOT/'tests')]
    selection=['-m','gpu'] if args.suite=='gpu' else ['-m','not gpu'] if args.suite in ('cpu','core') else []
    flags=['--collect-only'] if args.collect_only else []
    command=paths+['-q','--strict-markers',f'--durations={args.durations}']+selection+flags+extra
    print(f'[Tests] suite={args.suite}; FP32/highest/noTF32; disposable fixtures only',flush=True)
    return pytest.main(command)


if __name__=='__main__':
    sys.exit(main())
