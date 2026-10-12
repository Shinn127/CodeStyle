"""Export fixed unsmoothed evaluation snippets with the existing BVH writer."""
import json
import tempfile
from pathlib import Path
import numpy as np


def export_example(args, name, ref, gen, *, case):
    """Fixed, unsmoothed clips plus the existing Geno BVH exporter and root-track plot."""
    from Visualization.io.genoview_bvh import export_motion375_segment_to_bvh
    from Utils.motion375 import reconstruct_root_track_from_375
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = args.out / 'examples'; directory.mkdir(exist_ok=True)
    np.savez(directory / f'{name}.npz', reference=ref, generated=gen, case_json=json.dumps(case))
    meta = json.loads((args.db / 'meta.json').read_text())
    segments = json.loads((args.db / 'segments.json').read_text())
    mirror = next((bool(s.get('mirror')) for s in segments
                   if s['frame_start'] <= case['source_start'] < s['frame_start'] + s['num_frames']), False)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        db = Path(temporary)
        for label, clip in [('reference', ref), ('generated', gen)]:
            np.asarray(clip, np.float32).tofile(db / 'X.f32')
            (db / 'meta.json').write_text(json.dumps({'num_frames': len(clip), 'feature_dim': 375,
                'coordinate_convert': bool(meta.get('coordinate_convert', False))}))
            export_motion375_segment_to_bvh(db_path=db, template_bvh=args.bvh_template,
                template_frame=0, segment_index=0, segment_offset=0, frame_start=0,
                num_frames=len(clip), fps=60, root_wy_in_degrees=True, mirror_template=mirror,
                out_bvh=directory / f'{name}_{label}.bvh')
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, clip in [('reference', ref), ('generated', gen)]:
        track, _ = reconstruct_root_track_from_375(clip)
        ax.plot(track[:, 0], track[:, 2], label=label)
    ax.set(xlabel='World x (m)', ylabel='World z (m)', title=name, aspect='equal')
    ax.legend(); fig.tight_layout(); fig.savefig(directory / f'{name}_root.png'); plt.close(fig)

