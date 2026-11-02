"""Export the onset site-block catalogue (train fit windows, mix labels) used by tfb.decode.onset_snap to a small JSON file.

The catalogue maps panel -> list of [block (link indices), count]; tfb.decode.onset_snap --blocks <name>.json reads it instead of the
177 MB onset feature cache, with identical results.
Run: python -m tfb.decode.export_snap_blocks --out results_top3/queue/snap_blocks_v6_mix.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfb.decode.onset_snap import site_blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-file", default="feat_onset_fit_all_v6_mix_fillcausal_nodrop.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blocks = site_blocks(a.feat_file)
    data = {pn: [[list(b), int(c)] for b, c in cnt.items()] for pn, cnt in blocks.items()}
    Path(a.out).write_text(json.dumps(data))
    print({pn: len(v) for pn, v in data.items()})


if __name__ == "__main__":
    main()
