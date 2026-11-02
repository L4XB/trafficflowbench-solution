"""Add the topology's ramp-attachment flags and mainline neighbours to each panel's meta.json.

Run: python -m tfb.cache.patch_meta  (after tfb.cache.build)
"""

import json

import pandas as pd

from tfb.common import CACHE, PANELS, RELEASE


def main() -> None:
    for panel in PANELS:
        meta = json.loads((CACHE / panel / "meta.json").read_text())
        topo = (
            pd.read_csv(RELEASE / "corridors" / panel / "network" / "lwr_mainline_topology.csv", dtype=str)
            .set_index("link_id")
            .loc[meta["links"]]
        )
        meta["params"]["on_attached"] = (topo.on_ramp_link_ids.fillna("").str.len() > 0).tolist()
        meta["params"]["off_attached"] = (topo.off_ramp_link_ids.fillna("").str.len() > 0).tolist()
        # incoming / outgoing mainline neighbours by index (for spatial models)
        lidx = {l: i for i, l in enumerate(meta["links"])}
        meta["incoming"] = [[lidx[x] for x in str(v).split(";") if x in lidx] for v in topo.incoming_link_ids.fillna("")]
        meta["outgoing"] = [[lidx[x] for x in str(v).split(";") if x in lidx] for v in topo.outgoing_link_ids.fillna("")]
        (CACHE / panel / "meta.json").write_text(json.dumps(meta))
        # consistency: attachment-map links vs topology attachment
        rl = meta["ramps"]["ramp_link"]
        rt = meta["ramps"]["ramp_type"]
        on_map = {l for l, t in zip(rl, rt) if t == "OR"}
        off_map = {l for l, t in zip(rl, rt) if t == "FR"}
        on_topo = {i for i, a in enumerate(meta["params"]["on_attached"]) if a}
        off_topo = {i for i, a in enumerate(meta["params"]["off_attached"]) if a}
        print(
            panel,
            "on: map",
            len(on_map),
            "topo",
            len(on_topo),
            "both",
            len(on_map & on_topo),
            "| off: map",
            len(off_map),
            "topo",
            len(off_topo),
            "both",
            len(off_map & off_topo),
            "| unmapped ramps",
            sum(1 for x in rl if x < 0),
        )


if __name__ == "__main__":
    main()
