#!/usr/bin/env bash
# The organizers' Task-2 check, run locally: for each cut time, a copy of the release without any row of the split stamped after the
# cut is pushed through the Task-2 pipeline (run_all.sh with TASK2_ONLY=1) in a scratch root that shares code and models, and the
# predictions of every window with origin <= cut are compared with the full run. Expected output: "differing cells 0" for every cut.
#   bash tools/causality_test.sh                                  # default cuts below (private split)
#   SPLIT=validation CUTS="2031-03-04T07:00:00Z" bash tools/causality_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
FULL=$PWD
PY=$(command -v "${PY:-python}")
SPLIT=${SPLIT:-private}
CUTS=${CUTS:-"2031-04-01T19:05:00Z 2031-04-02T07:30:00Z@delete 2031-04-03T09:20:00Z 2031-04-05T16:05:00Z 2031-04-07T14:00:00Z 2031-04-17T15:15:00Z"}   # @delete: later day files removed instead of emptied
WORK=${WORK:-$FULL/causality_runs}
test -f $FULL/results_top3/queue/pred_A.csv || { echo "run run_all.sh first"; exit 1; }
for spec in $CUTS; do
  cut=${spec%@*}; mode=empty; [ "$spec" != "$cut" ] && mode=${spec#*@}
  R=$WORK/cut_${cut//:/}_$mode
  mkdir -p $R/data $R/results_top3/state $R/results_top3/queue $R/cache_top3
  # the code is COPIED: its project root comes from Path(__file__).resolve(), so a symlinked tfb/ would read the full release
  [ -L $R/tfb ] && rm $R/tfb
  mkdir -p $R/tfb && cp -R $FULL/tfb/. $R/tfb/ && cp $FULL/run_all.sh $R/run_all.sh
  for d in $FULL/results_top3/state/*/; do n=$(basename $d)
    case $n in unet_*|bo_long*|fc_fold*|bo_stack_*) ln -sfn $d $R/results_top3/state/$n ;; esac; done
  for d in $FULL/results_top3/queue/final_*/; do n=$(basename $d)
    case $n in *_e[0-3]) continue ;; esac
    mkdir -p $R/results_top3/queue/$n
    for f in $d*.txt $d/meta.json; do [ -e "$f" ] && ln -sfn "$f" $R/results_top3/queue/$n/; done
  done
  ln -sfn $FULL/results_top3/queue/windows_train_mix.csv $R/results_top3/queue/windows_train_mix.csv
  ln -sfn $FULL/results_top3/queue/snap_blocks_v6_mix.json $R/results_top3/queue/snap_blocks_v6_mix.json
  for d in $FULL/cache_top3/*/; do n=$(basename $d); mkdir -p $R/cache_top3/$n
    for f in train_queued_mix.npy occ_density_ratio.npy; do [ -e $d$f ] && ln -sfn $d$f $R/cache_top3/$n/$f; done; done
  $PY tools/truncate_release.py --src $FULL/data/kaggle_public --dst $R/data/kaggle_public --split $SPLIT --cut $cut --mode $mode
  (cd $R && TASK2_ONLY=1 PY=$PY bash run_all.sh > run.log 2>&1)
  nw=$($PY -c "import pandas as pd; print(pd.read_csv('$R/results_top3/queue/pred_A.csv').window_id.nunique())")
  echo "cut $cut $mode: truncated run predicts $nw windows (full run: 160)"
  for v in A B; do
    $PY tools/compare_queue.py --full $FULL/results_top3/queue/pred_$v.csv --trunc $R/results_top3/queue/pred_$v.csv --cut "$cut $mode ($v)"
  done
done
