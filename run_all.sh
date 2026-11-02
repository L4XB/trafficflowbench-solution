#!/usr/bin/env bash
# TrafficFlowBench -- single entry script: released data + shipped models -> the two selected final submissions.
#
#   bash run_all.sh                 # CPU, ~1.5-3 h on 8 cores (see README for the measured runtime)
#   DEVICE=cuda bash run_all.sh     # or mps; any torch device for the U-Net inference
#
# Inputs : data/kaggle_public/            the Kaggle release (unzipped, layout of the Data page)
#          results_top3/                  shipped trained models (U-Nets, LightGBM boosters, blackout stackers) and the
#          cache_top3/<panel>/train_queued_mix.npy + results_top3/queue/windows_train_mix.csv (train-derived labels and windows)
# Outputs: submission/submission_A.csv    final A (onset: eiou -> live gate -> site-block snap; ongoing: 10-member blend)
#          submission/submission_B.csv    final B (= A + rolling in-month recurrence for the April windows)
# Every step below only reads the release and the shipped models; Task-2 features use data stamped <= each window's T.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"                 # the code is the package tfb/ in this directory (python -m tfb...)
PY=$(command -v "${PY:-python}")          # absolute: some steps run from results_top3/queue
DEV=${DEVICE:-cpu}
Q=results_top3/queue; S=results_top3/state
QP="D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
test -f data/kaggle_public/sample_submission.csv || { echo "missing data/kaggle_public (the Kaggle release)"; exit 1; }

log "1/7 array cache from the release"
$PY -m tfb.cache.build --workers ${WORKERS:-1}   # panels in parallel; a large panel needs up to ~8 GB RAM
$PY -m tfb.cache.patch_meta

log "2/7 Task 1 state: U-Net ensembles, the blackout corrector and the free-flow level correction"
U=$S/unet_h31_long
T2=${TASK2_ONLY:-0}                         # 1: only what Task 2 needs (tools/causality_test.sh)
[ $T2 = 1 ] || $PY -m tfb.state.predict --name enslongF --ckpt ${U}46/best.pt ${U}48/best.pt ${U}49/best.pt ${U}50/best.pt ${U}51/best.pt ${U}52/best.pt --splits validation private --device $DEV
$PY -m tfb.state.predict --name enslongE --ckpt ${U}46/best.pt ${U}48/best.pt ${U}49/best.pt ${U}50/best.pt ${U}51/best.pt --splits validation private --panels $QP --device $DEV
$PY -m tfb.state.predict --name enslongC --ckpt ${U}46/best.pt ${U}48/best.pt ${U}49/best.pt --splits validation private --panels $QP --device $DEV
[ $T2 = 1 ] || $PY -m tfb.state.predict --name pred_spec2 --ckpt $S/bo_long46/last.pt $S/bo_long48/last.pt --splits validation private --panels $QP --device $DEV
[ $T2 = 1 ] || $PY -m tfb.state.blackout_apply --base enslongF --feed enslongF --corr bo_stack_ensF2_bB_all --out enslongF_bo2b --w 1.0
[ $T2 = 1 ] || $PY -m tfb.state.blackout_apply --base enslongF_bo2b --feed pred_spec2 --corr bo_stack_spec2_b_all --out enslongF_bo2bs --w 0.5
[ $T2 = 1 ] || $PY -m tfb.state.level_apply --base enslongF_bo2bs --out enslongF_bo2bs_lvl   # free-flow level correction

log "3/7 Task 2 causal inputs: per-window U-Net fills and 30-min forecasts with everything after T hidden"
for sp in validation private; do
  $PY -m tfb.queue.fill_causal --split $sp --device $DEV
  for k in 0 1 2 3; do
    ck=$S/fc_fold$k/best.pt; [ $k = 0 ] && ck=$S/fc_fold0b/best.pt
    $PY -m tfb.queue.fill_causal --split $sp --device $DEV --suffix _fc$k --ckpts $ck
    $PY -m tfb.queue.fill_causal --split $sp --device $DEV --suffix _fc${k}l1 --ckpts $S/fc_fold${k}l1/best.pt
    $PY -m tfb.queue.fill_qh --split $sp --device $DEV --suffix _fc${k}qf --ckpts $S/fc_fold${k}qf/best.pt
  done
done

log "4/7 Task 2 members from the saved LightGBM boosters"
unset QUEUE_LOMC QUEUE_FILL_TRAIN INMONTH_CAP QUEUE_DIRFIX QUEUE_INMH QUEUE_FC_FEATS QUEUE_FC QUEUE_FC_EVAL QUEUE_CAUSAL_PREFIX \
      QUEUE_INM_ROLL QUEUE_CLIM_PRIV QUEUE_ROBUST QUEUE_PERPANEL QUEUE_ADAPT QUEUE_WINWEIGHT QUEUE_PERPANEL_ONSET
export QUEUE_FILL=causal QUEUE_NODROP=1 QUEUE_INFER=1
member() {   # model_dir run_name args...   (environment from the caller)
  local src=$1 dst=$2; shift 2
  mkdir -p $Q/final_$dst
  [ "$src" = "$dst" ] || for f in $Q/final_$src/*.txt; do ln -sf "$PWD/$f" $Q/final_$dst/; done
  $PY -m tfb.queue.final --name $dst --reuse "$@" > $Q/final_$dst.log 2>&1
}
family() {   # model_dir FC EVAL_SUFFIX args...: the four fold forecasters' evaluation inputs, probabilities averaged
  local m=$1 fc=$2 es=$3 k; shift 3
  for k in 0 1 2 3; do QUEUE_FC=$fc QUEUE_FC_EVAL=_fc$k$es member $m ${m}_e$k "$@"; done
  $PY -m tfb.queue.probs_avg --out $Q/final_$m/probs_avg.npz $Q/final_${m}_e0/probs.npz $Q/final_${m}_e1/probs.npz $Q/final_${m}_e2/probs.npz $Q/final_${m}_e3/probs.npz
}
A2="--feats v6 --truth mix --seeds 2 --noclim-w-ongoing 1.0 --ongoing-mode thr --ongoing-thr 0.45 --ongoing-leaves 511 --ongoing-min-leaf 40 --ongoing-rounds 800 --dump-probs --conds ongoing"
A4="--feats v6 --truth mix --seeds 4 --noclim-w-ongoing 1.0 --ongoing-mode thr --ongoing-thr 0.45 --ongoing-leaves 511 --ongoing-min-leaf 40 --ongoing-rounds 800 --dump-probs --conds ongoing --ongoing-obs-gap0"
A28="--feats v6 --truth mix --seeds 2 --noclim-w-ongoing 1.0 --onset-mode thr --onset-thr 0.3 --ongoing-mode thr --ongoing-thr 0.45 --ongoing-leaves 511 --ongoing-min-leaf 40 --ongoing-rounds 800 --dump-probs"
V2="--feats v2 --truth mix --seeds 2 --noclim-w-ongoing 1.0 --ongoing-mode thr --ongoing-thr 0.45 --ongoing-leaves 511 --ongoing-min-leaf 40 --ongoing-rounds 800 --dump-probs --conds ongoing"
export QUEUE_FILL_STATE=enslongE
member v27mixcausal v27mixcausal --feats v6 --truth mix --seeds 2 --onset-hedge --noclim-w-onset 0.5 --noclim-w-ongoing 1.0 --onset-mode thr \
  --onset-thr 0.3 --ongoing-mode thr --ongoing-thr 0.5 --ongoing-leaves 511 --ongoing-min-leaf 40 --ongoing-rounds 800 --dump-probs
export QUEUE_ROBUST=3; member v27live v27live --feats v6 --truth mix --seeds 2 --noclim-w-onset 1.0 --onset-mode thr --onset-thr 0.3 --dump-probs --conds onset
unset QUEUE_ROBUST
member v2og v2og $V2
export QUEUE_PERPANEL=1                     # per-panel no-climatology models
member APnc APnc $A2
member V2Pnc V2Pnc $V2
unset QUEUE_PERPANEL
export QUEUE_FILL_STATE=enslongC           # the forecast families were fitted with the 3-member fill state
family v28fc _fcoof "" $A28
family v31qf _fcqfoof qf $A4
family Dnc _fcl1oof l1 $A2
export QUEUE_PERPANEL=1
family BPnc _fcoof "" $A2
family CPqf _fcqfoof qf $A4
family DPnc _fcl1oof l1 $A2
unset QUEUE_PERPANEL

# final B: the eight members with in-month features re-evaluated for the private split with rolling recurrence (an April window on
# day d sees the 16 days before d, the missing ones from the end of March, which precedes every April window)
export QUEUE_INM_ROLL=eval
RE="--splits private --conds ongoing"
family_re() {   # model_dir FC EVAL_SUFFIX args...
  local m=$1 fc=$2 es=$3 k; shift 3
  for k in 0 1 2 3; do QUEUE_FC=$fc QUEUE_FC_EVAL=_fc$k$es member $m ${m}_re_e$k "$@" $RE; done
  mkdir -p $Q/final_${m}_re
  $PY -m tfb.queue.probs_avg --out $Q/final_${m}_re/probs_avg.npz $Q/final_${m}_re_e0/probs.npz $Q/final_${m}_re_e1/probs.npz \
    $Q/final_${m}_re_e2/probs.npz $Q/final_${m}_re_e3/probs.npz
}
export QUEUE_FILL_STATE=enslongE
member v27mixcausal v27mixcausal_re --feats v6 --truth mix --seeds 2 --onset-hedge --noclim-w-onset 0.5 --noclim-w-ongoing 1.0 \
  --onset-mode thr --onset-thr 0.3 --ongoing-mode thr --ongoing-thr 0.5 --ongoing-leaves 511 --ongoing-min-leaf 40 --ongoing-rounds 800 \
  --dump-probs $RE
export QUEUE_PERPANEL=1; member APnc APnc_re $A2 $RE; unset QUEUE_PERPANEL
export QUEUE_FILL_STATE=enslongC
family_re v28fc _fcoof "" $A28
family_re v31qf _fcqfoof qf $A4
family_re Dnc _fcl1oof l1 $A2
export QUEUE_PERPANEL=1
family_re BPnc _fcoof "" $A2
family_re CPqf _fcqfoof qf $A4
family_re DPnc _fcl1oof l1 $A2
unset QUEUE_PERPANEL QUEUE_INM_ROLL

log "5/7 Task 2 ongoing blend and decoding (onset: eiou -> live gate -> site-block snap)"
$PY -m tfb.queue.probs_blend --out $Q/probs_blend10pp.npz $Q/final_v27mixcausal/probs.npz $Q/final_v28fc/probs_avg.npz \
  $Q/final_v31qf/probs_avg.npz $Q/final_Dnc/probs_avg.npz $Q/final_v2og/probs.npz $Q/final_APnc/probs.npz $Q/final_BPnc/probs_avg.npz \
  $Q/final_CPqf/probs_avg.npz $Q/final_DPnc/probs_avg.npz $Q/final_V2Pnc/probs.npz --weights 1 1 1 1 1 1 1 1 1 1
BLEND=probs_blend10pp.npz
$PY -m tfb.queue.probs_blend --out $Q/probs_blend10pp_re_priv.npz $Q/final_v27mixcausal_re/probs.npz $Q/final_v28fc_re/probs_avg.npz \
  $Q/final_v31qf_re/probs_avg.npz $Q/final_Dnc_re/probs_avg.npz $Q/final_v2og/probs.npz $Q/final_APnc_re/probs.npz \
  $Q/final_BPnc_re/probs_avg.npz $Q/final_CPqf_re/probs_avg.npz $Q/final_DPnc_re/probs_avg.npz $Q/final_V2Pnc/probs.npz --weights 1 1 1 1 1 1 1 1 1 1
$PY -m tfb.queue.probs_override --base $Q/probs_blend10pp.npz --override $Q/probs_blend10pp_re_priv.npz --out $Q/probs_blend10pp_B.npz
export QUEUE_FILL_STATE=enslongE
$PY -c "import pandas as pd; t = pd.read_csv('$Q/final_v27mixcausal/pred.csv'); t['queue_pred'] = 0; t.to_csv('$Q/pred_template.csv', index=False)"
for v in A B; do
  blend=$BLEND; [ $v = B ] && blend=probs_blend10pp_B.npz      # B: the ongoing blend with rolling recurrence for April
  (cd $Q
   $PY -m tfb.decode.onset_redecode --probs final_v27mixcausal/probs.npz --template pred_template.csv --mode eiou --out pred_e.csv
   $PY -m tfb.decode.onset_gate --template pred_e.csv --live final_v27live/probs.npz --mode ratio --tau 0.7 --redecode \
     --prod final_v27mixcausal/probs.npz --out pred_g$v.csv
   $PY -m tfb.decode.onset_snap --template pred_g$v.csv --prod final_v27mixcausal/probs.npz --blocks snap_blocks_v6_mix.json --out pred_s$v.csv
   $PY -m tfb.decode.ongoing_ratio_thr --probs $blend --template pred_s$v.csv --gamma=-0.25 --base 0.38 --lo 0.25 --hi 0.55 --out pred_r$v.csv
   $PY -m tfb.decode.ongoing_kind_thr --probs $blend --template pred_r$v.csv --kind bound --scale 1.5 --out pred_k$v.csv
   $PY -m tfb.decode.ongoing_kind_thr --probs $blend --template pred_k$v.csv --kind changepoint --scale 1.1 --out pred_$v.csv)
done

if [ $T2 = 1 ]; then log "Task-2 outputs ready: $Q/pred_A.csv, $Q/pred_B.csv"; exit 0; fi
log "6/7 Task 4 ODME: ridge NNLS toward the split's prior on its released counts (lambda 0.05)"
$PY -m tfb.odme.ridge --lam 0.05 --out results_top3/odme/ridge_lam0.05

log "7/7 assemble the two final submissions"
for v in A B; do
  $PY -m tfb.assemble --anchor data/kaggle_public/sample_submission.csv --out submission/submission_$v.csv --state-dir $S/enslongF_bo2bs_lvl \
    --queue $Q/pred_$v.csv --odme results_top3/odme/ridge_lam0.05/validation.csv results_top3/odme/ridge_lam0.05/private.csv
done
log "done: submission/submission_A.csv, submission/submission_B.csv"
