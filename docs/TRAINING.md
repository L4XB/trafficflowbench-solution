# Training every shipped model from the release

All models are fitted on the **train split only** (2030-06-01 .. 2031-02-28). Run every command from the repository root (`python -m`
puts it on the import path) with the same environment as `run_all.sh`. The 31 held-out train days (`holdout31`, used for model
selection and by the blackout stackers) are listed in `config/holdout31.json`. U-Nets were trained on an Apple M-series GPU
(`TRAIN_DEVICE=mps`; any torch device works) with a wall-clock budget (`--minutes`): after the first epoch the cosine schedule length
is fixed from the measured epoch time. The table lists the schedule length that resulted and the epoch of the shipped checkpoint
(`best.pt` = best EMA weights on the 31 held-out train days; the blackout specialists ship `last.pt`). To retrain on other hardware
use `--minutes 0 --epochs <schedule>`; results will differ slightly from the shipped weights (GPU non-determinism), which is why the
package ships them.

## 0. Cache
```bash
python -m tfb.cache.build --workers 1 && python -m tfb.cache.patch_meta
```

## 1. State U-Nets (`tfb/state/train.py`)
Common: `--w_phys 0.03 --state_loss mse --phys_loss l1` (defaults otherwise: widths 32 64 128 256, crop 64, batch 8, 250 steps/epoch,
AdamW lr 1e-3, wd 0.01, warmup 300, EMA 0.999, band probability 0.2). Folds: `fold<k>` = train days with index % 4 == k.

| model | seed | excluded days | budget (min) | schedule (epochs) | shipped epoch | other |
| --- | --- | --- | --- | --- | --- | --- |
| unet_h31_long46 | 46 | holdout31 | 300 | 382 | 355 | |
| unet_h31_long48 | 48 | holdout31 | 600 | 536 | 529 | |
| unet_h31_long49 | 49 | holdout31 | 720 | 500 | 499 | |
| unet_h31_long50 | 50 | holdout31 | 900 | 851 | 829 | |
| unet_h31_long51 | 51 | holdout31 | 900 | 1137 (stopped at the 1000-epoch cap) | 978 | |
| unet_h31_long52 | 52 | holdout31 | 900 | 1041 (stopped at 1000) | 986 | |
| unet_h31_fold0..3 | 60..63 | fold0..3 | 150 | 148 / 155 / 189 / 191 | 147 / 154 / 184 / 191 | bases of the forecasters |
| unet_s_v2, unet_s_v4_seed43, unet_s_v5_w05 | 42 / 43 / 44 | holdout31 + February | 40 | | 47 / 46 / 52 | label denoisers; v5: `--w_phys 0.05` |

```bash
python -m tfb.state.train --name unet_h31_long46 --seed 46 --exclude holdout31 --minutes 300 --w_phys 0.03 --state_loss mse --phys_loss l1
# ... likewise for 48-52 and the fold bases (--exclude fold<k> --minutes 150 --seed 60+k)
```

**Forecasters** (everything after a random origin hidden, `--forecast`, fine-tuned from the fold bases, `--warmup 200`):
| model | seed | resume from | lr | budget | schedule | shipped epoch | other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fc_fold0b, fc_fold1..3 | 62, 63..65 | unet_h31_fold<k> | 3e-4 | 120 | 146 / 153 / 148 / 144 | 129 / 153 / 135 / 129 | |
| fc_fold<k>l1 | 80..83 | fc_fold0b / fc_fold<k> | 1.5e-4 | 90 | 105 / 112 / 106 / 107 | 99 / 112 / 106 / 69 | `--state_loss l1` |
| fc_fold<k>qh | 72..75 | fc_fold0b / fc_fold<k> | 1.5e-4 | 120 | 144 / 148 / 146 / 153 | 141 / 135 / 146 / 153 | `tfb/state/train_qh.py --w_queue 0.25` (queue head) |
| fc_fold<k>qf | 90..93 | fc_fold<k>qh | 1e-4 | 90 | 113 / 113 / 110 / 113 | 113 / 105 / 110 / 99 | `tfb/state/train_qh.py --w_queue 2.0` (queue-focused; shipped, used by C and CP) |

**Blackout specialists** (bands of 18 steps right after a train window's origin, loss on every band cell):
```bash
python -m tfb.state.train --name bo_long46 --seed 146 --exclude holdout31 --minutes 40 --resume results_top3/state/unet_h31_long46/best.pt \
  --lr 2e-4 --warmup 200 --p_band 1.0 --band_windows results_top3/queue/windows_train_mix.csv --band_loss --w_phys 0.03 --state_loss mse --phys_loss l1
# bo_long48: --seed 148, resume unet_h31_long48 (schedules 43 / 50 epochs, shipped last.pt)
```

## 2. Queue labels and training windows
```bash
python -m tfb.labels.denoise                                   # train_speed_dn / train_flow_dn with the three unet_s_* denoisers
python -m tfb.labels.denoise --suffix dn2 --ckpts results_top3/state/unet_h31_long{46,48,49}/best.pt   # gap fields for train features
ORACLE_VERSIONS=mix0.7 python -m tfb.labels.label_write --version mix0.7 --name mix   # 0.7 x observation + 0.3 x denoised
python -m tfb.labels.build_windows --truth mix                   # selector replica on the mix labels (4 processes)
python -m tfb.decode.export_snap_blocks --out results_top3/queue/snap_blocks_v6_mix.json             # after the onset feature cache exists
```
`tfb/labels/build_windows.py` runs the selector replica (`tfb/labels/selector.py`) on the label field and writes
`cache_top3/<panel>/train_queued_mix.npy` and `results_top3/queue/windows_train_mix.csv` (6 755 windows).

## 3. Causal fills and forecasts of the train windows
```bash
python -m tfb.queue.fill_causal --split train                                   # long46/48/49, everything after T hidden
for k in 0 1 2 3; do
  python -m tfb.queue.fill_causal --split train --suffix _fc$k   --ckpts results_top3/state/fc_fold${k/0/0b}/best.pt
  python -m tfb.queue.fill_causal --split train --suffix _fc${k}l1 --ckpts results_top3/state/fc_fold${k}l1/best.pt
  python -m tfb.queue.fill_qh     --split train --suffix _fc${k}qh --ckpts results_top3/state/fc_fold${k}qh/best.pt
  python -m tfb.queue.fill_qh     --split train --suffix _fc${k}qf --ckpts results_top3/state/fc_fold${k}qf/best.pt
done
python -m tfb.queue.fc_oof                                                     # -> _fcoof (each train window from the fold model that never saw its day)
for pn in D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S; do for k in 0 1 2 3; do
  ln -sf train_causalfill_fc${k}qh.npz cache_top3/$pn/train_causalfill_qhf$k.npz; ln -sf train_causalfill_fc${k}l1.npz cache_top3/$pn/train_causalfill_l1f$k.npz;
  ln -sf train_causalfill_fc${k}qf.npz cache_top3/$pn/train_causalfill_qff$k.npz; done; done
python -m tfb.queue.fc_oof --prefix _qhf --out _fcqhoof
python -m tfb.queue.fc_oof --prefix _qff --out _fcqfoof
python -m tfb.queue.fc_oof --prefix _l1f --out _fcl1oof
```

## 4. LightGBM queue members
Exactly the `member` / `family` calls of `run_all.sh` **without** `--reuse` and without `QUEUE_INFER=1` (fits on all train windows,
seeds 42, 43, ...). Fill state of the evaluation splits: enslongE for v27mixcausal, v27live, APnc; enslongC for the forecast families.
Runtime on 10 CPU threads: 2--30 min per member.

## 5. Blackout stackers
```bash
python -m tfb.state.blackout_predict --ckpts results_top3/state/unet_h31_long{46,48,49,50,51,52}/best.pt --out bo_pred_ensF2
python -m tfb.state.blackout_predict --ckpts results_top3/state/bo_long46/last.pt results_top3/state/bo_long48/last.pt --out bo_pred_spec2
python -m tfb.state.blackout_stack --pred bo_pred_ensF2 --sub 1.0 --leaves 255 --rounds 1200 --all-days --drop pos panel --save bo_stack_ensF2_bB_all
python -m tfb.state.blackout_stack --pred bo_pred_spec2 --sub 1.0 --leaves 255 --rounds 1200 --all-days --drop pos panel --save bo_stack_spec2_b_all
```
