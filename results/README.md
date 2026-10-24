# results/

Outputs of `phonetic_guided_tts.ipynb` (final run, Colab T4).

| File | Contents |
|---|---|
| `figures/` | WER chart, intelligibility vs naturalness, spectrograms, duration MAE, fine-tuning curve |
| `wer.csv`, `wer_paired.csv`, `transcripts.csv` | Whisper WER with 95% CIs, paired tests, transcripts per system |
| `utmos.csv`, `utmos_paired.csv`, `utmos_per_sentence.csv` | automatic naturalness (UTMOS) |
| `duration_metrics.csv`, `duration_feature_importance.csv`, `duration_comparison.csv` | duration predictor |
| `pause_metrics.csv` | pause classifier vs punctuation rule |
| `finetune_log.csv` | decoder fine-tuning, dev mel L1 per epoch |
| `split.csv`, `speechbrain_heldout.csv`, `eval_utts_clean.csv` | our split, SpeechBrain's held-out set, the 100 leak-free eval sentences |
| `listening_test_*` | human A/B test: anonymised responses, answer key, per-judgement table, summary |

In `listening_test_key.csv`, `fastspeech2` is the neural system before the input-format fix and
`baseline_pred` is the averaged-mel baseline with LightGBM durations.
