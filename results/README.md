# results/

| File | Contents |
|---|---|
| `listening_test_responses_anonymised.csv` | human A/B test responses (emails and timestamps removed) |
| `listening_test_key.csv` | which system was Voice A / Voice B for each sentence |
| `listening_test_judgements.csv`, `listening_test_summary.json` | analysis of the human test |

After running the notebook, copy the contents of its `outputs/results/` folder here
(WER, UTMOS, duration and pause metrics, splits, transcripts, figures).

In `listening_test_key.csv`, `fastspeech2` is the neural system before the input-format fix and
`baseline_pred` is the averaged-mel baseline with LightGBM durations.
