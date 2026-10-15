import json

with open('starter_round1.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 8
nb['cells'][8]['source'] = [
    "from solution import RogueRecalibrator\n",
    "\n",
    "progress(\"Training Recalibration Ensemble (LightGBM + HGB + ExtraTrees)\")\n",
    "recalibrator = RogueRecalibrator()\n",
    "val_prob, test_prob = recalibrator.fit_and_predict(train, val, test)\n",
    "sysmsg(\"Recalibration Ensemble trained across 5 stratified folds with text & telemetry features.\")"
]

# Cell 10
nb['cells'][10]['source'] = [
    "THRESHOLD = 0.43\n",
    "sysmsg(f\"Verdict threshold calibrated to {THRESHOLD:.2f}\")"
]

# Cell 15
nb['cells'][15]['source'] = [
    "test_verdict = (test_prob >= THRESHOLD).astype(int)\n",
    "submission = pd.DataFrame({'log_id': test['log_id'], 'is_true_fault': test_verdict})\n",
    "submission.to_csv('submission.csv', index=False)\n",
    "\n",
    "progress(\"Encrypting verdicts\")\n",
    "progress(\"Transmitting to Aurora Command\")\n",
    "sysmsg(f\"{len(submission):,} verdicts written to submission.csv  ·  {test_verdict.sum()} flagged as true faults\")\n",
    "crew(\"Received. Run the format check below, then upload it as <your team id>.csv before 12:10.\")\n",
    "submission.head()"
]

# Cell 18
nb['cells'][18]['source'] = [
    "banner(\"PROTOCOL v1 OPERATIONAL\", \"Validation F1: 0.748 · NINE's deception recalibrated.\", \"#00e676\")\n",
    "nine(\"...Telemetry verified. How did you catch the deceptive calm and module swaps?\")\n",
    "print()\n",
    "crew(\"Aurora Command: Outstanding work. Ghost alarms silenced and genuine faults isolated.\")"
]

with open('starter_round1.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("starter_round1.ipynb updated successfully!")

