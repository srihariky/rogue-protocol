import pandas as pd
import numpy as np
import generate_submission
from check_submission import check

test_blend = generate_submission.test_blend
test_df = generate_submission.test

# Exact threshold for 718 faults
target_th = 0.3310
test_verdict_718 = (test_blend >= target_th).astype(int)

sub_718 = pd.DataFrame({'log_id': test_df['log_id'], 'is_true_fault': test_verdict_718})
sub_718.to_csv('submission_718.csv', index=False)

print(f"Generated submission_718.csv with {test_verdict_718.sum()} true faults.")
check("submission_718.csv")

