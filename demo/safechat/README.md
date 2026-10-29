# SafeChat scenario training

`scenarios_with_answers.csv` supplies the questions and answers. For each row,
use a nonblank `answer_ideal_agg`; otherwise use `answer_ideal`. The current CSV
contains 161 questions: 10 aggregate answers and 151 ideal answers.

`prepare_scenarios.py` updates the NLU examples, domain responses, and rules.
It preserves existing paraphrases and supporting chat behavior. Existing
summary/full-answer stories return the selected CSV answer on the first reply
as well as on a request for the full answer. Distinct questions with different
answers retain separate intents, including six-month and yearly subscriptions.
The importer rejects missing answers, duplicate questions, and intent collisions.

The NLU pipeline includes word-pair features to distinguish closely worded
questions (for example, whether a meeting occurred versus what time it occurred).
DIET trains for 200 epochs with a fixed seed; fallback thresholds are unchanged.

Training uses Rasa 3.6.20 with Python 3.8.20 in the existing `safechat` environment.
From this directory, with that environment activated:

```bash
python prepare_scenarios.py
python -m unittest discover -s tests -p 'test_*.py'
rasa data validate
RASA_TELEMETRY_ENABLED=false RASA_CACHE_DIRECTORY=/tmp/safechat-rasa-training-cache \
  rasa train --num-threads 2 --fixed-model-name scenarios-with-answers-20261029
python evaluate_scenarios.py --model models/scenarios-with-answers-20261029.tar.gz
```

The evaluator loads the trained model, asks every CSV question in a separate
conversation, and compares its actual reply with the selected answer. It writes
`results/scenarios_evaluation.json` and exits unsuccessfully if any answer differs.
This checks the supplied questions; it does not measure accuracy on unseen
paraphrases.

To serve the trained model through the existing REST adapter, run Rasa with
`--model models/scenarios-with-answers-20261029.tar.gz` and the existing server
options. A running Rasa process must load the new artifact to use the new answers.
The prior model archives remain available in `models/`.
