"""The prompt regression test. It calls a real model, so it only runs when asked for:

    uv run pytest -m live --provider local                      # the shipped prompt on SmolLM3
    uv run pytest -m live --provider claude-haiku               # the same on Claude Haiku 5.5
    uv run pytest -m live --provider local --prompt helpful-v1b # a prompt edit that should fail

The whole eval set runs once; each gated metric is then its own test, so a failure names the
metric, the baseline, the new value and the allowed margin.
"""

import pytest
import yaml

from bankrag.config import GATES_FILE
from bankrag.evaluate import check_against_baseline, run_eval

pytestmark = pytest.mark.live

METRICS = list(yaml.safe_load(GATES_FILE.read_text(encoding="utf-8"))["metrics"])


@pytest.fixture(scope="module")
def gate_results(request, index):
    provider = request.config.getoption("--provider")
    prompt = request.config.getoption("--prompt")
    run = run_eval(provider, prompt, index=index, progress=False)
    print(f"\nrun saved to {run.path}")
    return {r.metric: r for r in check_against_baseline(provider, run.summary)}


@pytest.mark.parametrize("metric", METRICS)
def test_metric_within_margin_of_baseline(gate_results, metric):
    result = gate_results[metric]
    assert result.passed, result.describe()
