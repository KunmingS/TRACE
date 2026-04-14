import numpy as np
import pytest

from trace_tad.cores.eval_engine import _jsonify_metrics


def test_jsonify_metrics_converts_nested_numpy_values():
    metrics = {
        "average_mAP": np.float32(0.9722),
        "per_class": np.array([0.9712, 0.9891], dtype=np.float64),
        "details": {
            "support": np.int64(4),
            "labels": ["drink", np.array([1, 2, 3], dtype=np.int32)],
        },
    }

    converted = _jsonify_metrics(metrics)

    assert converted["average_mAP"] == pytest.approx(0.9722, rel=1e-5)
    assert converted["per_class"] == pytest.approx([0.9712, 0.9891], rel=1e-6)
    assert converted["details"] == {
        "support": 4,
        "labels": ["drink", [1, 2, 3]],
    }
