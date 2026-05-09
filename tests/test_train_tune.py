def test_config_merge_from_dict_supports_list_indices(tmp_path):
    from trace_tad.config import Config

    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text(
        "dataset=dict(train=dict(pipeline=[dict(type='PrepareVideoInfo'), dict(type='VideoInit', num_threads=4)]))\n",
        encoding="utf-8",
    )

    cfg = Config.fromfile(str(cfg_path))
    cfg.merge_from_dict({"dataset.train.pipeline.1.num_threads": 2})

    assert cfg.dataset.train.pipeline[1].num_threads == 2


def test_tune_train_resources_returns_profiles_and_fastest_recommendation(monkeypatch):
    import trace_tad.utils.train_tune as train_tune

    monkeypatch.setattr(train_tune.Config, "fromfile", lambda path: object())
    monkeypatch.setattr(train_tune, "_detect_cache_mode", lambda annotation_path: "cached_video")
    monkeypatch.setattr(train_tune.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(train_tune, "_total_ram_mb", lambda: 32768)

    timings = {"Low": 40.0, "Balanced": 20.0, "High": 30.0}

    def fake_benchmark(base_cfg, profile, **kwargs):
        return {
            "name": profile["name"],
            "num_workers": profile["num_workers"],
            "decode_threads": profile["decode_threads"],
            "prefetch_factor": profile["prefetch_factor"],
            "avg_batch_ms": timings[profile["name"]],
            "peak_rss_mb": 512,
            "status": "ok",
        }

    monkeypatch.setattr(train_tune, "benchmark_train_profile", fake_benchmark)

    result = train_tune.tune_train_resources(
        "configs/small.py",
        model_dir="/model",
        annotation_path="/model/dataset.json",
        class_map="/model/classmap.txt",
    )

    assert result["recommended_profile"] == "Balanced"
    assert [p["name"] for p in result["profiles"]] == ["Low", "Balanced", "High"]
    assert result["cache_mode"] == "cached_video"


def test_tune_train_resources_skips_profiles_over_resource_guardrails(monkeypatch):
    import trace_tad.utils.train_tune as train_tune

    monkeypatch.setattr(train_tune.Config, "fromfile", lambda path: object())
    monkeypatch.setattr(train_tune, "_detect_cache_mode", lambda annotation_path: "cached_video")
    monkeypatch.setattr(train_tune.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(train_tune, "_total_ram_mb", lambda: 32768)

    timings = {"Low": 50.0, "Balanced": 30.0, "High": 10.0}

    def fake_benchmark(base_cfg, profile, **kwargs):
        return {
            "name": profile["name"],
            "num_workers": profile["num_workers"],
            "decode_threads": profile["decode_threads"],
            "prefetch_factor": profile["prefetch_factor"],
            "avg_batch_ms": timings[profile["name"]],
            "peak_rss_mb": 512,
            "status": "ok",
        }

    monkeypatch.setattr(train_tune, "benchmark_train_profile", fake_benchmark)

    result = train_tune.tune_train_resources(
        "configs/small.py",
        model_dir="/model",
        annotation_path="/model/dataset.json",
        class_map="/model/classmap.txt",
    )

    assert result["recommended_profile"] == "Balanced"
    assert "Resource guardrail skipped High" in " ".join(result["notes"])


def test_tune_train_resources_falls_back_when_profiles_fail(monkeypatch):
    import trace_tad.utils.train_tune as train_tune

    monkeypatch.setattr(train_tune.Config, "fromfile", lambda path: object())
    monkeypatch.setattr(train_tune, "_detect_cache_mode", lambda annotation_path: "virtual")
    monkeypatch.setattr(train_tune.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(train_tune, "_total_ram_mb", lambda: 32768)

    def fake_benchmark(base_cfg, profile, **kwargs):
        return {
            "name": profile["name"],
            "num_workers": profile["num_workers"],
            "decode_threads": profile["decode_threads"],
            "prefetch_factor": profile["prefetch_factor"],
            "avg_batch_ms": None,
            "peak_rss_mb": 512,
            "status": "failed",
            "error": "boom",
        }

    monkeypatch.setattr(train_tune, "benchmark_train_profile", fake_benchmark)

    result = train_tune.tune_train_resources(
        "configs/small.py",
        model_dir="/model",
        annotation_path="/model/dataset.json",
        class_map="/model/classmap.txt",
    )

    assert result["recommended_profile"] == "Low"
    assert all(p["status"] == "failed" for p in result["profiles"])
    assert "falling back to Low" in " ".join(result["notes"])
