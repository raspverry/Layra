"""src/common/config.py 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import LayraConfig, Stage1Config, get_config


class TestLayraConfig:
    def test_default_construction(self) -> None:
        cfg = LayraConfig()
        assert cfg.stage1.resolution == 1280
        assert cfg.stage1.num_inference_steps == 30
        assert cfg.stage1.dtype == "bfloat16"
        assert cfg.stage1.max_layers == 23
        assert cfg.stage2.device == "mps"
        assert "CoreMLExecutionProvider" in cfg.stage3.providers

    def test_resolve_paths_makes_absolute(self) -> None:
        cfg = LayraConfig().resolve_paths()
        assert cfg.model_dir.is_absolute()
        assert cfg.output_dir.is_absolute()
        assert cfg.stage1.layerdiff_weights.is_absolute()
        assert cfg.stage2.sam3_weights.is_absolute()
        assert cfg.stage3.rife_model.is_absolute()

    def test_stage1_dtype_validation(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Stage1Config(dtype="fp64")  # type: ignore[arg-type]


class TestGetConfig:
    def test_cached(self) -> None:
        a = get_config()
        b = get_config()
        assert a is b

    def test_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # lru_cache를 우회하려면 함수 자체를 다시 호출해야 함.
        get_config.cache_clear()

        monkeypatch.setenv("LAYRA_MODEL_DIR", str(tmp_path / "custom_models"))
        monkeypatch.setenv("LAYRA_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("SAM3_MODEL_PATH", str(tmp_path / "sam3.pth"))

        cfg = get_config()
        assert cfg.model_dir == (tmp_path / "custom_models").resolve()
        assert cfg.log_level == "DEBUG"
        assert cfg.stage2.sam3_weights == (tmp_path / "sam3.pth").resolve()

        get_config.cache_clear()
