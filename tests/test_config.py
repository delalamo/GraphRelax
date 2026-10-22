"""Tests for graphrelax.config module."""

from pathlib import Path

from graphrelax.config import (
    DesignConfig,
    PipelineConfig,
    PipelineMode,
    RelaxConfig,
)


class TestPipelineMode:
    """Tests for PipelineMode enum."""

    def test_all_modes_exist(self):
        """Test that all expected modes are defined."""
        assert PipelineMode.RELAX.value == "relax"
        assert PipelineMode.REPACK_ONLY.value == "repack_only"
        assert PipelineMode.NO_REPACK.value == "no_repack"
        assert PipelineMode.DESIGN.value == "design"
        assert PipelineMode.DESIGN_ONLY.value == "design_only"

    def test_mode_count(self):
        """Test that we have exactly 5 modes."""
        assert len(PipelineMode) == 5


class TestDesignConfig:
    """Tests for DesignConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DesignConfig()
        assert config.tsr_k == 4.0
        assert config.tsr_sigma == 12.0
        assert config.recycle_strategy == "medium/flat"
        assert config.seed is None

    def test_custom_values(self):
        config = DesignConfig(
            tsr_k=2.0,
            tsr_sigma=8.0,
            recycle_strategy="medium/front_loaded",
            seed=42,
        )
        assert config.tsr_k == 2.0
        assert config.tsr_sigma == 8.0
        assert config.recycle_strategy == "medium/front_loaded"
        assert config.seed == 42


class TestRelaxConfig:
    """Tests for RelaxConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RelaxConfig()
        assert config.max_iterations == 0
        assert config.tolerance == 2.39
        assert config.stiffness == 10.0
        assert config.max_outer_iterations == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RelaxConfig(
            max_iterations=1000,
            tolerance=1.0,
            stiffness=5.0,
            max_outer_iterations=5,
        )
        assert config.max_iterations == 1000
        assert config.tolerance == 1.0
        assert config.stiffness == 5.0
        assert config.max_outer_iterations == 5


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.mode == PipelineMode.RELAX
        assert config.n_iterations == 5
        assert config.n_outputs == 1
        assert config.scorefile is None
        assert config.verbose is False
        assert isinstance(config.design, DesignConfig)
        assert isinstance(config.relax, RelaxConfig)

    def test_custom_mode(self):
        """Test setting different modes."""
        for mode in PipelineMode:
            config = PipelineConfig(mode=mode)
            assert config.mode == mode

    def test_with_scorefile(self):
        """Test setting scorefile path."""
        config = PipelineConfig(scorefile=Path("/tmp/scores.sc"))
        assert config.scorefile == Path("/tmp/scores.sc")

    def test_nested_configs(self):
        """Test nested config objects."""
        config = PipelineConfig(
            design=DesignConfig(tsr_k=2.0),
            relax=RelaxConfig(stiffness=20.0),
        )
        assert config.design.tsr_k == 2.0
        assert config.relax.stiffness == 20.0

    def test_verbose_flag(self):
        """Test verbose flag setting."""
        config = PipelineConfig(verbose=True)
        assert config.verbose is True
