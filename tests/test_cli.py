"""Tests for graphrelax.cli module."""

import pytest

from graphrelax.cli import create_parser


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "graphrelax"

    def test_required_arguments(self):
        """Test that input and output are required."""
        parser = create_parser()

        # Should fail without required args
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_input_argument(self, tmp_path):
        """Test input argument parsing."""
        parser = create_parser()
        input_file = tmp_path / "input.pdb"
        output_file = tmp_path / "output.pdb"

        args = parser.parse_args(
            ["-i", str(input_file), "-o", str(output_file)]
        )
        assert args.input == input_file
        assert args.output == output_file

    def test_input_long_form(self, tmp_path):
        """Test input argument long form."""
        parser = create_parser()
        input_file = tmp_path / "input.pdb"
        output_file = tmp_path / "output.pdb"

        args = parser.parse_args(
            ["--input", str(input_file), "--output", str(output_file)]
        )
        assert args.input == input_file
        assert args.output == output_file


class TestModeSelection:
    """Tests for mode selection arguments."""

    def test_default_mode(self, tmp_path):
        """Test that default mode is relax."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )

        # All mode flags should be False by default (relax is implicit)
        assert args.relax is False
        assert args.repack_only is False
        assert args.no_repack is False
        assert args.design is False
        assert args.design_only is False

    def test_relax_flag(self, tmp_path):
        """Test --relax flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--relax",
            ]
        )
        assert args.relax is True

    def test_repack_only_flag(self, tmp_path):
        """Test --repack-only flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--repack-only",
            ]
        )
        assert args.repack_only is True

    def test_no_repack_flag(self, tmp_path):
        """Test --no-repack flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--no-repack",
            ]
        )
        assert args.no_repack is True

    def test_design_flag(self, tmp_path):
        """Test --design flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--design",
            ]
        )
        assert args.design is True

    def test_design_only_flag(self, tmp_path):
        """Test --design-only flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--design-only",
            ]
        )
        assert args.design_only is True

    def test_mutually_exclusive_modes(self, tmp_path):
        """Test that mode flags are mutually exclusive."""
        parser = create_parser()

        # Should fail with multiple mode flags
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "-i",
                    str(tmp_path / "in.pdb"),
                    "-o",
                    str(tmp_path / "out.pdb"),
                    "--relax",
                    "--design",
                ]
            )


class TestIterationAndOutputControl:
    """Tests for iteration and output control arguments."""

    def test_default_n_iter(self, tmp_path):
        """Test default n_iter value."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.n_iter == 5

    def test_custom_n_iter(self, tmp_path):
        """Test custom n_iter value."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--n-iter",
                "10",
            ]
        )
        assert args.n_iter == 10

    def test_default_n_outputs(self, tmp_path):
        """Test default n_outputs value."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.n_outputs == 1

    def test_custom_n_outputs(self, tmp_path):
        """Test custom n_outputs value."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "-n",
                "5",
            ]
        )
        assert args.n_outputs == 5

    def test_n_outputs_long_form(self, tmp_path):
        """Test --n-outputs long form."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--n-outputs",
                "3",
            ]
        )
        assert args.n_outputs == 3


class TestDesignOptions:
    """Tests for design option arguments."""

    def test_default_resfile(self, tmp_path):
        """Test default resfile is None."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.resfile is None

    def test_resfile_argument(self, tmp_path):
        """Test resfile argument."""
        parser = create_parser()
        resfile = tmp_path / "design.resfile"
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--resfile",
                str(resfile),
            ]
        )
        assert args.resfile == resfile

    def test_default_tsr_k(self, tmp_path):
        """Test default tsr_k value."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.tsr_k == 4.0

    def test_custom_tsr_k(self, tmp_path):
        """Test custom tsr_k value."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--tsr-k",
                "0.5",
            ]
        )
        assert args.tsr_k == 0.5

    def test_sampling_schedule(self, tmp_path):
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--tsr-sigma",
                "8",
                "--recycle-strategy",
                "medium/front_loaded",
            ]
        )
        assert args.tsr_sigma == 8
        assert args.recycle_strategy == "medium/front_loaded"


class TestRelaxationOptions:
    """Tests for relaxation option arguments."""

    def test_default_stiffness(self, tmp_path):
        """Test default stiffness value."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.stiffness == 10.0

    def test_custom_stiffness(self, tmp_path):
        """Test custom stiffness value."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--stiffness",
                "5.0",
            ]
        )
        assert args.stiffness == 5.0

    def test_default_max_iterations(self, tmp_path):
        """Test default max_iterations value."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.max_iterations == 0


class TestScoringOptions:
    """Tests for scoring option arguments."""

    def test_default_scorefile(self, tmp_path):
        """Test default scorefile is None."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.scorefile is None

    def test_scorefile_argument(self, tmp_path):
        """Test scorefile argument."""
        parser = create_parser()
        scorefile = tmp_path / "scores.sc"
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--scorefile",
                str(scorefile),
            ]
        )
        assert args.scorefile == scorefile


class TestGeneralOptions:
    """Tests for general option arguments."""

    def test_default_verbose(self, tmp_path):
        """Test default verbose is False."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.verbose is False

    def test_verbose_flag(self, tmp_path):
        """Test -v/--verbose flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "-v",
            ]
        )
        assert args.verbose is True

    def test_verbose_long_form(self, tmp_path):
        """Test --verbose long form."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--verbose",
            ]
        )
        assert args.verbose is True

    def test_default_seed(self, tmp_path):
        """Test default seed is None."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", str(tmp_path / "in.pdb"), "-o", str(tmp_path / "out.pdb")]
        )
        assert args.seed is None

    def test_seed_argument(self, tmp_path):
        """Test seed argument."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                str(tmp_path / "in.pdb"),
                "-o",
                str(tmp_path / "out.pdb"),
                "--seed",
                "42",
            ]
        )
        assert args.seed == 42
