"""Tests for graphrelax.weights module.

Tests the weight downloading and path resolution functionality.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from graphrelax.weights import (
    BASE_URL,
    WEIGHT_FILES,
    _get_package_weights_dir,
    download_weights,
    ensure_weights,
    find_weights_dir,
    get_weights_dir,
    weights_exist,
)


class TestGetWeightsDir:
    """Tests for get_weights_dir function."""

    def test_default_location(self):
        """Test default weights directory is ~/.graphrelax/weights."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove GRAPHRELAX_WEIGHTS_DIR if set
            os.environ.pop("GRAPHRELAX_WEIGHTS_DIR", None)
            result = get_weights_dir()
            assert result == Path.home() / ".graphrelax" / "weights"

    def test_env_override(self, tmp_path):
        """Test GRAPHRELAX_WEIGHTS_DIR environment variable override."""
        custom_dir = tmp_path / "custom_weights"
        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(custom_dir)}
        ):
            result = get_weights_dir()
            assert result == custom_dir

    def test_env_override_empty_string(self):
        """Test empty env var falls back to default."""
        with patch.dict(os.environ, {"GRAPHRELAX_WEIGHTS_DIR": ""}):
            result = get_weights_dir()
            # Empty string is falsy, so should use default
            assert result == Path.home() / ".graphrelax" / "weights"


class TestGetPackageWeightsDir:
    """Tests for _get_package_weights_dir function."""

    def test_returns_package_path(self):
        """Test that package weights dir is relative to weights.py."""
        result = _get_package_weights_dir()
        # Should be in src/graphrelax/LigandMPNN/model_params
        assert result.name == "model_params"
        assert result.parent.name == "LigandMPNN"


class TestFindWeightsDir:
    """Tests for find_weights_dir function."""

    def test_finds_user_dir_when_weights_exist(self, tmp_path):
        """Test that user directory is preferred when weights exist there."""
        user_dir = tmp_path / "user_weights"
        user_dir.mkdir(parents=True)

        # Create all weight files in user dir
        for f in WEIGHT_FILES:
            (user_dir / f).touch()

        with patch.dict(os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(user_dir)}):
            result = find_weights_dir()
            assert result == user_dir

    def test_finds_package_dir_when_user_empty(self, tmp_path):
        """Test fallback to package dir when user dir is empty."""
        user_dir = tmp_path / "empty_user_weights"
        user_dir.mkdir(parents=True)

        package_dir = tmp_path / "package_weights"
        package_dir.mkdir(parents=True)
        for f in WEIGHT_FILES:
            (package_dir / f).touch()

        with patch.dict(os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(user_dir)}):
            with patch(
                "graphrelax.weights._get_package_weights_dir",
                return_value=package_dir,
            ):
                result = find_weights_dir()
                assert result == package_dir

    def test_returns_user_dir_when_no_weights_anywhere(self, tmp_path):
        """Test returns user dir when no weights exist anywhere."""
        user_dir = tmp_path / "nonexistent"

        with patch.dict(os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(user_dir)}):
            result = find_weights_dir()
            # Should return user dir (where download will happen)
            assert result == user_dir


class TestWeightsExist:
    """Tests for weights_exist function."""

    def test_returns_true_when_all_files_exist(self, tmp_path):
        """Test returns True when all weight files exist."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir(parents=True)
        for f in WEIGHT_FILES:
            (weights_dir / f).touch()

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            assert weights_exist() is True

    def test_returns_false_when_missing_files(self, tmp_path):
        """Test returns False when some weight files are missing."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir(parents=True)
        # Only create first file
        (weights_dir / WEIGHT_FILES[0]).touch()

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            assert weights_exist() is False

    def test_returns_false_when_dir_not_exists(self, tmp_path):
        """Test returns False when weights directory doesn't exist."""
        weights_dir = tmp_path / "nonexistent"

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            assert weights_exist() is False


class TestDownloadWeights:
    """Tests for download_weights function."""

    def test_skips_download_when_weights_exist(self, tmp_path):
        """Test that download is skipped when weights already exist."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir(parents=True)
        for f in WEIGHT_FILES:
            (weights_dir / f).touch()

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            with patch("urllib.request.urlretrieve") as mock_retrieve:
                download_weights(verbose=False)
                mock_retrieve.assert_not_called()

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Test that weights directory is created if it doesn't exist."""
        weights_dir = tmp_path / "new_weights_dir"
        assert not weights_dir.exists()

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            with patch("urllib.request.urlretrieve"):
                download_weights(verbose=False)
                assert weights_dir.exists()

    def test_downloads_all_files(self, tmp_path):
        """Test that all weight files are downloaded."""
        weights_dir = tmp_path / "weights"

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            with patch("urllib.request.urlretrieve") as mock_retrieve:
                download_weights(verbose=False)

                # Should have called urlretrieve for each file
                assert mock_retrieve.call_count == len(WEIGHT_FILES)

                # Verify correct URLs were used
                for call in mock_retrieve.call_args_list:
                    url = call[0][0]
                    assert url.startswith(BASE_URL)

    def test_skips_existing_files(self, tmp_path):
        """Test that existing files are not re-downloaded."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir(parents=True)
        # Create first file only
        (weights_dir / WEIGHT_FILES[0]).touch()

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            with patch("urllib.request.urlretrieve") as mock_retrieve:
                download_weights(verbose=False)

                # Should skip the first file
                assert mock_retrieve.call_count == len(WEIGHT_FILES) - 1

    def test_cleans_up_partial_download_on_failure(self, tmp_path):
        """Test that partial downloads are cleaned up on failure."""
        weights_dir = tmp_path / "weights"

        def fake_download(url, filepath):
            # Create partial file then fail
            Path(filepath).touch()
            raise Exception("Network error")

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            with patch("urllib.request.urlretrieve", side_effect=fake_download):
                with pytest.raises(RuntimeError, match="Failed to download"):
                    download_weights(verbose=False)

                # Partial file should be cleaned up
                assert not (weights_dir / WEIGHT_FILES[0]).exists()

    def test_raises_permission_error_on_mkdir_failure(self, tmp_path):
        """Test that PermissionError is raised with helpful message."""
        weights_dir = tmp_path / "readonly" / "weights"

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            with patch(
                "pathlib.Path.mkdir", side_effect=PermissionError("denied")
            ):
                with pytest.raises(
                    PermissionError, match="GRAPHRELAX_WEIGHTS_DIR"
                ):
                    download_weights(verbose=False)


class TestEnsureWeights:
    """Tests for ensure_weights function."""

    def test_calls_download_weights(self, tmp_path):
        """Test that ensure_weights calls download_weights."""
        weights_dir = tmp_path / "weights"

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            with patch("graphrelax.weights.download_weights") as mock_download:
                ensure_weights(verbose=False)
                mock_download.assert_called_once_with(verbose=False)


class TestWeightFilesConstant:
    """Tests for WEIGHT_FILES constant."""

    def test_contains_required_files(self):
        """Test that WEIGHT_FILES contains all required model files."""
        assert "proteinmpnn_v_48_020.pt" in WEIGHT_FILES
        assert "ligandmpnn_v_32_010_25.pt" in WEIGHT_FILES
        assert "solublempnn_v_48_020.pt" in WEIGHT_FILES
        assert "ligandmpnn_sc_v_32_002_16.pt" in WEIGHT_FILES

    def test_count(self):
        """Test that we have exactly 4 weight files."""
        assert len(WEIGHT_FILES) == 4


class TestBaseUrlConstant:
    """Tests for BASE_URL constant."""

    def test_url_format(self):
        """Test that BASE_URL is a valid URL."""
        assert BASE_URL.startswith("https://")
        assert "ligandmpnn" in BASE_URL.lower()


@pytest.mark.integration
class TestActualDownload:
    """Integration tests that perform actual network downloads.

    These tests verify the download mechanism works end-to-end.
    They are marked as integration tests and can be skipped in CI.
    """

    def test_download_single_file(self, tmp_path):
        """Test downloading a single weight file."""
        import urllib.request

        weights_dir = tmp_path / "weights"
        weights_dir.mkdir(parents=True)

        # Download just the first file
        filename = WEIGHT_FILES[0]
        url = f"{BASE_URL}/{filename}"
        filepath = weights_dir / filename

        urllib.request.urlretrieve(url, filepath)

        assert filepath.exists()
        # File should be non-empty (at least 1MB for model weights)
        assert filepath.stat().st_size > 1_000_000

    def test_full_download_workflow(self, tmp_path):
        """Test the full download workflow with ensure_weights."""
        weights_dir = tmp_path / "weights"

        with patch.dict(
            os.environ, {"GRAPHRELAX_WEIGHTS_DIR": str(weights_dir)}
        ):
            # Verify weights don't exist initially
            assert not weights_exist()

            # Download weights
            ensure_weights(verbose=True)

            # Verify all weights now exist
            assert weights_exist()

            # Verify all files are present and non-empty
            for f in WEIGHT_FILES:
                filepath = weights_dir / f
                assert filepath.exists(), f"Missing: {f}"
                assert filepath.stat().st_size > 1_000_000, f"Too small: {f}"
