"""Automatic downloading of LigandMPNN model weights."""

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://files.ipd.uw.edu/pub/ligandmpnn"

WEIGHT_FILES = [
    "proteinmpnn_v_48_020.pt",
    "ligandmpnn_v_32_010_25.pt",
    "solublempnn_v_48_020.pt",
    "ligandmpnn_sc_v_32_002_16.pt",
]


def get_weights_dir() -> Path:
    """Get the directory where model weights should be stored."""
    return Path(__file__).parent / "LigandMPNN" / "model_params"


def weights_exist() -> bool:
    """Check if all required weight files exist."""
    weights_dir = get_weights_dir()
    return all((weights_dir / f).exists() for f in WEIGHT_FILES)


def download_weights(verbose: bool = True) -> None:
    """
    Download LigandMPNN model weights if they don't exist.

    Args:
        verbose: If True, print progress messages
    """
    if weights_exist():
        return

    weights_dir = get_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info("Downloading LigandMPNN model weights (~40MB)...")

    for filename in WEIGHT_FILES:
        filepath = weights_dir / filename
        if filepath.exists():
            continue

        url = f"{BASE_URL}/{filename}"
        if verbose:
            logger.info(f"  Downloading {filename}...")

        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            # Clean up partial download
            filepath.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {filename} from {url}: {e}"
            ) from e

    if verbose:
        logger.info("Model weights downloaded successfully.")


def ensure_weights(verbose: bool = True) -> None:
    """
    Ensure model weights are available, downloading if necessary.

    This is the main entry point for automatic weight management.

    Args:
        verbose: If True, print progress messages
    """
    download_weights(verbose=verbose)
