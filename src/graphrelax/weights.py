"""Automatic downloading of LigandMPNN model weights."""

import logging
import os
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
    """
    Get the directory where model weights should be stored.

    Uses ~/.graphrelax/weights/ as the primary location, following
    conventions used by other biological ML tools (boltz, chai, etc.).

    The directory can be overridden via the GRAPHRELAX_WEIGHTS_DIR
    environment variable.

    Returns:
        Path to the weights directory
    """
    if env_dir := os.environ.get("GRAPHRELAX_WEIGHTS_DIR"):
        return Path(env_dir)
    return Path.home() / ".graphrelax" / "weights"


def _get_package_weights_dir() -> Path:
    """Get the legacy package-internal weights directory.

    For backwards compatibility with existing installations.
    """
    return Path(__file__).parent / "LigandMPNN" / "model_params"


def find_weights_dir() -> Path:
    """
    Find the directory containing model weights.

    Checks locations in order:
    1. User weights directory (~/.graphrelax/weights/)
    2. Environment override (GRAPHRELAX_WEIGHTS_DIR)
    3. Package directory (legacy, for backwards compatibility)

    Returns:
        Path to directory containing weights, or the user weights dir
        if none found
    """
    # Check user directory first (primary location)
    user_dir = get_weights_dir()
    if user_dir.exists() and all((user_dir / f).exists() for f in WEIGHT_FILES):
        return user_dir

    # Check package directory (legacy/backwards compatibility)
    package_dir = _get_package_weights_dir()
    if package_dir.exists() and all(
        (package_dir / f).exists() for f in WEIGHT_FILES
    ):
        return package_dir

    # Default to user directory (will need download)
    return user_dir


def weights_exist() -> bool:
    """Check if all required weight files exist in any known location."""
    weights_dir = find_weights_dir()
    return all((weights_dir / f).exists() for f in WEIGHT_FILES)


def download_weights(verbose: bool = True) -> None:
    """
    Download LigandMPNN model weights if they don't exist.

    Downloads to ~/.graphrelax/weights/ by default.

    Args:
        verbose: If True, print progress messages

    Raises:
        RuntimeError: If download fails
        PermissionError: If unable to create weights directory
    """
    if weights_exist():
        if verbose:
            logger.info(f"Model weights already exist at {find_weights_dir()}")
        return

    weights_dir = get_weights_dir()

    # Create directory with explicit error handling
    try:
        weights_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot create weights directory at {weights_dir}. "
            f"Set GRAPHRELAX_WEIGHTS_DIR to a writable location. Error: {e}"
        ) from e
    except OSError as e:
        raise RuntimeError(
            f"Failed to create weights directory at {weights_dir}: {e}"
        ) from e

    # Always log when downloading (important user feedback)
    logger.info(
        f"Downloading LigandMPNN model weights (~40MB) to {weights_dir}..."
    )

    for filename in WEIGHT_FILES:
        filepath = weights_dir / filename
        if filepath.exists():
            if verbose:
                logger.info(f"  {filename} already exists, skipping")
            continue

        url = f"{BASE_URL}/{filename}"
        logger.info(f"  Downloading {filename}...")

        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            # Clean up partial download
            filepath.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {filename} from {url}: {e}"
            ) from e

    logger.info("Model weights downloaded successfully.")


def ensure_weights(verbose: bool = True) -> None:
    """
    Ensure model weights are available, downloading if necessary.

    This is the main entry point for automatic weight management.

    Args:
        verbose: If True, print progress messages

    Raises:
        RuntimeError: If weights cannot be downloaded
        PermissionError: If weights directory cannot be created
    """
    download_weights(verbose=verbose)
