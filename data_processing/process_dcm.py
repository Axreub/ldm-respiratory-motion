from pathlib import Path
import logging

from data_processing.dicom.processor import DicomProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def process_dcm(
    input_folder: str,
    output_folder: str,
    num_workers: int = 4,
    max_samples: int = None,
    hu_samples: int = 1000,
    target_vol_dimensions: tuple[int, int, int] | None = None,
) -> None:
    """
    Process DICOM files in the input folder and save normalized images and metadata.

    Args:
        input_folder: Path to the folder containing DICOM files.
        output_folder: Path to save processed files.
        num_workers: Number of worker processes for parallel processing.
        max_samples: Maximum number of files to analyze for metadata (None for all).
        hu_samples: Number of files to analyze for HU values.
        target_vol_dimensions: Dimensions of the target volume in (z,y,x) order.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    processor = DicomProcessor(
        input_folder=input_path,
        output_folder=output_path,
        num_workers=num_workers,
        max_samples=max_samples,
        hu_samples=hu_samples,
        target_vol_dimensions=target_vol_dimensions,
    )

    # Process files
    processor.process_folder()
    processor.stack_slices_to_volume(target_vol_dimensions=target_vol_dimensions)
