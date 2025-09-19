import json
from os import remove
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, List
import pydicom
import torch
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from shutil import rmtree

from .normalizer import normalize_ct_image, resample_volume
from .metadata import MetadataManager

logger = logging.getLogger(__name__)


class DicomProcessor:
    def __init__(
        self,
        input_folder: Path,
        output_folder: Path,
        num_workers: int = 4,
        max_samples: Optional[int] = None,
        hu_samples: int = 1000,
        target_vol_dimensions: Tuple[int, int, int] | None = None,
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.hu_samples = hu_samples
        self.metadata_manager = MetadataManager(output_folder)
        self.target_vol_dimensions = target_vol_dimensions

    def get_output_paths(self, ds: pydicom.Dataset) -> Tuple[Path, Path]:
        """Get the output paths for a DICOM file based on patient, study, and series information."""
        patient_id = str(getattr(ds, "PatientID", "unknown"))
        study_uid = str(getattr(ds, "StudyInstanceUID", "unknown"))
        series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        series_desc = str(getattr(ds, "SeriesDescription", "unknown")).replace(" ", "_")

        # Directory naming, NOTE: only last 8 digits of UID used for file/folder naming.
        study_dir_name = f"study_{study_uid[-8:].replace('.', '_')}"
        series_dir_name = f"series_{series_desc}_{series_uid[-8].replace('.', '_')}"

        # Final path
        patient_dir = self.output_folder / f"patient_{patient_id}"
        study_dir = patient_dir / study_dir_name
        series_dir = study_dir / series_dir_name

        images_dir = series_dir / "images"

        return images_dir, series_dir

    def _process_dicom_file(self, dicom_path: Path, images_dir: Path) -> None:
        """Process a single DICOM file."""
        try:
            ds = pydicom.dcmread(dicom_path)
            image_size = ds.pixel_array.shape

            if image_size != (512, 512):
                logger.warning(
                    f"Skipping image. Expected size (512, 512), got {image_size}"
                )
                return

            slice_position = float(getattr(ds, "SliceLocation", 0.0))
            instance_number = int(getattr(ds, "InstanceNumber", 0))
            slice_id = f"slice_{instance_number:04d}_pos_{slice_position:.2f}"

            image = ds.pixel_array.astype(np.float32)

            if ds.Modality == "CT":
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                image = image * slope + intercept
                image = normalize_ct_image(image)

            image_tensor = torch.from_numpy(image)
            torch.save(image_tensor, images_dir / f"{slice_id}.pt")
            remove(dicom_path)  # Remove original DICOM file after processing

        except Exception as e:
            logger.error(f"Error processing {dicom_path}: {str(e)}")

    def process_dicom_series(self, dicom_series: List[Path]) -> None:
        """Process a single DICOM series."""
        num_slices = len(dicom_series)

        first_dicom_path = dicom_series[0]
        first_ds = pydicom.dcmread(first_dicom_path)
        images_dir, series_dir = self.get_output_paths(first_ds)
        images_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_manager.save_scan_metadata(
            first_ds,
            series_dir,
            num_slices=num_slices,
            resampled_dimensions=self.target_vol_dimensions,
        )

        try:
            for dicom_path in dicom_series:
                self._process_dicom_file(dicom_path, images_dir)

        except RuntimeError as e:
            if "JPEG Extended" in str(e):
                logger.warning(f"Skipping file with JPEG Extended format: {str(e)}")
            else:
                raise
        except Exception as e:
            logger.error(f"Error processing {dicom_path}: {str(e)}")
            raise

    def process_folder(self) -> None:
        """Process all DICOM files in the input folder."""
        self.output_folder.mkdir(parents=True, exist_ok=True)

        dicom_files = list(self.input_folder.rglob("*.dcm"))
        if not dicom_files:
            logger.warning(f"No DICOM files found in {self.input_folder}")
            return

        logger.info(f"Found {len(dicom_files)} DICOM files")

        # Group files by patient -> study -> series
        patient_study_series_files = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for dicom_file in tqdm(dicom_files, desc="Sorting DICOM files"):
            try:
                ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
                patient_id = str(getattr(ds, "PatientID", "unknown"))
                study_uid = str(getattr(ds, "StudyInstanceUID", "unknown"))
                series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))

                patient_study_series_files[patient_id][study_uid][series_uid].append(
                    dicom_file
                )
            except Exception as e:
                logger.warning(f"Error reading metadata from {dicom_file}: {str(e)}")

        num_studies = sum(
            len(studies) for studies in patient_study_series_files.values()
        )
        logger.info(
            f"Found {len(patient_study_series_files)} patients and {num_studies} studies"
        )

        logger.info("Processing DICOM files...")

        # Flatten all DICOM file paths into a single list for parallel processing
        all_series = [
            series
            for patient in patient_study_series_files.values()
            for study in patient.values()
            for series in study.values()
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            list(
                tqdm(
                    executor.map(self.process_dicom_series, all_series),
                    total=len(all_series),
                    desc="Processing CT series",
                )
            )
        logger.info("Creating dataset summary...")
        summary = self.metadata_manager.create_dataset_summary()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_folder / f"dataset_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Dataset summary saved to {summary_path}")

    def stack_slices_to_volume(
        self,
        save_name: str = "volume.pt",
        delete_slices: bool = True,
        target_vol_dimensions: Tuple[int, int, int] | None = None,
    ) -> None:
        """
        Recursively walks through `output_folder`, finds all 'images' folders, stacks 2D .pt slices into 3D volumes,
        and saves the resulting volume as a new .pt file in the parent directory.

        Args:
            save_name (str): Name of the output volume file. Default: 'volume.pt'
            delete_slices (bool): If True, deletes the 'images' folder after stacking. Default: True
        """

        root_dir = Path(self.output_folder)
        image_folders = list(root_dir.rglob("images"))

        desc = (
            "Stacking slices"
            if target_vol_dimensions is None
            else f"Stacking slices and resampling to {target_vol_dimensions}"
        )

        for image_folder in tqdm(image_folders, desc=desc):
            pt_files = sorted(
                image_folder.glob("*.pt"), key=lambda f: float(f.stem.split("_")[-1])
            )  # sorts by position

            if not pt_files:
                logging.warning(f"No .pt files found in {image_folder}")
                continue

            try:
                slices = [torch.load(f) for f in pt_files]
                volume = torch.stack(slices, dim=0)  # [depth, height, width]

                if target_vol_dimensions:
                    volume = resample_volume(volume, target_vol_dimensions)

                parent_dir = image_folder.parent
                save_path = parent_dir / save_name
                torch.save(volume, save_path)

                if delete_slices:
                    rmtree(image_folder)

            except Exception as e:
                logging.error(f"Failed to process {image_folder}: {e}")
