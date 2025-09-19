import json
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Optional
import pydicom
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MetadataManager:
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def save_scan_metadata(
        self,
        ds: pydicom.Dataset,
        save_dir: Path,
        num_slices: int,
        resampled_dimensions: tuple[int, int, int] | None = None,
    ) -> None:
        """Save scan-level metadata."""
        pixel_spacing = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
        slice_thickness = float(getattr(ds, "SliceThickness", 1.0))

        if resampled_dimensions is not None:
            resampled_pixel_spacing = [
                pixel_spacing[0] * ds.pixel_array.shape[0] / resampled_dimensions[1],
                pixel_spacing[1] * ds.pixel_array.shape[1] / resampled_dimensions[2],
            ]
            resampled_slice_thickness = (
                slice_thickness * num_slices / resampled_dimensions[0]
            )

        orientation = np.array(ds.ImageOrientationPatient, dtype=float)
        row_direction = orientation[:3]
        column_direction = orientation[3:]

        # z-axis direction
        slice_direction = np.cross(row_direction, column_direction)

        scan_metadata = {
            "patient_id": str(getattr(ds, "PatientID", "unknown")),
            "study_date": str(getattr(ds, "StudyDate", "unknown")),
            "modality": str(getattr(ds, "Modality", "unknown")),
            "manufacturer": str(getattr(ds, "Manufacturer", "unknown")),
            "orientation": {
                "row": row_direction.tolist(),
                "column": column_direction.tolist(),
                "slice": slice_direction.tolist(),
            },
            "pixel_spacing": pixel_spacing,
            "slice_thickness": slice_thickness,
            "num_slices": num_slices,
            "resampled_pixel_spacing": (
                resampled_pixel_spacing if resampled_dimensions else None
            ),
            "resampled_slice_thickness": (
                resampled_slice_thickness if resampled_dimensions else None
            ),
        }

        with open(save_dir / "scan_info.json", "w") as f:
            json.dump(scan_metadata, f, indent=2)

    def create_dataset_summary(
        self,
    ) -> Dict[str, Any]:
        """Create summary of dataset metadata."""

        scan_metadatas = []

        for scan_metadata_path in self.output_path.rglob("*.json"):
            scan_metadatas.append(json.load(open(scan_metadata_path)))

        total_series = len(scan_metadatas)
        manufacturers = {}
        patients = set()
        pixel_spacings = []
        slice_thicknesses = []
        resampled_pixel_spacings = []
        resampled_slice_thicknesses = []
        num_slices = 0

        for scan_metadata in tqdm(scan_metadatas, desc="Analyzing DICOM metadata"):
            try:
                manufacturer = scan_metadata["manufacturer"]
                manufacturers[manufacturer] = manufacturers.get(manufacturer, 0) + 1

                patients.add(scan_metadata["patient_id"])

                num_slices += scan_metadata["num_slices"]

                # Extract spacing information
                pixel_spacings.append(
                    [float(x) for x in scan_metadata["pixel_spacing"]]
                )
                slice_thicknesses.append(float(scan_metadata["slice_thickness"]))

                if scan_metadata["resampled_pixel_spacing"] is not None:
                    resampled_pixel_spacings.append(
                        [float(x) for x in scan_metadata["resampled_pixel_spacing"]]
                    )
                if scan_metadata["resampled_slice_thickness"] is not None:
                    resampled_slice_thicknesses.append(
                        float(scan_metadata["resampled_slice_thickness"])
                    )

            except Exception as e:
                logger.warning(
                    f"Skipping metadata analysis for {scan_metadata}: {str(e)}"
                )

        summary = {
            "dataset_info": {
                "total_slices": num_slices,
                "total_series": total_series,
                "unique_patients": len(patients),
                "manufacturers": manufacturers,
            }
        }

        ps = np.array(pixel_spacings)
        summary["spatial_info"] = {
            "pixel_spacing": {
                "mean": ps.mean(axis=0).tolist(),
                "min": ps.min(axis=0).tolist(),
                "max": ps.max(axis=0).tolist(),
            }
        }

        st = np.array(slice_thicknesses)
        summary["spatial_info"]["slice_thickness"] = {
            "mean": float(st.mean()),
            "min": float(st.min()),
            "max": float(st.max()),
        }

        if resampled_pixel_spacings:
            resampled_ps = np.array(resampled_pixel_spacings)
            summary["spatial_info"]["resampled_pixel_spacing"] = {
                "mean": resampled_ps.mean(axis=0).tolist(),
                "min": resampled_ps.min(axis=0).tolist(),
                "max": resampled_ps.max(axis=0).tolist(),
            }

        if resampled_slice_thicknesses:
            resampled_st = np.array(resampled_slice_thicknesses)
            summary["spatial_info"]["resampled_slice_thickness"] = {
                "mean": float(resampled_st.mean()),
                "min": float(resampled_st.min()),
                "max": float(resampled_st.max()),
            }

        return summary
