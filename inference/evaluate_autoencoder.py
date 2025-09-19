import torch
import lpips
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from data_processing.data_loaders import MedicalDataLoader
from inference.utils.load_models import load_autoencoder
from utils.ct_views import view_to_axis, get_ct_view
import numpy as np
from tabulate import tabulate


class EvaluateAutoencoder:
    """Object that evaluates a model based on given metrics"""

    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path (str): The absolute path to the model to be evaluated.
        """
        self.device = "cuda" if torch.cuda is torch.cuda.is_available() else "cpu"
        self.model, self.model_args, self.train_args = load_autoencoder(
            model_path, self.device, use_eval=True
        )
        self.train_args.batch_size = 1
        self.lpips_metric = lpips.LPIPS(net="alex").to(self.device)
        self.psnr_metric = PeakSignalNoiseRatio().to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)

        self.metric_funcs = {
            "LPIPS": lambda x, y: self.lpips_metric(x, y),
            "PSNR": lambda x, y: self.psnr_metric(x, y),
            "MSE": lambda x, y: F.mse_loss(x, y),
            "SSIM": lambda x, y: self.ssim_metric(x, y),
        }
        if self.model_args.in_channels == 1:
            data_type = "volume"
        elif self.model_args.in_channels == 3:
            data_type = "dvf"

        self.train_loader, self.val_loader, self.test_loader = MedicalDataLoader(
            self.train_args, paired=False, data_type=data_type
        ).get_dataloaders()

    def _calculate_volume_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, slice_views: list, metrics: list
    ) -> dict:
        """Hidden method to calculate metrics over a batched ct volume"""
        results_per_view = {}

        for view in slice_views:
            view_totals = {m: 0.0 for m in metrics}

            axial_size = pred.shape[pred.ndim - 3 + view_to_axis(view)]

            for slice_idx in range(axial_size):
                pred_slice = get_ct_view(pred, view, slice_idx)
                target_slice = get_ct_view(target, view, slice_idx)

                for m in metrics:
                    value = self.metric_funcs[m](pred_slice, target_slice)
                    view_totals[m] += value.item() if torch.is_tensor(value) else value

            averaged = {m: view_totals[m] / axial_size for m in metrics}
            results_per_view[view] = averaged

        return results_per_view

    def calculate_metrics(
        self,
        metrics: list = ["LPIPS", "PSNR", "MSE", "SSIM"],
        slice_views: list = ["axial"],
        dataset_percentage: float = 1.0,
    ) -> dict:
        """Method to calculate the metrics for a pre-trained model

        Args:
            metrics (list): Which metrics to evaluate the model on.
            slice_views (list): Which plane(s) to use when generating slices for metric evaluation. You may specify one or more of: ['axial', 'coronal', 'sagittal'].
            dataset_percentage (float): What fraction of the dataset that should be evaluated.
        """

        compound_results = {view: {m: [] for m in metrics} for view in slice_views}

        dataset = self.val_loader.dataset
        indices = np.random.choice(
            len(dataset), int(len(dataset) * dataset_percentage), replace=False
        )
        subset = Subset(dataset, indices)
        subset_loader = DataLoader(subset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(subset_loader, desc="Evaluating"):
                target = batch.unsqueeze(1)
                pred = self.model(target)
                batch_results = self._calculate_volume_metrics(
                    pred, target, slice_views, metrics, dataset_percentage
                )
                for view in slice_views:
                    for m in metrics:
                        compound_results[view][m].append(batch_results[view][m])

        return compound_results

    def summarize_results(
        self, compound_results: dict, round_digits=4, print_table=True
    ) -> dict:
        """Method to process and summarize the compound results gathered from calculate_metrics"""
        summary = {}
        for view, metrics in compound_results.items():
            summary[view] = {}
            for metric, values in metrics.items():
                values_np = np.array(values)
                summary[view][metric] = {
                    "median": round(np.median(values_np), round_digits),
                    "mean": round(values_np.mean(), round_digits),
                    "std": round(values_np.std(), round_digits),
                    "min": round(values_np.min(), round_digits),
                    "max": round(values_np.max(), round_digits),
                }

        if print_table:
            for view in summary:
                print(f"\n Summary for view: {view.upper()}")
                headers = ["Metric", "Median", "Mean", "Std", "Min", "Max"]
                rows = []
                for metric, stats in summary[view].items():
                    rows.append(
                        [metric]
                        + [stats[k] for k in ["median", "mean", "std", "min", "max"]]
                    )
                print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

        return summary
