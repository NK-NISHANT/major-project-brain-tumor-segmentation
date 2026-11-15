"""Uncertainty calibration, Expected Calibration Error (ECE), Brier score, and reliability diagnostics.

Project: Brain Tumor Segmentation Using 2D CNNs and a Reliability-Aware AI Agent
Workspace: D:\MajorProject
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch import nn, optim


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for multi-class classification and segmentation.
    
    Calibrates model output logits z to calibrated probabilities:
        p(y = c | x) = softmax(z / T)_c
    where T > 0 is a single scalar parameter learned by minimizing Negative Log-Likelihood (NLL).
    Temperature scaling preserves argmax rankings and does not alter discrete predictions,
    but rescales overconfident or underconfident probabilities.
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0.0:
            raise ValueError(f"initial_temperature must be > 0, got {initial_temperature}")
        # Store log_temperature for unconstrained optimization while guaranteeing T > 0
        self.log_temperature = nn.Parameter(torch.tensor([math.log(initial_temperature)], dtype=torch.float32))

    @property
    def temperature(self) -> float:
        """Current temperature value T = exp(log_temperature)."""
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature: z / T."""
        t = torch.exp(self.log_temperature)
        return logits / t

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature: z / T (convenience alias for forward)."""
        return self.forward(logits)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.05,
        max_iter: int = 100,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Fit optimal temperature on a validation calibration cohort using L-BFGS.

        Args:
            logits: Logits tensor of shape [N, C] or [N, C, H, W].
            targets: Target class indices of shape [N] or [N, H, W] with integer class labels.
            lr: Learning rate for optimizer.
            max_iter: Maximum optimization iterations.
            device: Computation device.

        Returns:
            Dictionary with optimization diagnostics ('initial_temperature', 'optimal_temperature',
            'initial_nll', 'final_nll', 'iterations', 'converged').
        """
        self.to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        # Flatten spatial dimensions if needed: [N, C, H, W] -> [Total_Voxels, C]
        if logits.ndim == 4:
            c = logits.shape[1]
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c).to(device)
            targets_flat = targets.reshape(-1).to(device).long()
        elif logits.ndim == 2:
            logits_flat = logits.to(device)
            targets_flat = targets.to(device).long()
        else:
            raise ValueError(f"Expected 2D or 4D logits tensor, got shape {tuple(logits.shape)}")

        # Filter out invalid target labels if any
        valid_mask = (targets_flat >= 0) & (targets_flat < logits_flat.shape[1])
        if not valid_mask.all():
            logits_flat = logits_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]

        if logits_flat.shape[0] == 0:
            raise ValueError("No valid voxels available to fit temperature scaling.")

        init_t = self.temperature
        with torch.no_grad():
            init_nll = float(criterion(self.forward(logits_flat), targets_flat).item())

        # Optimize using L-BFGS with strong Wolfe line search
        optimizer = optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        iteration_count = 0
        def closure():
            nonlocal iteration_count
            iteration_count += 1
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_flat)
            loss = criterion(scaled_logits, targets_flat)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered during temperature scaling: {loss.item()}")
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            converged = True
        except Exception as e:
            # Fallback to Adam if L-BFGS encounters line search difficulty
            converged = False
            adam = optim.Adam([self.log_temperature], lr=0.01)
            for _ in range(50):
                adam.zero_grad()
                loss = criterion(self.forward(logits_flat), targets_flat)
                loss.backward()
                adam.step()

        # Clamp temperature to reasonable clinical bounds [0.05, 10.0]
        with torch.no_grad():
            clamped_t = float(torch.clamp(torch.exp(self.log_temperature), min=0.05, max=10.0).item())
            self.log_temperature.copy_(torch.tensor([math.log(clamped_t)], dtype=torch.float32))
            final_nll = float(criterion(self.forward(logits_flat), targets_flat).item())

        return {
            "initial_temperature": init_t,
            "optimal_temperature": self.temperature,
            "initial_nll": init_nll,
            "final_nll": final_nll,
            "iterations": iteration_count,
            "converged": converged,
        }


def compute_multiclass_ece(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute Multi-class Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Top-label confidence formulation:
        For each prediction i:
            conf_i = max_c p_{i, c}
            pred_y_i = argmax_c p_{i, c}
            acc_i = 1{ pred_y_i == y_i }

        Confidence interval [0, 1] is partitioned into M equal-width bins B_m.
        ECE = sum_{m=1}^M (|B_m| / N) * | acc(B_m) - conf(B_m) |
        MCE = max_{m=1}^M | acc(B_m) - conf(B_m) |

    Empty bins (|B_m| == 0) contribute 0 to ECE and are excluded from MCE.

    Args:
        probs: Softmax probabilities of shape [N, C] or [N, C, H, W] where sum_c p_c = 1.
        targets: Integer class labels of shape [N] or [N, H, W].
        n_bins: Number of equal-width confidence bins (default 15).

    Returns:
        Dictionary with keys:
            'ece': Expected Calibration Error float in [0, 1].
            'mce': Maximum Calibration Error float in [0, 1].
            'bin_accuracies': List of accuracy values per bin.
            'bin_confidences': List of mean confidence values per bin.
            'bin_counts': List of sample counts per bin.
            'bin_edges': List of bin edge values (length n_bins + 1).
    """
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = np.asarray(probs)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    # Flatten spatial dimensions if 4D
    if probs_np.ndim == 4:
        c = probs_np.shape[1]
        probs_flat = np.moveaxis(probs_np, 1, -1).reshape(-1, c)
        targets_flat = targets_np.reshape(-1)
    elif probs_np.ndim == 2:
        probs_flat = probs_np
        targets_flat = targets_np.reshape(-1)
    else:
        raise ValueError(f"Expected 2D or 4D probabilities array, got shape {probs_np.shape}")

    # Top-1 confidence and predicted class
    confidences = np.max(probs_flat, axis=-1)
    predictions = np.argmax(probs_flat, axis=-1)
    accuracies = (predictions == targets_flat).astype(np.float64)

    n_samples = len(confidences)
    if n_samples == 0:
        return {
            "ece": 0.0,
            "mce": 0.0,
            "bin_accuracies": [],
            "bin_confidences": [],
            "bin_counts": [],
            "bin_edges": list(np.linspace(0.0, 1.0, n_bins + 1)),
        }

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    bin_errors = []

    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_err = abs(bin_acc - bin_conf)
            ece += (count / n_samples) * bin_err
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_errors.append(bin_err)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((low + high) / 2.0)
            bin_errors.append(0.0)

    mce = float(max(bin_errors)) if bin_errors else 0.0

    return {
        "ece": float(ece),
        "mce": mce,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_edges": [float(x) for x in bin_edges],
    }


def compute_multiclass_brier_score(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_classes: int = 4,
) -> float:
    """Compute Multi-Class Brier Score.

    Convention:
        BS = (1 / N) * sum_{i=1}^N sum_{c=0}^{C-1} (p_{i, c} - 1{y_i == c})^2
    where p_{i, c} is the predicted probability for class c and 1{y_i == c} is one-hot target.
    A lower Brier score represents better calibrated and more accurate probabilistic predictions.

    Args:
        probs: Softmax probabilities of shape [N, C] or [N, C, H, W].
        targets: Target integer class indices of shape [N] or [N, H, W].
        num_classes: Number of distinct classes (default 4).

    Returns:
        Scalar Brier score float.
    """
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = np.asarray(probs)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    if probs_np.ndim == 4:
        c = probs_np.shape[1]
        probs_flat = np.moveaxis(probs_np, 1, -1).reshape(-1, c)
        targets_flat = targets_np.reshape(-1)
    elif probs_np.ndim == 2:
        probs_flat = probs_np
        targets_flat = targets_np.reshape(-1)
    else:
        raise ValueError(f"Expected 2D or 4D probabilities array, got shape {probs_np.shape}")

    n_samples = probs_flat.shape[0]
    if n_samples == 0:
        return 0.0

    # One-hot encode targets
    one_hot = np.zeros_like(probs_flat)
    valid_mask = (targets_flat >= 0) & (targets_flat < num_classes)
    one_hot[np.arange(n_samples)[valid_mask], targets_flat[valid_mask].astype(int)] = 1.0

    squared_errors = np.sum((probs_flat - one_hot) ** 2, axis=-1)
    return float(np.mean(squared_errors))


def compute_nll(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Compute Negative Log-Likelihood (NLL) / Cross-Entropy Loss from probabilities."""
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = np.asarray(probs)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    if probs_np.ndim == 4:
        c = probs_np.shape[1]
        probs_flat = np.moveaxis(probs_np, 1, -1).reshape(-1, c)
        targets_flat = targets_np.reshape(-1)
    elif probs_np.ndim == 2:
        probs_flat = probs_np
        targets_flat = targets_np.reshape(-1)
    else:
        raise ValueError(f"Expected 2D or 4D probabilities array, got shape {probs_np.shape}")

    n_samples = probs_flat.shape[0]
    if n_samples == 0:
        return 0.0

    probs_clamped = np.clip(probs_flat, epsilon, 1.0)
    target_probs = probs_clamped[np.arange(n_samples), targets_flat.astype(int)]
    return float(-np.mean(np.log(target_probs)))


def compute_risk_coverage_curve(
    uncertainties: Sequence[float] | np.ndarray,
    dice_scores: Sequence[float] | np.ndarray,
    n_thresholds: int = 50,
) -> dict[str, list[float]]:
    """Compute Risk-Coverage Curve for an abstention/selective classification policy.

    For a given uncertainty threshold tau:
        - Retain items with uncertainty <= tau (Accepted items).
        - Abstain / flag items with uncertainty > tau (Referred for human review).
        - Coverage = Count(Accepted) / Total.
        - Risk = 1.0 - Mean_Dice(Accepted) (or error rate).

    Args:
        uncertainties: Array of predictive uncertainty metrics per item (e.g. mean foreground entropy).
        dice_scores: Corresponding segmentation Dice scores for each item.
        n_thresholds: Number of threshold evaluation points.

    Returns:
        Dictionary with 'coverage', 'risk', 'mean_dice', and 'thresholds'.
    """
    unc_arr = np.asarray(uncertainties, dtype=np.float64)
    dice_arr = np.asarray(dice_scores, dtype=np.float64)

    # Filter non-finite pairs
    valid = np.isfinite(unc_arr) & np.isfinite(dice_arr)
    unc_arr = unc_arr[valid]
    dice_arr = dice_arr[valid]

    n_total = len(unc_arr)
    if n_total == 0:
        return {"coverage": [], "risk": [], "mean_dice": [], "thresholds": []}

    # Generate thresholds spanning the empirical percentile range
    thresholds = np.linspace(float(np.min(unc_arr)), float(np.max(unc_arr)), n_thresholds)

    coverages: list[float] = []
    risks: list[float] = []
    mean_dices: list[float] = []
    th_list: list[float] = []

    for tau in thresholds:
        retained = unc_arr <= tau
        count = int(np.sum(retained))
        if count == 0:
            continue
        cov = count / n_total
        m_dice = float(np.mean(dice_arr[retained]))
        risk = 1.0 - m_dice

        coverages.append(float(cov))
        risks.append(float(risk))
        mean_dices.append(float(m_dice))
        th_list.append(float(tau))

    return {
        "coverage": coverages,
        "risk": risks,
        "mean_dice": mean_dices,
        "thresholds": th_list,
    }
