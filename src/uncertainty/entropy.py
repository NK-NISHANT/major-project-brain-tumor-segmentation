import torch


def softmax_entropy(logits):
    """
    Calculate normalized pixel-wise predictive entropy
    from segmentation logits.

    Args:
        logits: Tensor of shape [B, C, H, W]

    Returns:
        Normalized entropy tensor of shape [B, H, W]
    """
    probabilities = torch.softmax(logits, dim=1)

    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-8))
    ).sum(dim=1)

    num_classes = logits.shape[1]

    max_entropy = torch.log(
        torch.tensor(float(num_classes), device=logits.device)
    )

    normalized_entropy = entropy / max_entropy

    return normalized_entropy
