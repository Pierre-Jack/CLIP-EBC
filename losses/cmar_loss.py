import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAlignmentRankingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, image_features, text_features, ground_truth_bins):
        """
        image_features: [B, H, W, Dim] (From CLIP_EBC)
        text_features: [Num_Bins, Dim] (From text encoder)
        ground_truth_bins: [B, 1, H, W] (Quantized bin indices from train.py)
        """
        # 1. Flatten the spatial dimensions to treat every pixel patch equally
        batch_size = image_features.shape[0]
        embed_dim = image_features.shape[-1]
        num_bins = text_features.shape[0]

        # Reshape to [Batch * H * W, Dim]
        img_flat = image_features.view(-1, embed_dim)
        # Reshape to [Batch * H * W]
        gt_flat = ground_truth_bins.view(-1).long()

        # 2. Normalize features
        img_flat = F.normalize(img_flat, dim=-1)
        txt_norm = F.normalize(text_features, dim=-1)

        # 3. Calculate cosine similarity: [Batch*H*W, Num_Bins]
        # This gives us the similarity score of every image patch against every text count prompt
        similarity = torch.matmul(img_flat, txt_norm.t())

        loss = 0.0
        valid_pairs = 0

        # 4. Ranking Logic (Vectorized for speed)
        # Instead of looping, we gather the similarities for the True Bin, the Next Bin, and the Prev Bin

        # Get the similarities of the correctly matching bins
        sim_true = similarity[torch.arange(similarity.size(0)), gt_flat]

        # Mask for valid "Next Bins" (Exclude the absolute highest bin)
        valid_next_mask = gt_flat < (num_bins - 1)
        if valid_next_mask.any():
            sim_next = similarity[valid_next_mask, gt_flat[valid_next_mask] + 1]
            sim_true_next = sim_true[valid_next_mask]
            # Penalty: Next Bin Sim should be smaller than True Bin Sim by at least 'margin'
            loss += F.relu(sim_next - sim_true_next + self.margin).sum()
            valid_pairs += valid_next_mask.sum().item()

        # Mask for valid "Prev Bins" (Exclude Bin 0)
        valid_prev_mask = gt_flat > 0
        if valid_prev_mask.any():
            sim_prev = similarity[valid_prev_mask, gt_flat[valid_prev_mask] - 1]
            sim_true_prev = sim_true[valid_prev_mask]
            # Penalty: Prev Bin Sim should be smaller than True Bin Sim by at least 'margin'
            loss += F.relu(sim_prev - sim_true_prev + self.margin).sum()
            valid_pairs += valid_prev_mask.sum().item()

        if valid_pairs > 0:
            return loss / valid_pairs
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)