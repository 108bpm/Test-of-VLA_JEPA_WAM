"""Minimal training-step example. Run after installing the package."""
import torch
from latent_world_model import LatentWorldModel, LatentWorldModelConfig

model = LatentWorldModel(
    "checkpoints/vjepa2-vitl-fpc64-256",
    LatentWorldModelConfig(latent_action_dim=2048, num_views=2),
).cuda()

# Replace these random tensors with your dataloader and action-token producer.
videos = torch.randint(0, 256, (2, 2, 8, 3, 256, 256), dtype=torch.uint8, device="cuda")
latent_actions = torch.randn(2, 3 * 8, 2048, device="cuda")
loss = model.loss(videos, latent_actions)
loss.backward()  # only predictor receives gradients; encoder remains frozen
print(loss.item())
