"""Regression tests for public latent sequence alignment."""

import unittest

import torch

from latent_world_model import LatentWorldModel, evaluate_latent_prediction


class LatentSequenceAlignmentTest(unittest.TestCase):
    def test_context_targets_are_one_step_shifted(self):
        """The predictor's context-length output must match the teacher target."""
        model = object.__new__(LatentWorldModel)
        model.latent_steps = 4
        latents = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 8, 3)

        context, target = model.split_context_target(latents)

        self.assertTrue(torch.equal(context, latents[:, :6]))
        self.assertTrue(torch.equal(target, latents[:, 2:]))
        self.assertEqual(context.shape, target.shape)
        metrics = evaluate_latent_prediction(context, target)
        self.assertEqual(set(metrics), {"l1", "mse", "mean_token_cosine", "retrieval_accuracy"})


if __name__ == "__main__":
    unittest.main()
