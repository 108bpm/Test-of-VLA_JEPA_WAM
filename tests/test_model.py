"""Regression tests for public latent sequence alignment."""

import unittest
from pathlib import Path

import torch

from latent_world_model import LatentWorldModel, evaluate_latent_prediction
from latent_world_model.evaluation.runner import CONDITION_SPECS, RunnerConfig, _run_condition, run_evaluation


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

    def test_multiview_fusion_preserves_sample_identity(self):
        """Sample-major [B,V] encoder outputs must never be chunked by view."""
        # Flattened order is b0v0, b0v1, b1v0, b1v1.  One token and one
        # channel are sufficient to make the expected pairing unambiguous.
        features = torch.tensor([[[0.0]], [[1.0]], [[10.0]], [[11.0]]])

        fused = LatentWorldModel.fuse_multiview_features(
            features,
            batch_size=2,
            num_views=2,
        )

        expected = torch.tensor([[[0.0, 1.0]], [[10.0, 11.0]]])
        self.assertTrue(torch.equal(fused, expected))

    def test_multiview_fusion_rejects_inconsistent_shape(self):
        with self.assertRaisesRegex(ValueError, r"batch_size \* num_views"):
            LatentWorldModel.fuse_multiview_features(
                torch.zeros(3, 1, 1),
                batch_size=2,
                num_views=2,
            )

    def test_legacy_h3_requires_explicit_opt_in(self):
        config = RunnerConfig(
            dataset_root=Path("unused"),
            index_path=Path("unused"),
            output_dir=Path("unused"),
            encoder_path=Path("unused"),
            checkpoint_path=Path("unused"),
            conditions=("S3",),
        )
        with self.assertRaisesRegex(ValueError, "temporally misaligned"):
            run_evaluation(config)

    def test_corrected_f2_stays_within_current_query(self):
        class IdentityPredictor:
            @staticmethod
            def predict_from_latents(context, actions):
                return context

        blocks = [torch.full((2, 3), float(index)) for index in range(6)]
        groups = [torch.full((8, 3), float(index)) for index in range(3)]

        metrics = _run_condition(
            IdentityPredictor(),
            "F2",
            CONDITION_SPECS["F2"],
            blocks,
            None,
            groups,
            None,
            blocks,
            device=torch.device("cpu"),
        )

        self.assertIn("h3_mse", metrics)
        self.assertEqual(metrics["h1_mse"], 1.0)
        self.assertEqual(metrics["h2_mse"], 4.0)
        self.assertEqual(metrics["h3_mse"], 9.0)


if __name__ == "__main__":
    unittest.main()
