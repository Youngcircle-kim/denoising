"""Synthetic shape, equation and gradient checks for the unofficial model."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mstmn_train import model
        cls.implementation = model
        torch.set_num_threads(1)

    def test_pyramid_round_trip_including_negative_laplacians(self):
        pyramid = self.implementation.GaussianPyramid().double()
        image = torch.rand(2, 4, 32, 48, dtype=torch.float64)
        gaussians, laplacians = pyramid(image)
        self.assertEqual([g.shape[-2:] for g in gaussians],
                         [torch.Size((32, 48)), torch.Size((16, 24)),
                          torch.Size((8, 12)), torch.Size((4, 6))])
        self.assertTrue(any(bool((level < 0).any()) for level in laplacians))
        torch.testing.assert_close(pyramid.collapse(laplacians, gaussians[-1]), image,
                                   atol=1e-12, rtol=1e-12)

    def test_patch_fold_normalizes_overlap_and_covers_boundaries(self):
        memory = self.implementation.PatchMemory(patch_size=3, stride=2)
        image = torch.rand(2, 16, 7, 8)
        tokens, padded = memory.patchify(image)
        recovered = memory.unpatchify(tokens, padded, image.shape[-2:])
        torch.testing.assert_close(recovered, image)
        with self.assertRaises(ValueError):
            self.implementation.PatchMemory(patch_size=3, stride=4)

    def test_query_chunks_keep_full_attention_equivalent(self):
        torch.manual_seed(1)
        query = torch.rand(1, 16, 8, 8)
        keys = torch.rand(1, 3, 16, 8, 8)
        values = torch.randn_like(keys)
        small = self.implementation.PatchMemory(4, 4, query_chunk_size=1)
        large = self.implementation.PatchMemory(4, 4, query_chunk_size=128)
        torch.testing.assert_close(small(query, keys, values), large(query, keys, values),
                                   atol=1e-5, rtol=1e-5)

    def test_patch_feature_attention_matches_explicit_matrix_reference(self):
        torch.manual_seed(2)
        memory = self.implementation.PatchMemory(2, 2, query_chunk_size=3)
        query = torch.rand(2, 16, 5, 6, dtype=torch.float64)
        keys = torch.rand(2, 3, 16, 5, 6, dtype=torch.float64)
        values = torch.randn_like(keys)
        q, padded_size = memory.patchify(query)
        k, _ = memory.patchify(keys.reshape(6, 16, 5, 6))
        v, _ = memory.patchify(values.reshape(6, 16, 5, 6))
        batch, patches, features = q.shape
        k = k.reshape(batch, 3, patches, features).permute(0, 2, 1, 3)
        k = k.reshape(batch, patches, 3 * features)
        v = v.reshape(batch, 3, patches, features).permute(0, 2, 1, 3)
        v = v.reshape(batch, patches, 3 * features)
        affinity = torch.sigmoid(q.transpose(1, 2) @ k)
        self.assertEqual(tuple(affinity.shape), (2, 64, 192))
        expected_tokens = (affinity @ v.transpose(1, 2)).transpose(1, 2)
        expected = memory.unpatchify(expected_tokens, padded_size, query.shape[-2:])
        torch.testing.assert_close(memory(query, keys, values), expected, atol=1e-12, rtol=1e-12)

    def test_fusion_keeps_exact_equation_and_negative_third_weight(self):
        scalar = lambda value: torch.tensor([[[[value]]]], dtype=torch.float64)
        result = self.implementation.weighted_fusion(
            scalar(2), scalar(3), scalar(4), scalar(0.8), scalar(0.7))
        torch.testing.assert_close(result, scalar(1.7))

    def test_flow_units_and_backward_warp_direction(self):
        previous = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4).expand(1, 1, 3, 4)
        flow = torch.zeros(1, 2, 3, 4)
        flow[:, 0] = 1
        expected = torch.tensor([1, 2, 3, 3], dtype=torch.float32).view(1, 1, 1, 4).expand_as(previous)
        torch.testing.assert_close(self.implementation.warp_previous(previous, flow), expected)
        resized = self.implementation.resize_flow(flow, (6, 8))
        torch.testing.assert_close(resized[:, 0], torch.full((1, 6, 8), 2.0))
        torch.testing.assert_close(resized[:, 1], torch.zeros(1, 6, 8))

    def test_forward_backward_pad_crop_and_flow_injection(self):
        implementation = self.implementation

        class ExplicitTestFlow(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = []

            def forward(self, current, previous):
                self.seen.append(tuple(current.shape))
                return current.new_zeros((current.shape[0], 2, *current.shape[-2:]))

        flow = ExplicitTestFlow()
        network = implementation.MSTMN(flow)
        current = torch.rand(1, 4, 17, 25)
        history = torch.rand(1, 3, 4, 17, 25)
        result = network(current, current, history, torch.tensor([0.001]), torch.tensor([0.00001]))
        self.assertEqual(tuple(result.shape), tuple(current.shape))
        self.assertEqual(flow.seen, [(1, 16, 4, 4)])
        result.abs().mean().backward()
        gradients = [p.grad for name, p in network.named_parameters() if not name.startswith("flow_model.")]
        self.assertTrue(all(gradient is not None and bool(torch.isfinite(gradient).all())
                            for gradient in gradients))
        with self.assertRaises(ValueError):
            implementation.MSTMN(None)
        with self.assertRaises(ValueError):
            network(current, current, history[:, :2], torch.tensor([0.001]), torch.tensor([0.00001]))


if __name__ == "__main__":
    unittest.main()
