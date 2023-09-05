import torch

from leem.similarity import cos


def test_pairwise_similarity_matrix():
    cos_sim = cos.CosineSimilarity()
    xs = torch.Tensor([[1, 2], [3, 4]])
    ys = torch.Tensor([[5, 6]])
    ret = cos_sim.pairwise_similarity_matrix(xs, ys)
    torch.testing.assert_close(ret, torch.Tensor([[0.973417], [0.998688]]))


def test_similarity_batch():
    cos_sim = cos.CosineSimilarity()
    xs = torch.Tensor([[1, 2], [3, 4]])
    ys = torch.Tensor([[5, 6], [7, 8]])
    ret = cos_sim.similarity_batch(xs, ys)
    torch.testing.assert_close(ret, torch.Tensor([0.973417, 0.997164]))
