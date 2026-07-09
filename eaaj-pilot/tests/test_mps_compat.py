"""Equivalence tests for the MPS execution workaround (CPU-checked, so they
run everywhere): the chunked kernel must match TRL's stock implementation."""
import pytest
import torch

from src.mps_compat import chunked_selective_log_softmax

trl_utils = pytest.importorskip("trl.trainer.utils")

RNG = torch.Generator().manual_seed(0)


def _rand_case(b, t, v, k=None):
    logits = torch.randn(b, t, v, generator=RNG, dtype=torch.float32)
    if k is None:
        index = torch.randint(0, v, (b, t), generator=RNG)
    else:
        index = torch.randint(0, v, (b, t, k), generator=RNG)
    return logits, index


@pytest.mark.parametrize("shape", [(2, 5, 33), (8, 170, 151), (1, 1, 7)])
def test_matches_stock_trl_squeeze_form(shape):
    logits, index = _rand_case(*shape)
    ours = chunked_selective_log_softmax(logits, index, chunk_size=16)
    stock = trl_utils.selective_log_softmax(logits, index)
    assert ours.shape == stock.shape == index.shape
    torch.testing.assert_close(ours, stock, rtol=1e-5, atol=1e-5)


def test_matches_stock_trl_topk_form():
    logits, index = _rand_case(3, 7, 29, k=4)
    ours = chunked_selective_log_softmax(logits, index, chunk_size=4)
    stock = trl_utils.selective_log_softmax(logits, index)
    torch.testing.assert_close(ours, stock, rtol=1e-5, atol=1e-5)


def test_gradients_flow_and_match():
    logits, index = _rand_case(2, 6, 41)
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    chunked_selective_log_softmax(a, index, chunk_size=3).sum().backward()
    trl_utils.selective_log_softmax(b, index).sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-5)


def test_chunk_size_does_not_change_result():
    logits, index = _rand_case(4, 9, 57)
    r1 = chunked_selective_log_softmax(logits, index, chunk_size=1)
    r2 = chunked_selective_log_softmax(logits, index, chunk_size=10_000)
    torch.testing.assert_close(r1, r2)
