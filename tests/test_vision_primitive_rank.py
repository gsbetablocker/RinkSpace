"""SVD rank reuse preserves degeneracy and library threshold decisions."""
import pytest
torch = pytest.importorskip("torch", reason="Primitive rank checks use the configured vision runtime")

from rinkspace.vision_geometry import _primitive_design_rank


@pytest.mark.parametrize("kind", ["line", "conic"])
def test_actual_like_source_designs_have_same_library_rank(kind):
    x = torch.linspace(-2, 2, 120)
    for y in (x * .3, .4 * x.square(), torch.sqrt((4 - x.square()).clamp_min(0))):
        design = (torch.stack((x, y, torch.ones_like(x)), 1) if kind == "line" else
                  torch.stack((x*x, x*y, y*y, x, y, torch.ones_like(x)), 1))
        _, singular, _ = torch.linalg.svd(design, full_matrices=False)
        rank, _ = _primitive_design_rank(design, singular)
        assert rank == int(torch.linalg.matrix_rank(design))


def test_near_threshold_uses_original_rank_implementation():
    design = torch.diag(torch.tensor([1., 3 * torch.finfo(torch.float32).eps * 1.1, 0.]))
    _, singular, _ = torch.linalg.svd(design, full_matrices=False)
    rank, method = _primitive_design_rank(design, singular)
    assert rank == int(torch.linalg.matrix_rank(design))
    assert method == "original_matrix_rank_near_threshold"
