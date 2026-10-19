"""A known absent role does not turn unlabelled material into background."""
import pytest

torch = pytest.importorskip("torch")

from rinkspace.vision import semantic_exclusion_loss


CLASSES = ("background", "ice_contact", "board_top", "ice_paint")


def test_exclusion_reduces_rejected_probability_without_setting_background():
    values = torch.tensor([0., 3., 2., -1.]).reshape(1, 4, 1, 1).requires_grad_()
    mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    before = values.softmax(1).detach()
    loss, stats = semantic_exclusion_loss(values, mask, ["ice_contact"], CLASSES)
    expected = -torch.log(1-before[:, 1]).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert values.grad[0, 1, 0, 0] > 0
    assert torch.all(values.grad[0, [0, 2, 3], 0, 0] < 0)
    assert not stats["positive_semantic_class_assigned"]
    assert not stats["primitive_class_assigned"]
    # The target is a set of still-possible roles, including the real board top.
    assert stats["excluded_kinds"] == ["ice_contact"]


def test_extreme_false_confidence_has_finite_loss_and_gradient():
    values = torch.tensor([0., 1000., -1000., 1.]).reshape(1, 4, 1, 1).requires_grad_()
    loss, _ = semantic_exclusion_loss(values, torch.ones((1, 1, 1, 1), dtype=torch.bool),
                                      ["ice_contact"], CLASSES)
    assert torch.isfinite(loss) and loss > 900
    loss.backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[0, 1, 0, 0] == 1


def test_confirmed_subpixel_position_protects_all_bilinear_source_pixels():
    values = torch.zeros((1, 4, 3, 3), requires_grad=True)
    mask = torch.zeros((1, 1, 3, 3), dtype=torch.bool)
    mask[0, 0, 1, 1] = True
    human = dict(kind="ice_contact", state="user_confirmed_visual_supervision", uv=[[.75, .75]])
    with pytest.raises(ValueError, match="confirmed human source positions"):
        semantic_exclusion_loss(values, mask, ["ice_contact"], CLASSES, [human])
    mask.zero_()
    mask[0, 0, 2, 2] = True
    loss, stats = semantic_exclusion_loss(values, mask, ["ice_contact"], CLASSES, [human])
    assert torch.isfinite(loss) and stats["supervised_pixels"] == 1
    assert human["uv"] == [[.75, .75]]


@pytest.mark.parametrize("roles", [[], ["missing"], ["ice_contact", "ice_contact"], list(CLASSES)])
def test_exclusion_cannot_reject_unknown_roles_or_every_role(roles):
    with pytest.raises(ValueError, match="at least one allowed role"):
        semantic_exclusion_loss(torch.zeros((1, 4, 1, 1)),
                                torch.ones((1, 1, 1, 1), dtype=torch.bool), roles, CLASSES)


def test_unreviewed_pixels_have_no_exclusion_gradient():
    values = torch.randn((1, 4, 3, 4), requires_grad=True)
    loss, stats = semantic_exclusion_loss(values, torch.zeros((1, 1, 3, 4), dtype=torch.bool),
                                          ["ice_contact"], CLASSES)
    loss.backward()
    assert loss == 0 and stats["supervised_pixels"] == 0
    assert torch.count_nonzero(values.grad) == 0
