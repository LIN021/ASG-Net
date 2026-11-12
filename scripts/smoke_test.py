import torch

from asgnet import ASGNet_SC
from asgnet.losses import BoundaryLaplacianLoss, fr2_loss


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    shape = (24, 24, 24)
    # Narrow stages keep the CPU check fast while exercising the complete network.
    dims = (8, 16, 24, 32)
    model = ASGNet_SC(dims=dims).train()
    data = torch.randn(1, 1, *shape)
    target = torch.zeros((1, 1, *shape), dtype=torch.long)
    target[:, :, 8:16, 8:16, 8:16] = 1

    logits, features, boundaries, diversity = model(data)
    assert logits.shape == (1, 2, *shape)
    assert features.shape == (1, dims[0], *shape)
    assert [tuple(b.shape[2:]) for b in boundaries] == [shape, (12, 12, 12), (6, 6, 6)]
    assert -1.001 <= diversity.item() <= 1.001

    boundary_loss = BoundaryLaplacianLoss()
    total = (
        torch.nn.functional.cross_entropy(logits, target[:, 0])
        + torch.stack([boundary_loss(b, target) for b in boundaries]).mean()
        + 0.3 * fr2_loss(features, target)
        + 0.001 * diversity
    )
    assert torch.isfinite(total)
    total.backward()
    gate_grad = model.encoder1[0].gate_generator[-1].weight.grad
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    assert fr2_loss(features, torch.zeros_like(target)).item() == 0

    multimodal = ASGNet_SC(in_chans=4, dims=dims).eval()
    with torch.inference_mode():
        assert model.eval()(data).shape == logits.shape
        assert multimodal(torch.randn(1, 4, *shape)).shape == logits.shape
    print("ASG-Net CPU smoke test passed.")


if __name__ == "__main__":
    main()
