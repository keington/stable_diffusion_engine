class GaussianDiffusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t0, t1, b, h, w):
        stddev = ((t1 - t0) ** 0.5).to(torch.float32)
        return torch.randn(b, 1, h, w).to(stddev.device) * stddev.view(-1, 1, 1, 1)


class LinearDiffusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, t0, t1, alpha=0.99):
        b, c, h, w = x0.shape

        noise_scale = (1 - alpha ** (t1 - t0)) / (1 - alpha)

        eps = torch.randn(b, c, h, w).to(x0.device) * noise_scale.sqrt()

        return x0 + eps
