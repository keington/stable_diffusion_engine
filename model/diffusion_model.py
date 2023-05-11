import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.diffusion_util import LinearDiffusion, GaussianDiffusion


class Encoder(nn.Module):
    def __init__(self, channels, z_dim):
        super().__init__()

        # 定义编码器网络结构
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 256, z_dim) # 均值
        self.fc2 = nn.Linear(4 * 4 * 256, z_dim) # 方差

    def forward(self, x):
        # 编码器前向传播过程
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = x.view(-1, 4 * 4 * 256) # 展开成一维向量
        mu = self.fc1(x) # 均值
        logvar = self.fc2(x) # 方差

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, channels, z_dim):
        super().__init__()

        # 定义解码器网络结构
        self.fc = nn.Linear(z_dim, 4 * 4 * 256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # 解码器前向传播过程
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        return x


class DiffusionModel(nn.Module):
    def __init__(self, encoder, decoder, diffusion_steps, noise_schedule):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.diffusion_steps = diffusion_steps

        # 初始化扩散和解扩散算法
        self.diffusion = GaussianDiffusion()
        self.solver = LinearDiffusion()

        self.noise_schedule = noise_schedule

    def forward(self, x0, timesteps=None):
        b, c, h, w = x0.shape

        if timesteps is None:
            timesteps = torch.arange(0., 1., 1. / self.diffusion_steps).to(x0.device)

        # 获取初始状态下的均值和方差
        z_mu, z_logvar = self.encoder(x0)
        z_stddev = (0.5 * z_logvar).exp()

        z_list = []
        for _ in range(self.diffusion_steps):
            # 从标准正态分布中采样出随机噪声
            eps = torch.randn_like(z_mu)
            # 应用重参数方法得到z
            z = z_mu + z_stddev * eps
            z_list.append(z)

            # 计算当前时间步的掩码
            mask = self.diffusion(timesteps[_], timesteps[_ + 1], b, h, w).to(x0.device)
            # 计算解扩散时的新的均值和方差
            z_mu_new, z_logvar_new = self.encoder(self.decoder(z))
            z_stddev_new = (0.5 * z_logvar_new).exp()
            dz = (z_mu_new - z).div(z_stddev)
            z = z + mask * dz

        x = self.decoder(z)

        return x, z_list
