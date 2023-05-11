import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tensorboardX import SummaryWriter
from model.diffusion_model import Encoder, Decoder, DiffusionModel

# 加载数据集并进行数据增强
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = MyDataset('data.hdf5', transform=transform)
train_set, val_set, test_set = random_split(dataset, [50000, 10000, 10000])

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 创建模型、优化器和学习率调度器
encoder = Encoder(channels=3, z_dim=256)
decoder = Decoder(channels=3, z_dim=256)
model = DiffusionModel(encoder, decoder, diffusion_steps=1000, noise_schedule='linear')
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

# 定义损失函数和提前停止策略
mse_loss = nn.MSELoss(reduction='mean')
writer = SummaryWriter(log_dir=log_dir)

def loss_fn(x0, x, z_list, mu, logvar):
    recon_loss = mse_loss(x, x0)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    tv_loss = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])) + \
              torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    loss = recon_loss + kl_loss + 0.001 * tv_loss

    return loss

early_stop = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# 定义训练函数
def train(model, optimizer, scheduler, train_loader, val_loader, num_epochs, early_stop, writer, log_dir):
    best_model_path = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0

        model.train() # 设置模型为训练模式

        for i, (x0, _) in enumerate(train_loader):
            x0 = x0.cuda()

            optimizer.zero_grad()

            x, z_list = model(x0)
            mu, logvar = encoder(x0)
            loss = loss_fn(x0, x, z_list, mu, logvar)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                writer.add_scalar('train_loss', running_loss / 10, epoch*len(train_loader)+i+1)
                print('[Epoch %d, Batch %d] train_loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            scheduler.step()

        epoch_end = time.time()
        print('Epoch %d finished, time elapsed: %.2f s' % (epoch + 1, epoch_end - epoch_start))

        # 在验证集上评估模型性能
        model.eval() # 设置模型为评估模式
        val_loss = 0.0

        with torch.no_grad():
            for j, (x0_val, _) in enumerate(val_loader):
                x0_val = x0_val.cuda()

                x_val, z_list_val = model(x0_val)
                mu_val, logvar_val = encoder(x0_val)
                loss_val = loss_fn(x0_val, x_val, z_list_val, mu_val, logvar_val)

                val_loss += loss_val.item()

            val_loss /= len(val_loader)
            writer.add_scalar('val_loss', val_loss, epoch+1)

                # 检查是否需要提前停止训练
        early_stop.step(val_loss)

        if early_stop.is_better(val_loss):
            best_val_loss = val_loss
            best_model_path = os.path.join(log_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

        print('[Epoch %d] val_loss: %.3f (best: %.3f)' % (epoch + 1, val_loss, best_val_loss))

        # 在测试集上评估模型性能
        test_loss = 0.0

        with torch.no_grad():
            for k, (x0_test, _) in enumerate(test_loader):
                x0_test = x0_test.cuda()

                x_test, z_list_test = model(x0_test)
                mu_test, logvar_test = encoder(x0_test)
                loss_test = loss_fn(x0_test, x_test, z_list_test, mu_test, logvar_test)

                test_loss += loss_test.item()

            test_loss /= len(test_loader)
            writer.add_scalar('test_loss', test_loss, epoch+1)
            print('[Epoch %d] test_loss: %.3f' % (epoch + 1, test_loss))

    return best_model_path

