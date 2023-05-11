log_dir = 'logs'
model = DiffusionModel(...)
optimizer = optim.Adam(...)
scheduler = CosineAnnealingLR(...)
early_stop = ReduceLROnPlateau(...)
writer = SummaryWriter(log_dir=log_dir)

train(model, optimizer, scheduler, train_loader, val_loader, num_epochs, early_stop, writer, log_dir)


# 训练扩散模型
best_model_path = train(model, optimizer, scheduler, train_loader, val_loader, num_epochs, early_stop, writer, log_dir)

# 加载在验证集上性能最好的模型
model.load_state_dict(torch.load(best_model_path))

# 在测试集上评估模型性能
test_loss = evaluate(model, test_loader)
print('Test Loss: %.3f' % test_loss)
