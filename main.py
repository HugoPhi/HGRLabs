import torch
import src.task as task
import src.model as model
import src.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 6  # UCI HAR 数据集中有 6 种活动类型

# 使用 MLP 作为基线模型
# model = model.har_MLPBaseline(input_dim=128*9, num_classes=num_classes).to(device)
# 或者使用 CNN
# model = model.har_CNNBaseline(num_classes=num_classes).to(device)
# 或者使用 LSTM
# model = model.har_LSTMBaseline(input_size=9, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)

model = model.har_ResBiLSTM(input_size=9, hidden_size=64, num_layers=1, num_classes=6)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example usage:
data_dir = './data/UCIHAR'
train_loader = data.get_dataloader(data_dir, batch_size=32, split='train')
test_loader = data.get_dataloader(data_dir, batch_size=32, split='test')

# 训练模型
task.train(model, train_loader, criterion, optimizer, num_epochs=25)

# 评估模型
task.evaluate(model, test_loader)
