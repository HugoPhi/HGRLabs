import torch

class har_MLPBaseline(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(har_MLPBaseline, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平成二维向量 (batch_size, input_dim)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class har_CNNBaseline(torch.nn.Module):
    def __init__(self, num_classes):
        super(har_CNNBaseline, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=9, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(18, 36, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(36 * 32, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # 交换维度以匹配 Conv1d 输入的维度要求 (batch_size, signals, timesteps)
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平成二维向量
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class har_LSTMBaseline(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(har_LSTMBaseline, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出
        return out

class har_ResBiLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, lambda_l2):
        super(ResBiLSTM, self).__init__()
        
        # 双向LSTM层
        self.lstm1 = torch.nn.LSTM(input_size=input_dim, 
                             hidden_size=hidden_dim, 
                             num_layers=1, 
                             batch_first=True, 
                             bidirectional=True, 
                             dropout=dropout_rate)
        
        self.lstm2 = torch.nn.LSTM(input_size=hidden_dim * 2, 
                             hidden_size=hidden_dim, 
                             num_layers=1, 
                             batch_first=True, 
                             bidirectional=True, 
                             dropout=dropout_rate)
        
        self.lstm3 = torch.nn.LSTM(input_size=hidden_dim * 2, 
                             hidden_size=hidden_dim, 
                             num_layers=1, 
                             batch_first=True, 
                             bidirectional=True, 
                             dropout=dropout_rate)
        
        self.lstm4 = torch.nn.LSTM(input_size=hidden_dim * 2, 
                             hidden_size=hidden_dim, 
                             num_layers=1, 
                             batch_first=True, 
                             bidirectional=True, 
                             dropout=dropout_rate)
        
        # 批归一化层
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim * 2)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim * 2)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_dim * 2)
        
        # 输出层
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        
        # L2正则化
        self.l2_reg = lambda_l2
    
    def forward(self, x):
        # 第1层Bi-LSTM + 残差连接
        x, _ = self.lstm1(x)
        x = self.batch_norm1(x)
        x_res1 = x
        
        # 第2层Bi-LSTM + 残差连接
        x, _ = self.lstm2(torch.nn.functional.relu(x))
        x = self.batch_norm2(x)
        x += x_res1  # 残差连接
        x_res2 = x
        
        # 第3层Bi-LSTM + 残差连接
        x, _ = self.lstm3(torch.nn.functional.relu(x))
        x = self.batch_norm3(x)
        x += x_res2  # 残差连接
        
        # 第4层Bi-LSTM，没有残差连接
        x, _ = self.lstm4(torch.nn.functional.relu(x))
        
        # 输出层
        x = self.fc(x[:, -1, :])  # 只保留最后一个时间步的输出
        
        return x


