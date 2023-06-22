import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载电影元数据
movies_metadata = pd.read_csv('movies.csv', usecols=['movieId', 'title'])
movie_id_to_title = dict(zip(movies_metadata['movieId'], movies_metadata['title']))

# 加载数据集
data = np.loadtxt('ratings.csv', delimiter=',', skiprows=1)
user_ids = data[:, 0]
movie_ids = data[:, 1]
ratings = data[:, 2]

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    np.column_stack((user_ids, movie_ids)), ratings, test_size=0.2, random_state=42
)

# 唯一化movie_ids，并重新构建字典
unique_movie_ids = np.unique(movie_ids)
movie_id_to_title = dict(zip(movies_metadata['movieId'], movies_metadata['title']))

# 超参数设置
num_users = int(max(user_ids)) + 1
num_movies = int(max(movie_ids)) + 1
embedding_dim = 32   #嵌入维度，用于表示用户和电影的向量表示的维度大小
hidden_dim = 32    #隐藏层维度，用于神经网络模型中隐藏层的大小
learning_rate = 0.001  #学习率，用于控制模型在训练过程中参数更新的速度
num_epochs = 20   #训练轮数，表示模型在整个训练集上的训练次数
batch_size = 128  #批量大小，用于指定每个训练批次的样本数量


# 构建模型
class Recommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, hidden_dim):
        super(Recommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        user_idx = inputs[:, 0]
        movie_idx = inputs[:, 1]
        user_embedded = self.user_embedding(user_idx)
        movie_embedded = self.movie_embedding(movie_idx)
        x = torch.cat((user_embedded, movie_embedded), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)  # 添加sigmoid函数
        return x.view(-1) * 5.0  # 将输出缩放到0到5之间


# 创建模型实例并将其移动到GPU上
model = Recommender(num_users, num_movies, embedding_dim, hidden_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 将数据转换为PyTorch张量并移动到GPU上
X_train = torch.LongTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).clamp(0, 5).to(device)  # 限制评分在0到5之间
X_test = torch.LongTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).clamp(0, 5).to(device)  # 限制评分在0到5之间

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i + batch_size]
        targets = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_rmse = np.sqrt(running_loss / len(X_train))

    # 在测试集上进行评估
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        test_loss = criterion(outputs, y_test)
        test_rmse = np.sqrt(test_loss.item() / len(X_test))

    print(f'Epoch {epoch + 1}/{num_epochs} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Test lose: {test_loss:.4f}')

# 生成推荐
user_ids = torch.LongTensor(user_ids).to(device)
movie_ids = torch.LongTensor(movie_ids).to(device)
recommendations = model(torch.column_stack((user_ids, movie_ids))).detach().cpu().numpy()

# 打印某个用户的前几个推荐电影
user_id = 1
top_k = 20
user_recommendations = recommendations[user_ids.cpu() == user_id]
top_movies = np.argsort(user_recommendations)[-top_k:][::-1]
print(f'Top {top_k} recommendations for User {user_id}:')
for movie_id in top_movies:
    movie_title = movie_id_to_title.get(unique_movie_ids[movie_id])
    print(f'Movie Title: {movie_title} | Rating: {user_recommendations[movie_id]:.2f}')
