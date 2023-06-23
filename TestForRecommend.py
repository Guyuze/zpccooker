import unittest
import torch
from NCFRecommend import Recommender
import torch.nn as nn
import NCFRecommend as ncf
import matplotlib.pyplot as plt


class TestRecommender(unittest.TestCase):

    def setUp(self):
        # 创建模型实例
        self.model = Recommender(num_users=100, num_movies=5000, embedding_dim=64, hidden_dim=64, lambda_reg=0.001)


    def test_forward(self):
        inputs = torch.tensor([[1, 1], [2, 2]])
        expected_output_shape = torch.Size([2])
        output = self.model(inputs)
        self.assertEqual(output.shape, expected_output_shape)

    def test_loss_function(self):
        criterion = nn.MSELoss()
        inputs = torch.tensor([[1, 1], [1, 367]])
        targets = torch.tensor([4.0, 4.0])
        output = self.model(inputs)
        loss = criterion(output, targets)
        expected_loss = torch.tensor(0.7)  # 假设的期望损失
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=0)

    def test_backward(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        inputs = torch.tensor([[1, 1], [2, 1704]])
        targets = torch.tensor([4.0, 4.5])
        optimizer.zero_grad()
        output = self.model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        # 检查模型的某个参数是否已经更新
        param = list(self.model.parameters())[0]
        self.assertNotEqual(param.grad.sum(), 0)
