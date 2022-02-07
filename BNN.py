import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class Linear_BBB(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        # 初始化一层BBB模型参数w的分布参数(正态分布分布)
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        # test时用
        self.w = None
        self.b = None

        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, input):
        # 模型参数w采样
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)  # epsilon为noise
        # 通过重参数得到从variational posterior采样出的weight(variational posterior是w的在latent变量theta下分布，假设variation posterior服从标准正态分布)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon
        # log prior项计算：利用正态分布估计
        w_log_prior = self.prior.log_prob(self.w) # 上一步采样出的weight在定义的prior分布（高斯分布）中的概率密度/概率的对数
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior) # 求和计算出ELBO中的log prior项
        # log variational posterior项计算：用正态分布估计参数w的variational posterior
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()  # 返回w_post分布在w处的概率密度对数值

        return F.linear(input, self.w, self.b) # 对输入数据做线性变换y=xwT+b，计算得到该层网络的输出


class MLP_BBB(nn.Module):
    def __init__(self, hidden_units, noise_tol=0.1, prior_var=1.0):
        super().__init__()
        # MLP BBB的hidden层由一个in feature =1 out feature=hidden_units的linear BBB构成
        self.hidden = Linear_BBB(1, hidden_units, prior_var=prior_var)
        # MPL BBB的output层由一个in feature=hidden_units，out feature=1的linear BBB构成
        self.out = Linear_BBB(hidden_units, 1, prior_var=prior_var)
        self.noise_tol = noise_tol # 初始化noise参数，用于模型参数w的采样

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # MLP BBB hidden layer的输出（利用sigmoid作为activation function）
        x = self.out(x) # MPL BBB output layer的输出
        return x

    def log_prior(self):
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):  # samples?
        # 初始化ELBO的各项
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)  # log prior
        log_posts = torch.zeros(samples)  # log variational posterior
        log_likes = torch.zeros(samples)  # log likelihood

        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # ?
            log_priors[i] = self.log_prior()  #hidden层和output层的log prior共同构成模型ELBO（cost function）的log prior
            log_posts[i] = self.log_post()
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum()
        # monte carlo估计
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        loss = log_post - log_prior - log_like
        return loss


# 定义输出函数
def toy_function(x):
    return -x ** 4 + 3 * x ** 2 + 1


x = torch.tensor([-2, -1.8, -1, 1, 1.8, 2]).reshape(-1, 1)
y = toy_function(x)

net = MLP_BBB(32, prior_var=10)
optimizer = optim.Adam(net.parameters(), lr=0.1)

epochs = 2000
# 开始训练
for epoch in range(epochs):
    optimizer.zero_grad()

    loss = net.sample_elbo(x, y, 1)
    loss.backward()
    optimizer.step()  # 更新模型参数（每次一层的w_mu, w_rho）
    if epoch % 10 == 0:
        print('epoch:{}/{}'.format(epoch+1, epochs))
        print('Loss:', loss.item())
print('Finished Training')

samples = 100
x_tmp = torch.linspace(-5, 5, 100).reshape(-1,1)
y_samp = np.zeros((samples, 100))
for s in range(samples):
    y_tmp = net(x_tmp).detach().numpy()
    y_samp[s] = y_tmp.reshape(-1)
plt.plot(x_tmp.numpy(), np.mean(y_samp, axis=0), label='Mean Posterior Predictive')
plt.fill_between(x_tmp.numpy().reshape(-1), np.percentile(y_samp, 2.5, axis=0), np.percentile(y_samp, 97.5, axis=0), alpha=0.25, label='95% Confidence')
plt.legend()
plt.scatter(x, toy_function(x))
plt.title('Posterior Predictive')
plt.show()




