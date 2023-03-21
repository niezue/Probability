import matplotlib.pyplot as plt
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6  # 生成一个骰子
counts = multinomial.Multinomial(15, fair_probs).sample((1000,))  # 随机取15个数，分成1000组,选取的数越多越接近0.167
cum_counts = counts.cumsum(dim=0)  # 对数值0维进行相加
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Group of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.gca().set_title('Probability')
d2l.plt.legend()
d2l.plt.show()  # 画图
