import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np
from text_convert import TextConverter
from text_dataset import TextDataset
import pdb
from my_rnn import CharRNN
from torchvision import transforms
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c

# 读数据
file_path = './/poetry.txt'
# file_path = 'D:\\myproject\\dataset\\corpus\\詩經.txt'
with open(file_path, 'r', encoding='utf8') as f:
    corpus = f.read()

"""
文本数值表示
对于每个文字，电脑并不能有效地识别，所以必须做一个转换，将文字转换成数字，
对所有非重复的字符，可以从 0 开始建立索引
"""
# pdb.set_trace()
convert = TextConverter(corpus, max_vocab=10000)

"""
构造时序样本数据
为了输入到循环神经网络中进行训练，我们需要构造时序样本的数据.
循环神经网络存在着长时依赖的问题，所以我们不能将所有的文本作为一个序列一起输入到循环神经网络中，
我们需要将整个文本分成很多很多个序列组成 batch 输入到网络中，只要我们定好每个序列的长度，
那么序列个数也就被决定了。
"""
n_step = 20

# 总的序列个数
num_seq = int(len(corpus) / n_step) # 63282/20=3164
# 去掉最后不足一个序列长度的部分
text = corpus[:num_seq*n_step]  #63280

# 接着我们将序列中所有的文字转换成数字表示，重新排列成 (num_seq x n_step) 的矩阵
arr = convert.text_to_arr(text)
arr = arr.reshape((num_seq, -1))
arr = torch.from_numpy(arr)
# pdb.set_trace()
# 据此，我们可以构建 PyTorch 中的数据读取来训练网络，
# 这里我们将最后一个字符的输出 label 定为输入的第一个字符，
# 也就是"床前明月光"的输出是"前明月光床"
train_set = TextDataset(arr)

"""
建立模型
模型可以定义成非常简单的三层，第一层是词嵌入，第二层是 RNN 层，
因为最后是一个分类问题，所以第三层是线性层，最后输出预测的字符。
"""
batch_size = 128
epochs = 200
num_classes = convert.vocab_size
embed_dim = num_classes
hidden_size = 512
num_layers = 2
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)
fea_type = 'one_hot'  # 'embed' or 'one_hot'
rnn_type = 'RNN'  # 'RNN' or 'LSTM' or 'GRU'
train_data = DataLoader(train_set, batch_size, shuffle=True)
model = CharRNN(num_classes, embed_dim, hidden_size, num_layers, dropout, device,
                fea_type=fea_type, rnn_type=rnn_type).to(device)
model = model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for e in range(epochs):
    train_loss = 0
    for data in train_data:
        x, y = data
        # pdb.set_trace()
        x = x.long().to(device)  # 128 * 20
        y = y.long().to(device)
        # x = x.to(device)
        # y = y.to(device)

        # pdb.set_trace()
        score, _ = model(x)
        loss = criterion(score, y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        # clip gradient
        nn.utils.clip_grad_norm_(model.parameters(), 1e-2)
        optimizer.step()

        train_loss += loss.item()
    print('epoch: {}, loss is: {:.3f}, perplexity is: {:.3f}'.format(e+1, train_loss, np.exp(train_loss/len(train_data))))
    losses.append(train_loss)

    """
    生成文本
    给定了开始的字符，然后不断向后生成字符，将生成的字符作为新的输入再传入网络。
    这里需要注意的是，为了增加更多的随机性，在预测的概率最高的前五个里面依据他们的概率来进行随机选择。
    """
    # pdb.set_trace()
    with torch.no_grad():
        begin = '蒹葭苍苍'
        text_len = 30
        model = model.eval()
        samples = [convert.word_to_int(c) for c in begin]
        input_txt = torch.LongTensor(samples)[None].to(device)
        _, init_state = model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]
        for i in range(text_len):
            out, init_state = model(model_input, init_state)
            # pdb.set_trace()
            pred = pick_top_n(out, top_n=5)
            # pred = out.argmax(dim=1).item()
            result.append(pred[0])
        text = convert.arr_to_text(result)
        text = text.replace('<UNK>', ' ')
        print('Generate text is: {}'.format(text))
    model.train()

# pdb.set_trace()
plt.figure()
plt.plot(losses)
plt.show()
