import re
import nltk
import tqdm
import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetBinaryClassifier

sns.set(style="whitegrid")

# 下载punkt，wordnet和stopwords数据集
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')
# 创建WordNetLemmatizer实例
lemmatizer = WordNetLemmatizer()
# 获取停用词列表
stops = set(stopwords.words("english"))

# 文本预处理函数
def cleantext(string):

    # 将字符串转换为小写并分词
    text = string.lower().split()
    # 将分词后的单词以空格分隔并连接
    text = " ".join(text)
    # 去除URL
    text = re.sub(r"http(\S)+", ' ', text)
    text = re.sub(r"www(\S)+", ' ', text)
    # 将 & 替换为 and
    text = re.sub(r"&", ' and ', text)
    text = text.replace('&amp', ' ')
    # 去除非数字、字母的字符
    text = re.sub(r"[^0-9a-zA-Z]+", ' ', text)
    # 分词
    text = text.split()
    # 对单词进行词形还原处理
    text = [lemmatizer.lemmatize(w) for w in text if not w in stops]
    # 将单词以空格分隔并连接
    text = " ".join(text)
    return text


# 读入训练数据和验证数据
train = pd.read_csv('./data/Constraint_Train.csv', header=0)
val = pd.read_csv('./data/Constraint_Val.csv', header=0)

# 对训练数据和验证数据的文本进行预处理
x_train = train['tweet'].map(lambda x: cleantext(x)).values
x_val = val['tweet'].map(lambda x: cleantext(x)).values

# 获取训练数据和验证数据的标签
y_train = train['label'].values
y_val = val['label'].values

# 对训练数据和验证数据的文本进行词频统计
vetor = CountVectorizer()
x_train = vetor.fit_transform(x_train)
x_val = vetor.transform(x_val)

# 对训练数据和验证数据的文本进行TF-IDF转换
tdidf = TfidfTransformer()
x_train = tdidf.fit_transform(x_train).toarray()
x_val = tdidf.transform(x_val).toarray()

# 将训练数据和验证数据的标签转换为数字
y_train = [0 if y_train[i] == 'fake' else 1 for i in range(len(y_train))]
y_val = [0 if y_val[i] == 'fake' else 1 for i in range(len(y_val))]
# Load test data from CSV file
test_data = pd.read_csv('./data/english_test_with_labels.csv')

# Preprocess test data
x_test = test_data['tweet'].map(lambda x: cleantext(x)).values
y_test = test_data['label'].values

# Transform the test data using CountVectorizer and TfidfTransformer
x_test = vetor.transform(x_test)
x_test = tdidf.transform(x_test).toarray()

# Convert test labels to numeric form
y_test = [0 if y_test[i] == 'fake' else 1 for i in range(len(y_test))]


# Evaluate the model on the test data
# 定义数据集类，继承自Dataset
class MyDataset(Dataset):

    # 初始化函数，传入X和Y
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    # 获取数据集长度
    def __len__(self):
        return len(self.X)

    # 获取指定索引的数据
    def __getitem__(self, item):
        return self.X[item], self.Y[item]

class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(BiLSTMAttention, self).__init__()
        # 定义双向 LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        # 定义注意力层
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 定义全连接层 1
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        # 定义全连接层 2
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 定义 Dropout 层
        self.dropout = nn.Dropout(dropout)
        # 定义 Sigmoid 层
        self.sigmoid = nn.Sigmoid()

    # 定义前向传播函数
    def forward(self, feature):
        # 运行 LSTM 层，并获取输出和隐藏状态
        packed_output, (hidden, cell) = self.lstm(feature.transpose(-1,-2))
        # 计算注意力权重
        attention_weights = torch.tanh(self.attention(packed_output))
        # 压缩维度
        attention_weights = attention_weights.squeeze(2)
        # 增加维度
        attention_weights = attention_weights.unsqueeze(1)
        # 计算上下文向量
        context_vector = torch.bmm(attention_weights, packed_output)
        # 压缩维度
        context_vector = context_vector.squeeze(1)
        # 全连接层
        hidden = self.dropout(self.fc1(context_vector))

        return self.sigmoid(self.fc2(hidden))

# 选择用于训练模型的设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 计算准确度
def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

class CustomEstimator(NeuralNetBinaryClassifier):
    def __init__(self, *args, **kwargs):
        super(CustomEstimator, self).__init__(*args, **kwargs)

    def fit(self, X, y, **fit_params):
        return super(CustomEstimator, self).fit(X, y, **fit_params)

    def evaluate(self, model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in iterator:
                text = batch[0].to(device).double()
                labels = batch[1].to(device).long()

                predictions = model(text).squeeze(1)
                loss = criterion(predictions, labels.double())
                acc = binary_accuracy(predictions, labels.double())

                epoch_loss += loss.detach()
                epoch_acc += acc.detach()

                all_preds.extend(torch.round(predictions).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return epoch_loss / len(iterator), epoch_acc / len(iterator), np.array(all_preds), np.array(all_labels)

def cross_validate(estimator, X, y, n_splits=4):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_set = MyDataset(X_train.reshape(*X_train.shape, 1), y_train)
        val_set = MyDataset(X_val.reshape(*X_val.shape, 1), y_val)

        batch_size = 64

        train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_iter = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        estimator.fit(X_train, y_train)
        train_loss, train_acc, _, _ = estimator.evaluate(estimator.module_, train_iter, estimator.criterion)
        val_loss, val_acc, val_preds, val_labels = estimator.evaluate(estimator.module_, val_iter, estimator.criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f"Fold {fold_idx + 1}: Validation accuracy: {val_acc:.2f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, title=""):
    n_epochs = len(train_losses)
    x_epochs = range(1, n_epochs + 1)

    # Plot the learning curves for loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(x_epochs, val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Epoch {title}')
    plt.legend()

    # Plot the learning curves for accuracy
    plt.subplot(1, 2, 2)
    plt.plot(x_epochs, train_accuracies, label='Training Accuracy', linewidth=2)
    plt.plot(x_epochs, val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Epoch {title}')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

batch_size = 64
n_splits=5
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
# 计算词汇量大小
vocab_size = x_train.shape[-1]
# 隐藏层维度 512, 256, 128, 64, 16
hidden_dim = 128
# 输出维度
output_dim = 1
# LSTM的层数 1, 2, 3
n_layers = 3
# dropout的比率 0.1, 0.3, 0.5
dropout = 0.5

# Instantiate your model
model = BiLSTMAttention(vocab_size, hidden_dim, output_dim, n_layers, dropout)

# Create a custom estimator
estimator = CustomEstimator(
    model,
    max_epochs=60,
    batch_size=64,
    device=device,
    criterion=nn.BCELoss,
    optimizer=optim.Adam,
    optimizer__lr=3e-4,
    callbacks=[('scheduler', MultiStepLR, {'milestones': [10, 20, 30, 40, 50], 'gamma': 0.5})]
)
# Create a DataLoader for the test data
test_set = MyDataset(x_test.reshape(*x_test.shape, 1), y_test)
test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)
criterion = nn.BCELoss().to(device)
# Create a parameter grid for hyperparameter tuning
param_grid = {
    'module__hidden_dim': [512, 256, 128, 64, 16],
     'module__n_layers': [1, 2, 3],
    'module__dropout': [0.1, 0.3, 0.5],
    'optimizer__lr': [1e-3, 3e-4, 1e-4]
}

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(estimator, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search.fit(np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)

# Use the best estimator for training
best_estimator = grid_search.best_estimator_
train_losses, val_losses, train_accuracies, val_accuracies = cross_validate(best_estimator, np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))

# Plot learning curves using the modified plot function
plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, title="Best Model")

# Evaluate the best estimator on the test data
_, _, test_preds, test_labels = best_estimator.evaluate(best_estimator.module_, test_iter, best_estimator.criterion)

# Calculate performance metrics
test_accuracy = accuracy_score(test_labels, test_preds)
test_f1_score = f1_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds)

print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test F1 Score: {test_f1_score:.2f}')
print(f'Test Precision: {test_precision:.2f}')


















