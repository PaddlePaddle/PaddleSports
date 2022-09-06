import paddle
import paddle.nn as nn



class Classifer_Net(nn.Layer):
    def __init__(self,channels_num,class_num):
        super().__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(channels_num,512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,class_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x = self.model(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x