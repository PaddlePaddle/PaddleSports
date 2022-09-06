import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

# faster debug
# paddle.set_device('cpu')

class PNet(nn.Layer):

    def __init__(self):

        super(PNet, self).__init__()

        # suppose we have input with size HxW, then
        # after first layer: H - 2,
        # after pool: ceil((H - 2)/2),
        # after second conv: ceil((H - 2)/2) - 2,
        # after last conv: ceil((H - 2)/2) - 4,
        # and the same for W

        self.features = nn.Sequential(
            ('conv1', nn.Conv2D(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2D(2, 2, ceil_mode = True)),

            ('conv2', nn.Conv2D(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),
            ('conv3', nn.Conv2D(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(num_parameters=32))
        )

        self.conv4_1 = nn.Conv2D(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2D(32, 4, 1, 1)

        weights = np.load("./pnet.npy", allow_pickle=True)[()]

        state_dict = self.state_dict()

        for key in state_dict.keys():
            if key.find('prelu')>-1:
                # prelu weight in paddle named 'features.prelu1._weight' but in torch named 'features.prelu1.weight'
                key_pytorch = key[:-7]+'weight'
                state_dict[key] = weights[key_pytorch]
            else:
                state_dict[key] = weights[key]
        self.set_dict(state_dict)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a)
        return b, a


class RNet(nn.Layer):

    def __init__(self):

        super(RNet, self).__init__()

        self.features = nn.Sequential(
            ('conv1', nn.Conv2D(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2D(3, 2, ceil_mode = True)),

            ('conv2', nn.Conv2D(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2D(3, 2, ceil_mode = True)),

            ('conv3', nn.Conv2D(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten',nn.Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        )

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load("./rnet.npy", allow_pickle=True)[()]

        # pytorch some weight receives load to paddle
        receives_list = ['features.conv4.weight',
                         'conv5_1.weight',
                         'conv5_2.weight']
        for key in receives_list:
            weights[key] = np.transpose(weights[key], (1, 0))

        state_dict = self.state_dict()

        for key in state_dict.keys():
            if key.find('prelu') > -1:
                # prelu weight in paddle named 'features.prelu1._weight' but in torch named 'features.prelu1.weight'
                key_pytorch = key[:-7] + 'weight'
                state_dict[key] = weights[key_pytorch]
            else:
                state_dict[key] = weights[key]
        self.set_dict(state_dict)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a)
        return b, a


class ONet(nn.Layer):

    def __init__(self):

        super(ONet, self).__init__()

        self.features = nn.Sequential(
            ('conv1', nn.Conv2D(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2D(3, 2, ceil_mode = True)),

            ('conv2', nn.Conv2D(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2D(3, 2, ceil_mode = True)),

            ('conv3', nn.Conv2D(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2D(2, 2, ceil_mode = True)),

            ('conv4', nn.Conv2D(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', nn.Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load("./onet.npy", allow_pickle=True)[()]
        state_dict = self.state_dict()

        # pytorch some weight receives load to paddle
        receives_list = ['features.conv5.weight',
                         'conv6_1.weight',
                         'conv6_2.weight',
                         'conv6_3.weight']
        for key in receives_list:
            weights[key] = np.transpose(weights[key],(1,0))

        for key in state_dict.keys():
            if key.find('prelu') > -1:
                # prelu weight in paddle named 'features.prelu1._weight' but in torch named 'features.prelu1.weight'
                key_pytorch = key[:-7] + 'weight'
                state_dict[key] = weights[key_pytorch]
            else:
                state_dict[key] = weights[key]
        self.set_dict(state_dict)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a)
        return c, b, a


if __name__ == '__main__':
    pnet = PNet()
    onet = ONet()
    rnet = RNet()