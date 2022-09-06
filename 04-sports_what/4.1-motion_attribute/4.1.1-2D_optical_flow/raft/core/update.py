import paddle
import paddle.nn as nn
import paddle.nn.functional as F 


# 本文件涉及了更新光流成本量的方法


class FlowHead(nn.Layer):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2D(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2D(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Layer):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = paddle.concat([h, x], axis=1)
        z = F.sigmoid(self.convz1(hx))
        r = F.sigmoid(self.convr1(hx))
        q = F.tanh(self.convq1(paddle.concat([r*h, x], axis=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = paddle.concat([h, x], axis=1)
        z = F.sigmoid(self.convz2(hx))
        r = F.sigmoid(self.convr2(hx)) # 在v1版本中，本行的convr2被写成convz2!

        q = F.tanh(self.convq2(paddle.concat([r*h, x], axis=1)))
        h = (1-z) * h + z* q
        return h


class BasicMotionEncoder(nn.Layer):
    def __init__(self, corr_levels, corr_radius):
        # args.corr_levels  args.corr_radius
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1)**2
        self.convc1 = nn.Conv2D(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2D(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2D(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2D(128, 64, 3, padding=1)
        self.conv = nn.Conv2D(64+192, 128-2, 3, padding=1)
    
    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = paddle.concat([cor, flo], axis=1)
        out = F.relu(self.conv(cor_flo))
        return paddle.concat([out, flow], axis=1)


class BasicUpdateBlock(nn.Layer):
    def __init__(self, corr_levels, corr_radius, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2D(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 64*9, 1, padding=0)
        )
    
    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = paddle.concat([inp, motion_features], axis=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow