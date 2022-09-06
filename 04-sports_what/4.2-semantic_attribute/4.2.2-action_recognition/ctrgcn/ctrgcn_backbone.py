# 导包
import paddle
import paddle.nn as nn
import numpy as np
import paddle as pp
from weight_init import weight_init_
def zero(x):
    return 0


def iden(x):
    return x


def einsum(x, A):
    """paddle.einsum will be implemented in release/2.2.
    self.kernel_size = 3
    x.shape = (n, self.kernel_size, (kc // self.kernel_size) ==out_channels, t, v)
    """
    x = x.transpose((0, 2, 3, 1, 4))
    n, c, t, k, v = x.shape
    k2, v2, w = A.shape
    assert (k == k2 and v == v2), "Args of einsum not match!"
    x = x.reshape((n, c, t, k * v))
    A = A.reshape((k * v, w))
    y = paddle.matmul(x, A)
    return y


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


class Graph():
    '''
    记录数据集的骨骼点连接信息，就是如果还是基于骨骼点进行动作识别，不同数据集骨骼点标注不同，变动的就是这个部分
    '''
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'fsd10':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)]
            self.edge = self_link + neighbor_link
            self.center = 8
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                              (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                              (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")



def conv_init(conv):
    if conv.weight is not None:
        weight_init_(conv.weight, 'kaiming_normal_', mode='fan_in')
    if conv.bias is not None:
        nn.initializer.Constant(value=0.0)(conv.bias)


def bn_init(bn, scale):
    nn.initializer.Constant(value=float(scale))(bn.weight)
    nn.initializer.Constant(value=0.0)(bn.bias)


def einsum(x1, x3):
    """paddle.einsum only support in dynamic graph mode.
    x1 : n c u v
    x2 : n c t v
    """
    n, c, u, v1 = x1.shape
    n, c, t, v3 = x3.shape
    assert (v1 == v3), "Args of einsum not match!"
    x1 = paddle.transpose(x1, perm=[0, 1, 3, 2])  # n c v u
    y = paddle.matmul(x3, x1)
    # out: n c t u
    return y


class CTRGC(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 rel_reduction=8,
                 mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # if in_channels == 3 or in_channels == 9:
        #     self.rel_channels = 8
        #     self.mid_channels = 16
        # else:
        #     self.rel_channels = in_channels // rel_reduction
        #     self.mid_channels = in_channels // mid_reduction

        self.rel_channels = 8
        self.mid_channels = 16

        # print(self.in_channels,self.rel_channels)
        self.conv1 = nn.Conv2D(self.in_channels,
                               self.rel_channels,
                               kernel_size=1)
        self.conv2 = nn.Conv2D(self.in_channels,
                               self.rel_channels,
                               kernel_size=1)
        self.conv3 = nn.Conv2D(self.in_channels,
                               self.out_channels,
                               kernel_size=1)
        self.conv4 = nn.Conv2D(self.rel_channels,
                               self.out_channels,
                               kernel_size=1)
        self.tanh = nn.Tanh()

    def init_weights(self):
        """Initiate the parameters.
        """
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2D):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        #x.shape = []
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(
            x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (
            A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        # We only support 'paddle.einsum()' in dynamic graph mode, if use in infer model please implement self.
        # x1 = paddle.einsum('ncuv,nctv->nctu', x1, x3)
        x1 = einsum(x1, x3)
        return x1


class TemporalConv(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1),
                              dilation=(dilation, 1))

        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super(MultiScale_TemporalConv, self).__init__()
        assert out_channels % (
            len(dilations) +
            2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(in_channels,
                          branch_channels,
                          kernel_size=1,
                          padding=0),
                nn.BatchNorm2D(branch_channels),
                nn.ReLU(),
                TemporalConv(branch_channels,
                             branch_channels,
                             kernel_size=ks,
                             stride=stride,
                             dilation=dilation),
            ) for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2D(in_channels,
                          branch_channels,
                          kernel_size=1,
                          padding=0), nn.BatchNorm2D(branch_channels),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=(3, 1),
                             stride=(stride, 1),
                             padding=(1, 0)), nn.BatchNorm2D(branch_channels)))

        self.branches.append(
            nn.Sequential(
                nn.Conv2D(in_channels,
                          branch_channels,
                          kernel_size=1,
                          padding=0,
                          stride=(stride, 1)), nn.BatchNorm2D(branch_channels)))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels,
                                         out_channels,
                                         kernel_size=residual_kernel_size,
                                         stride=stride)

    def init_weights(self):
        """Initiate the parameters.
        """
        # initialize
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2D):
                weight_init_(m.weight, 'Normal', std=0.02, mean=1.0)
                nn.initializer.Constant(value=0.0)(m.bias)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = paddle.concat(branch_outs, axis=1)
        out += res
        return out


class unit_tcn(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 coff_embedding=4,
                 adaptive=True,
                 residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.LayerList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 1),
                    nn.BatchNorm2D(out_channels))
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            pa_param = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(A.astype(np.float32)))
            self.PA = paddle.create_parameter(shape=A.shape,
                                              dtype='float32',
                                              attr=pa_param)
        else:
            A_tensor = paddle.to_tensor(A, dtype="float32")
            self.A = paddle.create_parameter(
                shape=A_tensor.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(A_tensor))
            self.A.stop_gradient = True
        alpha_tensor = paddle.to_tensor(np.zeros(1), dtype="float32")
        self.alpha = paddle.create_parameter(
            shape=alpha_tensor.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(alpha_tensor))
        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2D):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class TCN_GCN_unit(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 adaptive=True,
                 kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y



class CTRGCN_backbone(nn.Layer):
    """
    CTR-GCN model from:
    `"Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition" <https://arxiv.org/abs/2107.12213>`_
    Args:
        num_point: int, numbers of sketeton point.
        num_person: int, numbers of person.
        base_channel: int, model's hidden dim.
        graph: str, sketeton adjacency matrix name.
        graph_args: dict, sketeton adjacency graph class args.
        in_channels: int, channels of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default 3.
        adaptive: bool, if adjacency matrix can adaptive.
    """

    def __init__(self,
                 num_point=25,
                 num_person=1,
                 base_channel=64,
                 graph='ntu_rgb_d',
                 in_channels=2,
                 adaptive=True):
        super(CTRGCN_backbone, self).__init__()


        self.graph = Graph(layout='fsd10',
                 strategy='spatial')
        A = self.graph.A  # 3,25,25

        self.num_point = num_point
        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)
        self.base_channel = base_channel

        self.l1 = TCN_GCN_unit(in_channels,
                               self.base_channel,
                               A,
                               residual=False,
                               adaptive=adaptive)
        self.l2 = TCN_GCN_unit(self.base_channel,
                               self.base_channel,
                               A,
                               adaptive=adaptive)
        self.l3 = TCN_GCN_unit(self.base_channel,
                               self.base_channel,
                               A,
                               adaptive=adaptive)
        self.l4 = TCN_GCN_unit(self.base_channel,
                               self.base_channel,
                               A,
                               adaptive=adaptive)
        self.l5 = TCN_GCN_unit(self.base_channel,
                               self.base_channel * 2,
                               A,
                               stride=2,
                               adaptive=adaptive)
        self.l6 = TCN_GCN_unit(self.base_channel * 2,
                               self.base_channel * 2,
                               A,
                               adaptive=adaptive)
        self.l7 = TCN_GCN_unit(self.base_channel * 2,
                               self.base_channel * 2,
                               A,
                               adaptive=adaptive)
        self.l8 = TCN_GCN_unit(self.base_channel * 2,
                               self.base_channel * 4,
                               A,
                               stride=2,
                               adaptive=adaptive)
        self.l9 = TCN_GCN_unit(self.base_channel * 4,
                               self.base_channel * 4,
                               A,
                               adaptive=adaptive)
        self.l10 = TCN_GCN_unit(self.base_channel * 4,
                                self.base_channel * 4,
                                A,
                                adaptive=adaptive)

    def init_weights(self):
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.shape
        x = paddle.transpose(x, perm=[0, 4, 3, 1, 2])
        x = paddle.reshape(x, (N, M * V * C, T))

        x = self.data_bn(x)

        x = paddle.reshape(x, (N, M, V, C, T))
        x = paddle.transpose(x, perm=(0, 1, 3, 4, 2))

        x = paddle.reshape(x, (N * M, C, T, V))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        return x, N, M