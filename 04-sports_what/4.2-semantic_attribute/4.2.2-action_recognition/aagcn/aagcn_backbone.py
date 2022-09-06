# 导包
import paddle
import paddle.nn as nn
import numpy as np
import paddle as pp

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




class unit_tcn(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        " input size : (N*M, C, T, V)"
        x = self.bn(self.conv(x))
        return x



class unit_gcn(nn.Layer):
    '''
    这里是spatial adaptive graph convolutional layer加上Attention Module
    '''
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        alpha = self.create_parameter([1],default_initializer=nn.initializer.Constant(1e-4))
        self.add_parameter("alpha",alpha)
        # paddle.nn.initializer.Constant(val
        # self.alpha = nn.Parameter(paddle.zeros(1))
        pa_param = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(A.astype(np.float32)))
        self.PA = paddle.create_parameter(shape=A.shape,
                                              dtype='float32',
                                              attr=pa_param)
        self.num_subset = num_subset
        self.sigmoid = nn.Sigmoid()
        self.conv_a = nn.LayerList()
        self.conv_b = nn.LayerList()
        self.conv_d = nn.LayerList()
        # print("in_channels",in_channels)
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2D(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2D(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2D(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.down = lambda x: x


        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        num_jpts = A.shape[-1]
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        rr=2
        self.conv_sa = nn.Conv1D(out_channels, 1, ker_jpt, padding=pad)
        self.conv_ta = nn.Conv1D(out_channels, 1, 9, padding=4)
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        self.attention = True

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.PA

        y = None
        for i in range(self.num_subset):
            # print(i)
            m = self.conv_a[i](x)
            A1 = pp.transpose(m, perm=[0, 3, 1, 2]).reshape([N, V, self.inter_c * T])
            A2 = self.conv_b[i](x).reshape([N, self.inter_c * T, V])
            A1 = self.soft(pp.matmul(A1, A2) / A1.shape[-1])
            A1 = A1* self.sigmoid(self.alpha) + A[i]
            A2 = x.reshape([N, C * T, V])
            z = self.conv_d[i](pp.matmul(A2, A1).reshape([N, C, T, V]))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)

            # unified attention
            # y = y * self.Attention + y
            # y = y + y * ((a2 + a3) / 2)
            # y = self.bn(y)
        return y




class TCN_GCN_unit(nn.Layer):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class AAGCN_backbone(nn.Layer):
    def __init__(self, num_class=30, num_point=25, num_person=1, graph_args=None, in_channels=2):
        super(AAGCN_backbone, self).__init__()
        self.graph = Graph(layout='fsd10',
                 strategy='spatial')
        A = self.graph.A
        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)


        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        N, C, T, V, M = x.shape

        x = x.transpose([0, 4, 3, 1, 2]).reshape_([N, M * V * C, T])
        x = self.data_bn(x)
        x = x.reshape_([N, M, V, C, T]).transpose([0, 1, 3, 4, 2]).reshape_([N * M, C, T, V])

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

        x = self.pool(x)
        # N*M,C,T,V
        # c_new = x.shape[1]
        # x = x.reshape([N, M, c_new, -1])
        # x = x.mean(3).mean(1)

        # return self.fc(x)
        return x

