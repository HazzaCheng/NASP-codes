import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_NORMAL, PRIMITIVES_REDUCE, PARAMS
from genotypes import Genotype
import pdb


class MixedOp(nn.Module):

    def __init__(self, C, stride, reduction):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        if reduction:
            primitives = PRIMITIVES_REDUCE
        else:
            primitives = PRIMITIVES_NORMAL
        for primitive in primitives:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, updateType):
        if updateType == "weights":
            # w为0的话就不计算op
            result = [w * op(x) if w.data.cpu().numpy()[0] else w for w, op in zip(weights, self._ops)]
        else:
            result = [w * op(x) for w, op in zip(weights, self._ops)]
        return sum(result)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        # 前两层固定不参与搜索，根据之前所连cell类型进行设定
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        # steps是4代表cell里有四个node需要搜索ope
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        # 这里steps是4原因是每个cell一共7个nodes，两个作为input即C_prev_prev和C_prev，一个作为output即C
        for i in range(self._steps):
            # j = 2,3,4,5 因为每处理一个节点，该节点就变为下一个节点的前继
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                # 初始化得到一个节点到另一个节点的操作集合
                op = MixedOp(C, stride, reduction)
                # step是4时，共用14个连接
                self._ops.append(op)
        '''
        self._ops[0, 1] 代表的是内部节点0的前继操作
        self._ops[2, 3, 4] 代表的是内部节点1的前继操作
        self._ops[5, 6, 7, 8] 代表的是内部节点2的前继操作
        self._ops[9, 10, 11, 12 ,13] 代表的是内部节点3的前继操作
        '''

    def forward(self, s0, s1, weights, updateType):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            # 因为先将该节点到另一个节点各操作后的输出相加，再把该节点与所有前继节点的操作相加，所以输出维度不变
            # 虽然这里是将所有前继结点的op相加，但实际上因为前面的约束选取的weights里只有两行不是全为0
            s = sum(self._ops[offset + j](h, weights[offset + j], updateType) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        '''
        i=1:  s2 = self._ops[0](s0,weights[0]) + self._ops[1](s1,weights[1])即内部节点0
        i=2:  s3 = self._ops[2](s0,weights[2]) + self._ops[3](s1,weights[3]) + self._ops[4](s2,weights[4])即内部节点1
        i=3、4依次计算得到s4，s5
        由此可知len(weights)也等于14，因为有8个操作，所以weight[i]有8个值
        '''

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, greedy=0, l2=0, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._greedy = greedy
        self._l2 = l2
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # 缩减一次通道数乘2
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        # 自适应平均池化层，不改变通道大小，每个通道只保留一个平均值
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        self.saved_params = []
        for w in self._arch_parameters:
            temp = w.data.clone()
            self.saved_params.append(temp)

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, updateType="weights"):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas_reduce
            else:
                weights = self.alphas_normal
            s0, s1 = s1, cell(s0, s1, weights, updateType)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target, updateType):
        logits = self(input, updateType)
        return self._criterion(logits, target) + self._l2_loss()

    # 对应论文里限制模型复杂度的部分R(A)，只对normal cell做了限制
    def _l2_loss(self):
        normal_burden = []
        params = 0
        for key in PRIMITIVES_NORMAL:
            params += PARAMS[key]
        for key in PRIMITIVES_NORMAL:
            normal_burden.append(PARAMS[key] / params)
        normal_burden = torch.autograd.Variable(torch.Tensor(normal_burden).cuda(), requires_grad=False)
        return (self.alphas_normal * self.alphas_normal * normal_burden).sum() * self._l2

    def _initialize_alphas(self):
        # 初始化A矩阵
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops_normal = len(PRIMITIVES_NORMAL)
        num_ops_reduce = len(PRIMITIVES_REDUCE)
        self.alphas_normal = Variable(torch.ones(k, num_ops_normal).cuda() / 2, requires_grad=True)
        self.alphas_reduce = Variable(torch.ones(k, num_ops_reduce).cuda() / 2, requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def save_params(self):
        for index, value in enumerate(self._arch_parameters):
            self.saved_params[index].copy_(value.data)

    def clip(self):
        clip_scale = []
        m = nn.Hardtanh(0, 1)
        for index in range(len(self._arch_parameters)):
            clip_scale.append(m(Variable(self._arch_parameters[index].data)))
        for index in range(len(self._arch_parameters)):
            self._arch_parameters[index].data = clip_scale[index].data

    def binarization(self, e_greedy=0):
        self.save_params()
        for index in range(len(self._arch_parameters)):
            m, n = self._arch_parameters[index].size()
            if np.random.rand() <= e_greedy:
                maxIndexs = np.random.choice(range(n), m)
            else:
                maxIndexs = self._arch_parameters[index].data.cpu().numpy().argmax(axis=1)
            self._arch_parameters[index].data = self.proximal_step(self._arch_parameters[index], maxIndexs)

    def restore(self):
        for index in range(len(self._arch_parameters)):
            self._arch_parameters[index].data = self.saved_params[index]

    # def proximal(self):
    #     for index in range(len(self._arch_parameters)):
    #         self._arch_parameters[index].data = self.proximal_step(self._arch_parameters[index])

    def proximal_step(self, var, maxIndexs=None):
        values = var.data.cpu().numpy()
        m, n = values.shape
        alphas = []
        # 每行最大的alpha赋值为1，其余赋值为0
        for i in range(m):
            for j in range(n):
                if j == maxIndexs[i]:
                    alphas.append(values[i][j].copy())
                    values[i][j] = 1
                else:
                    values[i][j] = 0
        step = 2
        cur = 0
        while cur < m:
            cur_alphas = alphas[cur:cur + step]
            # 只保留两个最大值
            reserve_index = [v[0] for v in sorted(list(zip(range(len(cur_alphas)), cur_alphas)), key=lambda x: x[1],
                                                  reverse=True)[:2]]
            for index in range(cur, cur + step):
                if (index - cur) in reserve_index:
                    continue
                else:
                    values[index] = np.zeros(n)
            cur = cur + step
            step += 1
        return torch.Tensor(values).cuda()

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights, primitives):
            gene = []
            n = 2
            start = 0
            # [0,1,2,3] 代表了中间节点
            for i in range(self._steps):
                end = start + n
                # 获取当前中间节点至前继节点的权重
                W = weights[start:end].copy()
                # sorted返回按每条edge的最大alpha降序排列的前继节点的索引，去两条
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            # 找出与每个前继节点op中alpha值最大的操作索引
                            k_best = k
                    gene.append((primitives[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), PRIMITIVES_NORMAL)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), PRIMITIVES_REDUCE)

        # 计算出来是四个内部节点 [2,3,4,5]
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        # 生成Cell结构
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat
        )
        return genotype
