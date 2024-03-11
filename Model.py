from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
# from torch_sparse import spspmm
from torch_sparse import spmm
import torch

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
class LocalGraph(nn.Module):
    def __init__(self):
        super(LocalGraph, self).__init__()
    
    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + noise
    
    def forward(self, allOneAdj, embeds):
        # allOneAdj should be without self-loop
        # embeds should be zero-order embeds
        order = t.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = t.spmm(allOneAdj, embeds) - embeds
        fstNum = order
        scdEmbeds = (t.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
        scdNum = (t.spmm(allOneAdj, fstNum) - fstNum) - order
        subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
        embeds = F.normalize(embeds, p=2)
        scores = t.sum(subgraphEmbeds * embeds, dim=-1)
        scores = self.makeNoise(scores)
        _, seeds = t.topk(scores, args.seedNum)
        return scores, seeds

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        # self.gcnLayers = nn.Sequential(MixProp(args.latdim, args.latdim, args.gnn_layer, args.dropout, args.propalpha))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        self.edgeDropper = SpAdjDropEdge(args.keepRate)

    def getEgoEmbeds(self, adjlist,l):
        # adj = adjlist[0]
        # print("adjegho:", adj.size())
        uEmbeds, iEmbeds = self.forward(adjlist,l)
        return t.concat([uEmbeds, iEmbeds], axis=0)

    def forward(self, adjlist, keepRate=1.0, l=1):
        adj = adjlist[0]
        iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
        
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            # print("gcn embeds:", embeds)
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)# / len(embedsLst)
        if keepRate == 1.0 and l==1:
            return mainEmbeds[:args.user], mainEmbeds[args.user:]
        if keepRate == 1.0 and l==3:
            adj1 = adjlist[1]
            adj2 = adjlist[2]
            iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adj1, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView1 = sum(embedsLst)# / len(embedsLst)
            for gcn in self.gcnLayers:
                embeds = gcn(adj2, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView2 = sum(embedsLst)# / len(embedsLst)
            return mainEmbeds[:args.user], mainEmbeds[args.user:], embedsView1[:args.user], embedsView1[args.user:], embedsView2[:args.user], embedsView2[args.user:]
        # for edge drop
        if args.aug_data == 'ed' or args.aug_data == 'ED':
            adjView1 = self.edgeDropper(adj, keepRate)
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adjView1, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView1 = sum(embedsLst)

            adjView2 = self.edgeDropper(adj, keepRate)
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adjView2, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView2 = sum(embedsLst)
        # for random walk
        elif args.aug_data == 'rw' or args.aug_data == 'RW':
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                temadj = self.edgeDropper(adj, keepRate)
                embeds = gcn(temadj, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView1 = sum(embedsLst)

            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                temadj = self.edgeDropper(adj, keepRate)
                embeds = gcn(temadj, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView2 = sum(embedsLst)
        # for node drop
        elif args.aug_data == 'nd' or args.aug_data == 'ND':
            rdmMask = (t.rand(iniEmbeds.shape[0]) < keepRate) * 1.0
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adj, embedsLst[-1] * rdmMask)
                embedsLst.append(embeds)
            embedsView1 = sum(embedsLst)

            rdmMask = (t.rand(iniEmbeds.shape[0]) < keepRate) * 1.0
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adj, embedsLst[-1] * rdmMask)
                embedsLst.append(embeds)
            embedsView2 = sum(embedsLst)
        return mainEmbeds[:args.user], mainEmbeds[args.user:], embedsView1[:args.user], embedsView1[args.user:], embedsView2[:args.user], embedsView2[args.user:]
from torch_sparse import coalesce

# index = torch.tensor([[1, 0, 1, 0, 2, 1],
#                       [0, 1, 1, 1, 0, 0]])
# value = torch.Tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

    def forward(self, adj, embeds):
        idxs = adj._indices()
        vals = adj._values()
        index, value = coalesce(idxs, vals, m=adj.size(0), n=adj.size(1))
        return self.act(spmm(index, value, adj.size(0), adj.size(1), embeds))
        # return t.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)
class RandomMaskSubgraphs(nn.Module):
    def __init__(self):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
    
    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        maskNodes = [seeds]

        for i in range(args.maskDepth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                idct = t.logical_or(rowIdct, colIdct)

                if i != args.maskDepth - 1:
                    mskRows = rows[idct]
                    mskCols = cols[idct]
                    nxtSeeds.append(mskRows)
                    nxtSeeds.append(mskCols)

                rows = rows[t.logical_not(idct)]
                cols = cols[t.logical_not(idct)]
            if len(nxtSeeds) > 0:
                nxtSeeds = t.unique(t.concat(nxtSeeds))
                maskNodes.append(nxtSeeds)
        sampNum = int((args.user + args.item) * args.keepRate)
        sampedNodes = t.randint(args.user + args.item, size=[sampNum]).cuda()
        if self.flag == False:
            l1 = adj._values().shape[0]
            l2 = rows.shape[0]
            print('-----')
            print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
            tem = t.unique(t.concat(maskNodes))
            print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (args.user + args.item)), tem.shape[0], (args.user + args.item))
        maskNodes.append(sampedNodes)
        maskNodes = t.unique(t.concat(maskNodes))
        if self.flag == False:
            print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (args.user + args.item)), maskNodes.shape[0], (args.user + args.item))
            self.flag = True
            print('-----')

        
        encoderAdj = self.normalizeAdj(t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

        temNum = maskNodes.shape[0]
        temRows = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]
        temCols = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]

        newRows = t.concat([temRows, temCols, t.arange(args.user+args.item).cuda(), rows])
        newCols = t.concat([temCols, temRows, t.arange(args.user+args.item).cuda(), cols])

        # filter duplicated
        hashVal = newRows * (args.user + args.item) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (args.user + args.item)
        newRows = ((hashVal - newCols) / (args.user + args.item)).long()


        decoderAdj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(), adj.shape)
        return encoderAdj, decoderAdj