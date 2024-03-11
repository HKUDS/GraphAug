import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, RandomMaskSubgraphs, LocalGraph
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import calcRegLoss, contrastLoss, pairPredict
import os
import torch
from layers import *
from view_learner import *
from torch_sparse import SparseTensor
from torch.utils.data.dataloader import default_collate
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
            print()
        
        self.graph_learner = GraphLearner(input_size=args.latdim, hidden_size=args.latdim,
                                          graph_type=args.graph_type, top_k=args.top_k,
                                          epsilon=args.epsilon, num_pers=args.num_per, metric_type=args.graph_metric_type,
                                          feature_denoise=args.feature_denoise, device=args.gpu)
        
        self.backbone = args.backbone

        if self.backbone == "GCN":
            self.backbone_gnn = myGCN(args, in_dim=args.latdim, out_dim=args.IB_size*2,
                                      hidden_dim=args.latdim).cuda()
        elif self.backbone == "GIN":
            self.backbone_gnn = myGIN(args, in_dim=args.latdim, out_dim=args.IB_size*2,
                                      hidden_dim=args.latdim).cuda()
        elif self.backbone == "GAT":
            self.backbone_gnn = myGAT(args, in_dim=args.latdim, out_dim=args.IB_size*2,
                                      hidden_dim=args.latdim).cuda()
        elif self.backbone == "mixhop":
            self.backbone_gnn = MixHopNetwork(args, feature_number = args.latdim, class_number = args.IB_size*2)
        self.view_learner = ViewLearner(self.backbone_gnn, mlp_edge_model_dim = args.latdim)

        self.IB_size = args.IB_size    

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        
        
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        import time
        start_time = time.time()
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory()
        print("------training time------:", (time.time()-start_time)/args.epoch)    
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()
    
    def learn_graph(self, node_features, edge_index):
        # print("node_features:", node_features[:10])
        # print("edge_index:", edge_index[:10])
        new_feature, new_adj = self.graph_learner(node_features, edge_index)
        return new_feature, new_adj

    def prepareModel(self):
        self.model = Model().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs()
        self.sampler = LocalGraph()
    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std
    def transsparse(self, mat, edge_index, s):
        # mat = self.handler.normalizeAdj(mat.cpu().detach().numpy())
        # print("mat：", mat.size())
        # println()
        # idxs = edge_index[:,int(0.2*(edge_index.size()[1])):]
        # vals = mat[int(0.2*(mat.size()[0])):]
        idxs = edge_index
        vals = mat
        # print("&&&:", idxs[:,20000:20300], vals[20000:20300])
        # println()
        shape = torch.Size((s, s))
        new_adj =  torch.sparse.FloatTensor(idxs, vals, shape).cuda()
       
        return new_adj
    def sim_loss(self, x1, x2):
        simi_loss = F.l1_loss(x1, x2)
        simi_loss = torch.exp(1 - simi_loss)
        return simi_loss
    def calc_loss(self, x, x_aug, temperature=0.2, sym=True):
        # x and x_aug shape -> Batch x proj_hidden_dim

        batch_size,_ = x.size()
        # print("batch_size:", batch_size)
        # print("x:", x.size())
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1)/2.0
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1

        return loss
    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for i, tem in enumerate(trnLoader):
            # kl_loss = []
            adjlist = {}
            adjlist[0] =  self.handler.torchBiAdj
            # adjlist[0] =  self.handler.allOneAdj
            shape = self.handler.torchBiAdj.size()

            o_edge_index, new_edge_attr = self.handler.torchBiAdj._indices(), self.handler.torchBiAdj._values()
            
            # print(new_edge_index.size())
            # println()
            ofea = self.model.getEgoEmbeds(adjlist, 1)
            # number = int(0.25*(o_edge_index.size()[1]))
            number = int(100000)
            rdmUsrs = t.randint(args.user, [number])#ancs
            rdmItms1 = t.randint_like(rdmUsrs, args.item)
            
            new_idxs = default_collate([rdmUsrs,rdmItms1])
            
            new_vals = t.tensor([0.05]*number)

           
            new_graphs_list = []
            node_embs = []
            for j in range(args.gen):
                # new_feature, new_adj = self.learn_graph(node_features=ofea, edge_index = o_edge_index)
                # new_adj = self.transsparse(new_adj, o_edge_index, ofea.size()[0])
                    
                # com_adj = add_new + new_adj
                
                # # new_edge_index, new_edge_attr = new_adj._indices(), new_adj._values()
                # new_edge_index, new_edge_attr = com_adj._indices(), com_adj._values()
                # adjlist[1] = com_adj

                # new_graph = Data(x=new_feature, edge_index=new_edge_index, edge_attr=new_edge_attr)
                # new_graphs_list.append(new_graph)
                
                # loader = DataLoader(new_graphs_list, batch_size=len(new_graphs_list))
                # batch_data = next(iter(loader))
                # node_embs, _ = self.backbone_gnn(batch_data.x, batch_data.edge_index)
                # another generator to generate graph
                add_new = t.sparse.FloatTensor(new_idxs, new_vals, shape).cuda()
                ant_node, ant_adj = self.view_learner(ofea, o_edge_index, self.handler.torchBiAdj)
                new_adjs = self.transsparse(ant_adj, o_edge_index, ofea.size()[0])
                com_adj_ant = new_adjs + add_new
                # com_adj_ant = new_adjs
                new_edge_index, new_edge_attr = com_adj_ant._indices(), com_adj_ant._values()
                adjlist[j+1] = com_adj_ant
                node_embs.append(ant_node)
            node_embs = t.mean(t.stack(node_embs, 0), dim=0)
            # print("node_embs:", node_embs.size())
            # pritnln()
            mu = node_embs[:, :self.IB_size]
            std = F.softplus(node_embs[:, self.IB_size:]-self.IB_size, beta=1)
            num_sample = 2
            new_node_embs = self.reparametrize_n(mu, std, num_sample)
            
            # pos_emb = new_node_embs[::2 ]
            # neg_emb = new_node_embs[1::2]

            klloss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
            
            # simloss = self.sim_loss(pos_emb, neg_emb)
            
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()
            
            # usrEmbeds, itmEmbeds, usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model(self.handler.torchBiAdj, args.keepRate)
            usrEmbeds, itmEmbeds, usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model(adjlist, args.keepRate, 3)
            
            
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]
            
            simloss_u1 = self.sim_loss(usrEmbeds1, usrEmbeds2)
            simloss_e1 = self.sim_loss(itmEmbeds1, itmEmbeds2)
            clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg            
            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = - (scoreDiff.sigmoid()+ 1e-8).log().sum() / args.batch
            regLoss = calcRegLoss(self.model) * args.reg

            # loss = bprLoss + regLoss + clLoss + 0.00001*klloss + 60*simloss_u1 + 60*simloss_e1
            # loss = bprLoss + regLoss + clLoss+ 0.00001*klloss
            loss = bprLoss  + regLoss + clLoss + 0.00001*klloss
            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            
            # loss.backward(retain_graph=True)
            loss.backward()
            if loss != loss:
                raise Exception('NaN in loss, crack!')
                pritnln()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            # print("back done")
            self.opt.step()
            log('Step %d/%d: loss = %.3f, regLoss = %.3f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg = [0] * 2
        epRecall2, epNdcg2 = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            adjlist = {}
            adjlist[0] =  self.handler.torchBiAdj
            usrEmbeds, itmEmbeds = self.model(adjlist, 1.0)

            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            if i  == steps: 
                tmp = nn.Linear(256, 32).cuda()(allPreds.T.float()).float().detach().cpu().numpy()
                print("prediction results dimension:", tmp.shape)
                import pickle
                file=open(r"../Models/cluster_ours.pickle","wb")
                pickle.dump(tmp,file) #storing_list
                file.close()
            # println()
            _, topLocs = t.topk(allPreds, args.topk)
            _, topLocs2 = t.topk(allPreds, args.topk2)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            recall2, ndcg2 = self.calcRes(topLocs2.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            epRecall2 += recall2
            epNdcg2 += ndcg2
            # log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f, recall40 = %.2f, ndcg40 = %.2f          ' % (i, steps, recall, ndcg, recall2, ndcg2), save=False, oneline=True)

        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        ret['Recall2'] = epRecall2 / num
        ret['NDCG2'] = epNdcg2 / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig =0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('../History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, '../Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = t.load('../Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('../History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded') 
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True
    
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()