import torch
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import random
import os
import os.path as osp
import pandas as pd
import torch.nn.functional as F
import torch_geometric

from model import GraphSAGE
from dataloader import NodeNegativeLoader
from loss.contrastive_loss import ContrastiveLoss, GraceLoss


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_title(path, paperid2node):
    text_feature={}
    with open(path,'r',encoding='utf-8') as f:
        next(f)  # pass the first line
        print('pass the first line')
        n=0
        for line in f:
            n+=1
            if n==179719:
                continue
            line = line.strip('\n').split('\t')
            p_id = int(line[0])
            if p_id in paperid2node.keys():
                text_feature[paperid2node[p_id]]=[line[1]]  # title
                text_feature[paperid2node[p_id]].append(line[2])  # Full abstract
    print('done')
    return text_feature


def train(args, data, model, optimizer, criterion):
    train_loader = NodeNegativeLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        neg_ratio=args.num_negs,
        num_neighbors=fans_out,
        mask_feat_ratio_1=args.drop_feature_rate_1,
        mask_feat_ratio_2=args.drop_feature_rate_2,
        drop_edge_ratio_1=args.drop_edge_rate_1,
        drop_edge_ratio_2=args.drop_edge_rate_2,
    )
    model.train()
    total_loss = 0
    total_ins_loss = 0
    total_con_loss = 0

    pbar = tqdm(total=len(train_loader))
    for step, (ori_graph, view_1, view_2) in enumerate(train_loader):
        ori_graph, view_1, view_2 = ori_graph.to(device), view_1.to(device), view_2.to(device)

        optimizer.zero_grad()

        z1 = model(view_1.x, view_1.edge_index)[view_1.node_label_index]
        z2 = model(view_2.x, view_2.edge_index)[view_2.node_label_index]

        proj_z1 = model.projection(z1)
        proj_z2 = model.projection(z2)

        # proj_z1 = z1
        # proj_z2 = z2

        if args.self_tp:
            principal_component = all_principal_component[ori_graph.raw_nodes]
        else:
            principal_component = all_principal_component

        if args.use_tp:
            # loss, ins_loss, align_loss, contrast_loss = criterion(z1, z2, proj_z1, proj_z2, principal_component)
            loss, ins_loss, contrast_loss = criterion(z1, z2, proj_z1, proj_z2, principal_component)
            total_ins_loss += ins_loss * proj_z1.shape[0]
            total_con_loss += contrast_loss * proj_z1.shape[0]
            total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]
        else:
            loss = criterion(proj_z1, proj_z2)
            total_loss += loss.data.item() * proj_z1.shape[0]

        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            if args.use_tp:
                print('Step {:05d} | Loss {:.4f} | Instance Loss {:.4f} | Contrastive Loss {:.4f}'.format(step, loss.item(), ins_loss, contrast_loss))
            else:  
                print('Step {:05d} | Loss {:.4f}'.format(step, loss.item()))
        pbar.update()
    pbar.close()
    total_mean_loss = total_loss / train_id.shape[0]
    total_mean_instance_loss = total_ins_loss / train_id.shape[0]
    total_mean_contrastive_loss = total_con_loss / train_id.shape[0]

    return total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss

@torch.no_grad()
def test(args, test_data, num_nodes):

    test_loader = NodeNegativeLoader(
        test_data,
        batch_size=512,
        shuffle=False,
        neg_ratio=0,
        num_neighbors=[-1],
        mask_feat_ratio_1=args.drop_feature_rate_1,
        mask_feat_ratio_2=args.drop_feature_rate_2,
        drop_edge_ratio_1=args.drop_edge_rate_1,
        drop_edge_ratio_2=args.drop_edge_rate_2,
    )

    model.eval()
    total_loss = 0
    total_ins_loss = 0
    total_con_loss = 0

    pbar = tqdm(total=len(test_loader))
    for step, (ori_graph, view_1, view_2) in enumerate(test_loader):
        ori_graph, view_1, view_2 = ori_graph.to(device), view_1.to(device), view_2.to(device)

        z1 = model(view_1.x, view_1.edge_index)[view_1.node_label_index]
        z2 = model(view_2.x, view_2.edge_index)[view_2.node_label_index]

        proj_z1 = model.projection(z1)
        proj_z2 = model.projection(z2)

        if args.self_tp:
            principal_component = all_principal_component[ori_graph.raw_nodes]
        else:
            principal_component = all_principal_component

        if args.use_tp:
            loss, ins_loss, contrast_loss = criterion(z1, z2, proj_z1, proj_z2, principal_component)
            total_ins_loss += ins_loss * proj_z1.shape[0]
            total_con_loss += contrast_loss * proj_z1.shape[0]
            total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]
        else:
            loss = criterion(proj_z1, proj_z2)
            total_loss += loss.data.item() * proj_z1.shape[0]
        pbar.update()
    pbar.close()

    total_mean_loss = total_loss / num_nodes
    total_mean_instance_loss = total_ins_loss / num_nodes
    total_mean_contrastive_loss = total_con_loss / num_nodes

    print(f"Mean Test Loss: {total_mean_loss}\nMean Test Instance Loss: {total_mean_instance_loss}\nMean Test Contrastive Loss: {total_mean_contrastive_loss}")

    return total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num_epochs', type=int, default=70)
    argparser.add_argument('--num_runs', type=int, default=2)
    argparser.add_argument('--num_hidden', type=int, default=2048)
    argparser.add_argument('--num_out', type=int, default=4096)
    argparser.add_argument('--num_layers', type=int, default=2)
    argparser.add_argument('--num_negs', type=int, default=0)
    argparser.add_argument('--patience', type=int, default=10)
    argparser.add_argument('--fan_out', type=str, default='25,10')
    argparser.add_argument('--batch_size', type=int, default=512)
    argparser.add_argument('--log_every', type=int, default=20)
    argparser.add_argument('--eval_every', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.002)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num_workers', type=int, default=0, help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--lazy_load', default=True)
    argparser.add_argument('--use_tp', default=True)
    argparser.add_argument('--self_tp', default=False)
    argparser.add_argument('--drop_edge_rate_1', type=int, default=0.3)
    argparser.add_argument('--drop_edge_rate_2', type=int, default=0.4)
    argparser.add_argument('--drop_feature_rate_1', type=int, default=0.0)
    argparser.add_argument('--drop_feature_rate_2', type=int, default=0.1)
    argparser.add_argument('--tau', type=int, default=0.4)
    argparser.add_argument('--gnn_type', type=str, default='sage')
    
    args = argparser.parse_args()

    fans_out = [int(i) for i in args.fan_out.split(',')]
    assert len(fans_out) == args.num_layers

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    if not os.path.exists(f'./saved_model'):
        os.makedirs(f'./saved_model')

    print("Node Unsupervised")
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'graph_data_paper.pt')
    datasets = torch.load(path)
    data_list = [datasets['arxiv']]
    n_ls = []
    for data in data_list:
        n_d = torch_geometric.transforms.ToUndirected()(data)
        n_d = torch_geometric.transforms.AddRemainingSelfLoops()(n_d)
        n_ls.append(n_d)
    data = n_ls[0]
    data.x = data.x.type(torch.bfloat16)
    train_id = data.x

    for run in range(args.num_runs):
        print(f"Run {run}")
        seed_everything(run)

        data = data.to(device, 'x', 'edge_index')

        num_node_features = data.x.shape[1]
        print(num_node_features)
        model = GraphSAGE(
            num_node_features,
            hidden_channels=args.num_hidden,
            out_channels=args.num_out,
            n_layers=args.num_layers,
            num_proj_hidden=args.num_out,
            activation=F.relu,
            dropout=args.dropout,
            edge_dim=None,
            gnn_type=args.gnn_type
        ).to(device)
        model = model.to(dtype=torch.bfloat16)
        print(model)

        all_principal_component = torch.load('./PCA_1000_pc_llama.pt').to(device, dtype=torch.bfloat16)

        if args.use_tp:
            criterion = ContrastiveLoss(args.tau, self_tp=args.self_tp).to(device)
        else:
            criterion = GraceLoss(args.tau).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        no_increase = 0
        best_loss = 1000000000
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch}")
            total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss = train(args, data, model, optimizer, criterion)
            if total_mean_loss < best_loss:
                best_loss = total_mean_loss
                no_increase = 0
                torch.save(model.state_dict(), f'./saved_model/GraphSAGE_arxiv_1000_tp_run_{run+1}.pth')
            else:
                no_increase += 1
                if no_increase > args.patience:
                    break
        