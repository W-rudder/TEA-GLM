from torch import nn
import torch
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, tau, self_tp=False) -> None:
        super().__init__()
        self.tau = tau
        self.self_tp = self_tp
        self.f = lambda x: torch.exp(x / self.tau)
    
    def sim(self, proj_z1, proj_z2):
        z1 = F.normalize(proj_z1)
        z2 = F.normalize(proj_z2)
        return torch.mm(z1, z2.t())
    
    def instance_loss(self, proj_z1, proj_z2):

        refl_sim = self.f(self.sim(proj_z1, proj_z1))
        between_sim = self.f(self.sim(proj_z1, proj_z2))

        loss = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        return loss

    def text_contrast(self, z1, z2, principal_component=None):
        if self.self_tp:
            z1, z2 = z1.unsqueeze(1), z2.unsqueeze(1)
            matrix_1 = torch.bmm(z1, principal_component.permute(0, 2, 1)).squeeze().t()
            matrix_2 = torch.bmm(z2, principal_component.permute(0, 2, 1)).squeeze().t()
        else:
            matrix_1 = torch.mm(z1, principal_component.t()).t()
            matrix_2 = torch.mm(z2, principal_component.t()).t()
    
        refl_sim = self.f(self.sim(matrix_1, matrix_1))
        between_sim = self.f(self.sim(matrix_1, matrix_2))

        loss = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - between_sim.diag()))

        return loss

    def get_loss(self, z1, z2, proj_z1, proj_z2, principal_component):
        instance_loss = self.instance_loss(proj_z1, proj_z2).mean()
        text_contrast = self.text_contrast(z1, z2, principal_component).mean()

        return (instance_loss + text_contrast), instance_loss.data.item(), text_contrast.data.item()

    def forward(self, z1, z2, proj_z1, proj_z2, principal_component):
        l1, ins_loss_1, contrast_loss_1 = self.get_loss(z1, z2, proj_z1, proj_z2, principal_component)
        l2, ins_loss_2, contrast_loss_2 = self.get_loss(z2, z1, proj_z2, proj_z1, principal_component)

        ret = (l1 + l2) * 0.5
        ins_loss = (ins_loss_1 + ins_loss_2) * 0.5
        contrast_loss = (contrast_loss_1 + contrast_loss_2) * 0.5

        return ret, ins_loss, contrast_loss


class GraceLoss(nn.Module):
    def __init__(self, tau) -> None:
        super().__init__()
        self.tau = tau
    
    def sim(self, proj_z1, proj_z2):
        z1 = F.normalize(proj_z1)
        z2 = F.normalize(proj_z2)
        return torch.mm(z1, z2.t())

    def get_loss(self, proj_z1, proj_z2):
        f = lambda x: torch.exp(x / self.tau)

        refl_sim = f(self.sim(proj_z1, proj_z1))
        between_sim = f(self.sim(proj_z1, proj_z2))

        loss = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        return loss

    def get_real_loss(self, proj_z1, proj_z2, batch):
        f = lambda x: torch.exp(x / self.tau)
        pos_z1 = proj_z1[batch.node_label_index[:batch.input_id.shape[0]]]
        pos_z2 = proj_z2[batch.node_label_index[:batch.input_id.shape[0]]]

        neg_z1 = proj_z1[batch.node_label_index[batch.input_id.shape[0]:]]
        neg_z2 = proj_z2[batch.node_label_index[batch.input_id.shape[0]:]]

        pos_sim = f(self.sim(pos_z1, pos_z2)).diag()
        intra_sim = f(self.sim(pos_z1, neg_z1)).sum(1)
        inter_sim = f(self.sim(pos_z1, neg_z2)).sum(1)

        loss = -torch.log(
            pos_sim
            / (pos_sim + intra_sim + inter_sim))

        return loss

    def get_semi_loss(self, proj_z1, proj_z2, neg, neg_share=True):
        f = lambda x: torch.exp(x / self.tau)

        pos_sim = f(self.sim(proj_z1, proj_z2)).diag()
        if neg_share:
            neg_sim = f(self.sim(proj_z1, neg)).sum(1)
        else:
            neg_sim = f(self.sim(proj_z1, neg)).diag()

        loss = -torch.log(
            pos_sim
            / (pos_sim + neg_sim))

        return loss

    def forward(self, proj_z1, proj_z2, neg=None, batch=None):
        if neg is not None:
            l1 = self.get_semi_loss(proj_z1, proj_z2, neg)
            l2 = self.get_semi_loss(proj_z2, proj_z1, neg)
        elif batch is not None:
            l1 = self.get_real_loss(proj_z1, proj_z2, batch)
            l2 = self.get_real_loss(proj_z2, proj_z1, batch)          
        else:
            l1 = self.get_loss(proj_z1, proj_z2)
            l2 = self.get_loss(proj_z2, proj_z1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        
        return ret