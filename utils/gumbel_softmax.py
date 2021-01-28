import torch
import torch.nn.functional as F

def sample_gumbel(bs, n_classes, eps=1e-20):
    unif = torch.distributions.Uniform(0,1).sample((bs, n_classes))
    g = -torch.log(-torch.log(unif + eps) + eps)
    return g

def gumbel_softmax(logits, hard=False, tau=1, dim=-1):
    '''
    logits: bs*n_classes
    '''
    gumbel = sample_gumbel(*logits.shape)
    y = logits + gumbel.to(logits.device)
    y = F.softmax(y / tau, dim=dim)
    if hard:
        _, ind = y.max(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(y).view(-1, y.shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*y.shape)
        y_hard = (y_hard - y).detach() + y
        ret = y_hard
    else:
        ret = y
    return ret

if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        for hard in [True, False]:
            x = torch.rand(1,3).to(device)
            x.requires_grad = True
            y = gumbel_softmax(x, hard)
            z = x*y