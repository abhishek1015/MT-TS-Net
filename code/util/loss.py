
import torch
from torch import Tensor
import pdb

#https://github.com/DataX-JieHao/Cox-PASNet/blob/master/coxpasnet/Survival_CostFunc_CIndex.py
def R_set(x):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
        x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
        indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return(indicator_matrix)


def neg_par_log_likelihood(pred, ytime, yevent):
    '''Calculate the average Cox negative partial log-likelihood.
    Note that this function requires the input data have been sorted in descending order.
    Input:
        pred: linear predictors from trained model.
        ytime: true survival time from load_data().
        yevent: true censoring status from load_data().
    Output:
        cost: the cost that is to be minimized.
    '''
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    ###if gpu is being used
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    ###
    risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))

    return(cost)

# based on pycox package
# https://github.com/havakv/pycox/blob/master/pycox/models/loss.py
# tutorial: https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning/
def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    #log_cumsum_h = log_cumsum_h
    loss = - log_h.sub(log_cumsum_h).mul(events).sum()
    if events.sum().item()==0:
        return loss
    return loss.div(events.sum())

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    #pdb.set_trace()
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

class CoxPHLossSorted(torch.nn.Module):
    """Loss for CoxPH.
    Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def __init__(self):
        super().__init__()

    def forward(self, log_h: Tensor, events: Tensor) -> Tensor:
        return cox_ph_loss_sorted(log_h, events)


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return cox_ph_loss(log_h, durations, events)

if __name__=='__main__':
    criterion=CoxPHLoss()
    log_h = torch.rand(10)
    durations = torch.rand(10)
    events = torch.randint(2, (10,))
    pdb.set_trace()
    loss = criterion(log_h, durations, events)
