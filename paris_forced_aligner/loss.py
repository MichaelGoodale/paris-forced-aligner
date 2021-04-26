#Adapted from https://github.com/liuhu-bigeye/enctc.crnn
import torch

def m_eye(n, k=0, device='cpu'):
    assert k < n and k >= 0
    if k == 0:
        return torch.eye(n, dtype=torch.float, device=device)
    else:
        return torch.cat((torch.cat((torch.zeros(n-k, k), torch.eye(n-k)), dim=1), torch.zeros(k, n)), dim=0).type(torch.float).to(device)


def log_batch_dot(alpha_t, rec):
    '''
    alpha_t: (batch, 2U+1)
    rec: (batch, 2U+1, 2U+1)
    '''
    eps_nan = -1e8
    # a+b
    _sum = alpha_t[:, :, None] + rec
    _max_sum = torch.max(_sum, dim=1)[0]
    nz_mask1 = torch.gt(_max_sum, eps_nan) # max > eps_nan
    nz_mask2 = torch.gt(_sum, eps_nan)     # item > eps_nan

    # a+b-max
    _sum = _sum - _max_sum[:, None]

    # exp
    _exp = torch.zeros_like(_sum, dtype=torch.float)
    _exp[nz_mask2] = torch.exp(_sum[nz_mask2])

    # sum exp
    _sum_exp = torch.sum(_exp, dim=1)

    out = torch.ones_like(_max_sum, dtype=torch.float) * eps_nan
    out[nz_mask1] = torch.log(_sum_exp[nz_mask1]) + _max_sum[nz_mask1]
    return out

def log_sum_exp_axis(a, uniform_mask=None, dim=0, device='cpu'):
    assert dim == 0
    eps_nan = -1e8
    eps = 1e-26
    _max = torch.max(a, dim=dim)[0]

    if not uniform_mask is None:
        nz_mask2 = torch.gt(a, eps_nan) * uniform_mask
        nz_mask1 = torch.gt(_max, eps_nan) * torch.ge(torch.max(uniform_mask, dim=dim)[0], 1)
    else:
        nz_mask2 = torch.gt(a, eps_nan)
        nz_mask1 = torch.gt(_max, eps_nan)

    # a-max
    a = a - _max[None]

    # exp
    _exp_a = torch.zeros_like(a, dtype=torch.float)
    _exp_a[nz_mask2] = torch.exp(a[nz_mask2])

    # sum exp
    _sum_exp_a = torch.sum(_exp_a, dim=dim)

    out = torch.ones_like(_max, dtype=torch.float, device=device) * eps_nan
    out[nz_mask1] = torch.log(_sum_exp_a[nz_mask1] + eps) + _max[nz_mask1]
    return out

def log_sum_exp(*arrs, device='cpu'):
    c = torch.cat(list(map(lambda x:x[None], arrs)), dim=0)
    return log_sum_exp_axis(c, dim=0, device=device)

def ctc_ent_loss_log(pred, token, pred_len, token_len, blank=0, device='cpu', h_rate=0.1):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    :out alpha: (Time, batch, 2U+1) ∑p(π|x)
    :out beta: (Time, batch, 2U+1)  ∑p(π|x)logp(π|x)
    :out H: -beta/alpha+log(alpha)
    '''
    Time, batch = pred.shape[0], pred.shape[1]
    token = token.type(torch.long)
    U = token.shape[1]
    eps_nan = -1e8
    eps = 1e-8

    # token_with_blank
    token_with_blank = torch.cat((torch.zeros(batch, U, 1, dtype=torch.long, device=device), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = torch.cat((token_with_blank, torch.zeros(batch, 1, dtype=torch.long, device=device)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.shape[1]

    pred = pred[torch.arange(0, Time, dtype=torch.long, device=device)[:, None, None], 
                torch.arange(0, batch, dtype=torch.long, device=device)[None, :, None],
                token_with_blank[None, :]]  # (torch, batch, 2U+1)

    # recurrence relation
    sec_diag = torch.cat((torch.zeros((batch, 2), dtype=torch.float, device=device),
                          torch.ne(token_with_blank[:, :-2],
                                   token_with_blank[:, 2:]).type(torch.float)
                          ), dim=1) *\
              torch.ne(token_with_blank, blank).type(torch.float)	# (batch, 2U+1)
    recurrence_relation = (m_eye(length, device=device) + m_eye(length, k=1, device=device)).repeat(batch, 1, 1) + m_eye(length, k=2, device=device).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)
    recurrence_relation = eps_nan * (torch.ones_like(recurrence_relation) - recurrence_relation)

    # alpha
    alpha_t = torch.cat((pred[0, :, :2], torch.ones(batch, 2*U-1, dtype=torch.float, device=device)*eps_nan), dim=1) # (batch, 2U+1)
    beta_t = torch.cat((pred[0, :, :2] + torch.log(-pred[0, :, :2]+eps),
                    torch.ones(batch, 2*U-1, dtype=torch.float, device=device)*eps_nan), dim=1) # (batch, 2U+1)

    alphas = alpha_t[None] # (1, batch, 2U+1)
    betas = beta_t[None] # (1, batch, 2U+1)

    # dynamic programming
    # (T, batch, 2U+1)
    for t in torch.arange(1, Time, dtype=torch.long):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation).to(device) + pred[t]
        beta_t = log_sum_exp(log_batch_dot(beta_t, recurrence_relation).to(device) + pred[t], torch.log(-pred[t]+eps) + alpha_t, device=device)

        alphas = torch.cat((alphas, alpha_t[None]), dim=0)
        betas = torch.cat((betas, beta_t[None]), dim=0)

    def collect_label(probability):
        labels_2 = probability[pred_len-1, torch.arange(batch, dtype=torch.long, device=device), 2*token_len-1]
        labels_1 = probability[pred_len-1, torch.arange(batch, dtype=torch.long, device=device), 2*token_len]
        labels_prob = log_sum_exp(labels_2, labels_1, device=device)
        return labels_prob

    alpha = collect_label(alphas)
    beta = collect_label(betas)

    H = torch.exp(beta-alpha) + alpha
    costs = -alpha

    costs = costs.sum()
    H = H.sum()
    inf = float("inf")
    if costs == inf or costs == -inf or torch.isnan(costs) or torch.isnan(H):
        print("Warning: received an inf loss, setting loss value to 0")
        return torch.zeros(H.size()), torch.zeros(costs.size())

    loss = -h_rate*H + (1-h_rate)*costs
    return loss
