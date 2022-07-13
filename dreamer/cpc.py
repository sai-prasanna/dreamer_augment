import torch
from torch import distributions as td


def compute_cpc_loss(pred, features, cpc_amount=(10, 30), cpc_contrast='window'):

    batch_size, batch_len = features.size(0), features.size(1)
    if cpc_contrast == 'batch':
        ta = []
        for i in range(features.size(0)):
            ta.append(pred.log_prob(torch.roll(features, i, 0)))

        positive = pred.log_prob(features)
        negative = torch.logsumexp(torch.stack(ta), 0)
        return positive - negative

    if cpc_contrast == 'time':
        ta = []
        for i in range(features.size(0)):
            ta.append(pred.log_prob(torch.roll(features, i, 1)))

        positive = pred.log_prob(features)
        negative = torch.logsumexp(torch.stack(ta), 1)
        return positive - negative

    elif cpc_contrast == 'window':
        batch_amount, time_amount = cpc_amount[0], cpc_amount[1]
        assert batch_amount <= batch_size
        assert time_amount <= batch_len
        total_amount = batch_amount * time_amount
        ta = []

        def compute_negatives(index):
            batch_shift = index // time_amount
            time_shift = index % time_amount
            batch_shift -= batch_amount // 2
            time_shift -= time_amount // 2
            rolled = torch.roll(torch.roll(features, batch_shift, 0), time_shift, 1)
            return pred.log_prob(rolled)

        for i in range(total_amount):
            ta.append(compute_negatives(i))
        positive = pred.log_prob(features)
        negative = torch.logsumexp(torch.stack(ta), 0)
        return positive - negative


if __name__ == "__main__":
    feats = torch.rand(10, 20, 3)
    dist = td.Independent(td.Normal(torch.rand(10, 20, 3), 1), 1)
    t = compute_cpc_loss(dist, feats, "time", None)
    b = compute_cpc_loss(dist, feats, "batch", None)
    w = compute_cpc_loss(dist, feats, "window", (6, 11))
    print(t, t.mean(), t.size())
    print(b, b.mean(), b.size())
    print(w, w.mean(), w.size())
