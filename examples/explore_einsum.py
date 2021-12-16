import torch


def main():
    N = 10
    L = 30
    H = 512
    q = torch.rand(size=(N, L, H))
    k = torch.rand(size=(N, L, H))
    v = torch.rand(size=(N, L, H))

    # query의 길이 = q
    # key의 길이 = k
    sims = torch.einsum("nqh,nkh->nqk", q, k)  # (N, L, H) *  (N, L, H) -> (N, L, L)
    print(sims)
    attentions = torch.softmax(sims, dim=2)  # (N, q의 길이 L, k의 갈이 L <- 마지막 차원을 정규화)
    # "j"차원에 대하여 벡터의 내적이 계산, 그렇게 j 차원은 reduce.
    print(torch.sum(attentions, dim=2))
    print(torch.sum(attentions, dim=1))

    contexts = torch.einsum("nqk,nkh->nqh", attentions, v)  # (N, L, L) * (N, L, H) -> (N, L, H)
    #
    # #
    # attentions = torch.softmax(sims, dim=1)  # (N, q의 길이 L, k의 갈이 L <- 마지막 차원을 정규화)


if __name__ == '__main__':
    main()
