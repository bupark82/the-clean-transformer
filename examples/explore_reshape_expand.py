from cleanformer.tensors import subsequent_mask

def main():
    N = 3
    L = 4
    heads = 2
    mask = subsequent_mask(L)
    print(mask)
    print(mask.shape)
    # (L, L) -> (1, 1, L, L) -> (N, heads, L, L)
    mask = mask.reshape(1, 1, L, L)
    print(mask)
    print(mask.shape)
    mask = mask.expand(N, heads, -1, -1)
    print(mask)
    print(mask[0][0], mask[0][0].shape)
    print(mask[1][0], mask[1][0].shape)
    print(mask[2][0], mask[2][0].shape)
    print(mask[0][1], mask[0][1].shape)
    print(mask[1][1], mask[1][1].shape)
    print(mask[2][1], mask[2][1].shape)


if __name__ == '__main__':
    main()

    