

import torch
def main():
    L = 10
    out = torch.ones(size=(L, L))
    print(out)
    out = torch.tril(out, diagonal=0)
    print(out)
    """
    1 0 0 
    1 1 0
    1 1 1 
    """


if __name__ == '__main__':
    main()

