import torch


def main():
    x = torch.Tensor([4, 5, 1, 2])
    print(torch.argmax(x))

if __name__ == '__main__':
    main()