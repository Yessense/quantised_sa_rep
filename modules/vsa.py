import torch


def generate(dim=1000):
    v = torch.randn(dim)
    return v / torch.linalg.norm(v)


def make_unitary(v):
    fft_val = torch.fft.fft(v)
    fft_imag = fft_val.imag
    fft_real = fft_val.real
    fft_norms = torch.sqrt(fft_imag ** 2 + fft_real ** 2)
    invalid = fft_norms <= 0.0
    fft_val[invalid] = 1.0
    fft_norms[invalid] = 1.0
    fft_unit = fft_val / fft_norms
    return (torch.fft.ifft(fft_unit, n=len(v))).real


def get_vsa_grid(epsilon=0.07, dim=1000, n=500, start=1):
    x_0 = make_unitary(generate(dim))
    end = (n - 1) * epsilon + start
    xs = torch.linspace(start, end, n)
    grid = torch.fft.ifft(torch.fft.fft(x_0).view(1, -1)**xs.view(-1, 1), dim=1)
    return grid.real
 