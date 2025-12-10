# Marek Hric xhricma00
# Martin Vaculík xvaculm00

import numpy as np
from primitives import Tensor
import pdb

"""
We are using the Vacko/Marekk initialization for the kernels.
Authors can be found on discord: @vacko_ and @marek3k
Props to noyada to keep us sane.
"""


class Conv1DLayer:
    def __init__(self, out_channels: int, kernel_size: int):
        self.c_out = out_channels
        self.k_s = kernel_size
        self.kernels = [Tensor(np.clip(np.random.randn(self.k_s), -1.5, 1.5)) for _ in range(self.c_out)] # we call this Vacko/Marekk intialization, authors can be found on discord 
        # @vacko_ and @marek3k

    def forward(self, x: Tensor) -> Tensor:
        # in:  (b_s, l_in)
        # out: (b_s, c_out, l_out)

        b_s, l_in = x.value.shape

        l_out = l_in - self.k_s + 1

        # im2col: (b_s, k_s, l_out)
        conv_mat = np.zeros(shape=(b_s, self.k_s, l_out))
        for b in range(b_s):
            for r in range(l_out):
                conv_mat[b, :, r] = x.value[b, r : r + self.k_s]

        # stacked kernels: (c_out, k_s)
        ker_mat = np.stack([k.value for k in self.kernels], axis=0)

        def back_op(grad: np.ndarray):
            for b, batch_grad in enumerate(grad):
                for c, filter_grad in enumerate(batch_grad):
                    x.grad[b] += np.convolve(filter_grad, self.kernels[c].value, mode='full') # full convolution based on clanek 
                    self.kernels[c].grad += np.correlate(x.value[b], filter_grad, mode='valid')  # we do not want to flip the kernel, math from clanek 
            return

            

        return Tensor(ker_mat[np.newaxis, :, :] @ conv_mat, back_op=back_op)

    def parameters(self) -> list[Tensor]:
        return self.kernels


class MaxPool1DLayer:
    def __init__(self, pool_size, stride):
        self.p_s = pool_size
        self.s = stride

    def forward(self, x):
        b_s, c, l_in = x.shape
        l_out = (l_in - self.p_s) // self.s + 1

        pooled = np.zeros((b_s, c, l_out))
        pooled_idxs = np.zeros_like(pooled, dtype=int)

        for b in range(b_s):
            for ch in range(c):
                for i in range(l_out):
                    start = i * self.s
                    end = min(start + self.p_s, l_in)
                    pooled[b, ch, i] = np.max(x.value[b, ch, start:end], axis=-1)
                    pooled_idxs[b, ch, i] = (
                        np.argmax(x.value[b, ch, start:end], axis=-1) + start
                    )

        def back_op(grad: np.ndarray):
            for b in range(b_s):
                for ch in range(c):
                    for i in range(l_out):
                        max_idx = pooled_idxs[b, ch, i]
                        x.grad[b, ch, max_idx] += grad[b, ch, i]

        return Tensor(pooled, back_op=back_op)

    def parameters(self):
        pass
