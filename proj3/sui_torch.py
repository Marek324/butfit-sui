import numpy as np
from primitives import Tensor


class Conv1DLayer:
    def __init__(self, out_channels, kernel_size):
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.kernels = [
            Tensor(np.random.randn(kernel_size)) for _ in range(out_channels)
        ]

    def forward(self, x: Tensor):
        x_val = x.value
        batch, length = x_val.shape
        k = self.kernel_size

        out_length = length - k + 1

        x_cols = np.zeros((batch, out_length, k))
        for b in range(batch):
            for i in range(out_length):
                x_cols[b, i] = x_val[b, i : i + k]

        W = np.stack([kernel.value for kernel in self.kernels], axis=0)

        out_val = np.zeros((batch, self.out_channels, out_length))

        for b in range(batch):
            out_val[b] = (x_cols[b] @ W.T).T

        saved_x_cols = x_cols

        def back_op(grad_out):
            grad_x = np.zeros_like(x_val)

            grad_W = np.zeros_like(W)

            for b in range(batch):
                go = grad_out[b].T

                grad_W += go.T @ saved_x_cols[b]

                dx_cols = go @ W
                for i in range(out_length):
                    grad_x[b, i : i + k] += dx_cols[i]

            for oc in range(self.out_channels):
                self.kernels[oc].grad += grad_W[oc]

            x.grad += grad_x

        return Tensor(out_val, back_op)

    def parameters(self):
        return self.kernels


class MaxPool1DLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x: Tensor):
        x_val = x.value

        if x_val.ndim == 2:
            batch, length = x_val.shape
            channels = 1
            x_reshaped = x_val[:, None, :]
        else:
            batch, channels, length = x_val.shape
            x_reshaped = x_val

        pool_size = self.pool_size
        stride = self.stride

        out_length = 1 + (length - pool_size) // stride
        out_val = np.zeros((batch, channels, out_length))
        argmax_positions = np.zeros((batch, channels, out_length), dtype=int)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_length):
                    start = i * stride
                    end = start + pool_size

                    window = x_reshaped[b, c, start:end]
                    max_idx = np.argmax(window)

                    out_val[b, c, i] = window[max_idx]
                    argmax_positions[b, c, i] = start + max_idx

        def back_op(grad_out):
            grad_input = np.zeros_like(x_reshaped)

            for b in range(batch):
                for c in range(channels):
                    for i in range(out_length):
                        idx = argmax_positions[b, c, i]
                        grad_input[b, c, idx] += grad_out[b, c, i]

            if x_val.ndim == 2:
                x.grad += grad_input[:, 0, :]
            else:
                x.grad += grad_input

        return Tensor(out_val, back_op)

    def parameters(self):
        return
