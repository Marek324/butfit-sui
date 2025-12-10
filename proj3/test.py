import torch
import unittest
from numpy.testing import assert_allclose
import numpy as np
import sui_torch as st
import primitives as p


class TestConv1DLayer(unittest.TestCase):
    def test_forward_output_shape(self):
        batch_size = 2
        signal_length = 10
        out_channels = 4
        kernel_size = 3

        conv = st.Conv1DLayer(out_channels=out_channels, kernel_size=kernel_size)
        input_tensor = p.Tensor(np.random.randn(batch_size, signal_length))

        output = conv.forward(input_tensor)

        # Expected output: (batch_size, out_channels, signal_length - kernel_size + 1)
        expected_shape = (batch_size, out_channels, signal_length - kernel_size + 1)
        self.assertEqual(output.value.shape, expected_shape)

    def test_forward_simple_convolution(self):
        conv = st.Conv1DLayer(out_channels=1, kernel_size=2)

        conv.kernels[0].value = np.array([1.0, 1.0])

        input_tensor = p.Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
        output = conv.forward(input_tensor)

        expected = np.array([[[3.0, 5.0, 7.0]]])
        assert_allclose(output.value, expected, rtol=1e-5)

    def test_backward_gradient_flow(self):
        conv = st.Conv1DLayer(out_channels=2, kernel_size=3)
        input_tensor = p.Tensor(np.random.randn(1, 5))

        output = conv.forward(input_tensor)
        grad = np.ones_like(output.value)
        output.backward(grad)

        self.assertTrue(np.any(input_tensor.grad != 0))
        self.assertTrue(np.any(conv.kernels[0].grad != 0))

    def test_forward_against_pytorch(self):
        # 1. Setup parameters
        batch_size = 4
        signal_length = 20
        out_channels = 5
        kernel_size = 3

        # 2. Create input data and custom layer
        input_data = np.random.randn(batch_size, signal_length)
        custom_conv = st.Conv1DLayer(out_channels=out_channels, kernel_size=kernel_size)

        # 3. Synchronize kernels between custom layer and PyTorch
        # Create a single numpy array for all kernels
        kernels_np = np.random.randn(out_channels, kernel_size)

        # Assign kernels to the custom layer
        for c in range(out_channels):
            custom_conv.kernels[c].value = kernels_np[c]

        # 4. Perform forward pass with the custom layer
        custom_input = p.Tensor(input_data)
        custom_output = custom_conv.forward(custom_input)

        # 5. Perform forward pass with PyTorch
        # PyTorch expects input as (N, C_in, L) and kernel as (C_out, C_in, K)
        # Our implementation assumes C_in=1
        input_torch = torch.from_numpy(input_data).unsqueeze(1).float()
        kernels_torch = torch.from_numpy(kernels_np).unsqueeze(1).float()

        # Use PyTorch's functional conv1d
        pytorch_output = torch.nn.functional.conv1d(input_torch, kernels_torch)

        # 6. Compare the results
        assert_allclose(
            custom_output.value, pytorch_output.detach().numpy(), rtol=1e-5, atol=1e-5
        )

    def test_backward_against_pytorch(self):
        # 1. Setup parameters
        batch_size = 4
        signal_length = 20
        out_channels = 5
        kernel_size = 3

        # 2. Create common data
        input_data = np.random.randn(batch_size, signal_length)
        kernels_np = np.random.randn(out_channels, kernel_size)
        # Create a random gradient for the output to start backpropagation
        output_grad_np = np.random.randn(
            batch_size, out_channels, signal_length - kernel_size + 1
        )

        # 3. Perform backward pass with the custom layer
        custom_conv = st.Conv1DLayer(out_channels=out_channels, kernel_size=kernel_size)
        for c in range(out_channels):
            custom_conv.kernels[c].value = kernels_np[c]

        custom_input = p.Tensor(input_data)
        custom_output = custom_conv.forward(custom_input)
        custom_output.backward(output_grad_np)

        # 4. Perform backward pass with PyTorch
        # PyTorch expects input as (N, C_in, L) and kernel as (C_out, C_in, K)
        input_torch = torch.from_numpy(input_data).unsqueeze(1).float()
        input_torch.requires_grad = True
        kernels_torch = torch.from_numpy(kernels_np).unsqueeze(1).float()
        kernels_torch.requires_grad = True

        # Use PyTorch's functional conv1d
        pytorch_output = torch.nn.functional.conv1d(input_torch, kernels_torch)
        output_grad_torch = torch.from_numpy(output_grad_np).float()
        pytorch_output.backward(gradient=output_grad_torch)

        # 5. Compare the gradients
        # Compare input gradients
        # Squeeze PyTorch's grad to match shape (N, L) from (N, 1, L)
        assert_allclose(
            custom_input.grad, input_torch.grad.squeeze(1).numpy(), rtol=1e-5, atol=1e-5
        )

        # Compare kernel gradients
        # Squeeze PyTorch's grad to match shape (C_out, K) from (C_out, 1, K)
        pytorch_kernel_grads = kernels_torch.grad.squeeze(1).numpy()
        for c in range(out_channels):
            assert_allclose(
                custom_conv.kernels[c].grad,
                pytorch_kernel_grads[c],
                rtol=1e-5,
                atol=1e-5,
            )


class TestMaxPool1DLayer(unittest.TestCase):
    def test_forward_output_shape(self):
        batch_size = 2
        channels = 3
        length = 10
        pool_size = 2
        stride = 2

        pool = st.MaxPool1DLayer(pool_size=pool_size, stride=stride)
        input_tensor = p.Tensor(np.random.randn(batch_size, channels, length))

        output = pool.forward(input_tensor)

        # Expected output: (batch_size, channels, (length - pool_size) // stride + 1)
        expected_length = (length - pool_size) // stride + 1
        expected_shape = (batch_size, channels, expected_length)
        self.assertEqual(output.value.shape, expected_shape)

    def test_forward_simple_pooling(self):
        pool = st.MaxPool1DLayer(pool_size=2, stride=2)

        input_tensor = p.Tensor(np.array([[[1.0, 5.0, 3.0, 7.0]]]))
        output = pool.forward(input_tensor)

        expected = np.array([[[5.0, 7.0]]])
        assert_allclose(output.value, expected, rtol=1e-5)

    def test_backward_gradient_flow(self):
        pool = st.MaxPool1DLayer(pool_size=2, stride=2)
        input_tensor = p.Tensor(np.random.randn(1, 2, 6))

        output = pool.forward(input_tensor)
        grad = np.ones_like(output.value)
        output.backward(grad)

        self.assertTrue(np.any(input_tensor.grad != 0))

    def test_backward_stride(self):
        pool = st.MaxPool1DLayer(pool_size=3, stride=2)
        input_tensor = p.Tensor(np.random.randn(1, 2, 12))

        output = pool.forward(input_tensor)
        grad = np.ones_like(output.value)
        output.backward(grad)

        self.assertTrue(np.any(input_tensor.grad != 0))

    def test_forward_against_pytorch(self):
        # 1. Setup parameters
        batch_size = 4
        channels = 3
        length = 20
        pool_size = 3
        stride = 2

        # 2. Create input data
        input_data = np.random.randn(batch_size, channels, length)

        # 3. Perform forward pass with the custom layer
        custom_pool = st.MaxPool1DLayer(pool_size=pool_size, stride=stride)
        custom_input = p.Tensor(input_data)
        custom_output = custom_pool.forward(custom_input)

        # 4. Perform forward pass with PyTorch
        pytorch_pool = torch.nn.MaxPool1d(kernel_size=pool_size, stride=stride)
        input_torch = torch.from_numpy(input_data).float()
        pytorch_output = pytorch_pool(input_torch)

        # 5. Compare the results
        assert_allclose(custom_output.value, pytorch_output.detach().numpy(), rtol=1e-5)

    def test_backward_against_pytorch(self):
        # 1. Setup parameters
        batch_size = 4
        channels = 3
        length = 20
        pool_size = 3
        stride = 2

        # 2. Create common data
        input_data = np.random.randn(batch_size, channels, length)
        # Calculate output shape to create a matching gradient
        output_length = (length - pool_size) // stride + 1
        output_grad_np = np.random.randn(batch_size, channels, output_length)

        # 3. Perform backward pass with the custom layer
        custom_pool = st.MaxPool1DLayer(pool_size=pool_size, stride=stride)
        custom_input = p.Tensor(input_data)
        custom_output = custom_pool.forward(custom_input)
        custom_output.backward(output_grad_np)

        # 4. Perform backward pass with PyTorch
        pytorch_pool = torch.nn.MaxPool1d(kernel_size=pool_size, stride=stride)
        input_torch = torch.from_numpy(input_data).float()
        input_torch.requires_grad = True

        pytorch_output = pytorch_pool(input_torch)
        output_grad_torch = torch.from_numpy(output_grad_np).float()
        pytorch_output.backward(gradient=output_grad_torch)

        # 5. Compare the input gradients
        assert_allclose(custom_input.grad, input_torch.grad.numpy(), rtol=1e-5)


t = TestConv1DLayer()


t.test_forward_output_shape()
t.test_forward_simple_convolution()
t.test_backward_gradient_flow()
t.test_forward_against_pytorch()
t.test_backward_against_pytorch()


t = TestMaxPool1DLayer()


t.test_forward_output_shape()
t.test_forward_simple_pooling()
t.test_backward_gradient_flow()
t.test_backward_stride()
t.test_forward_against_pytorch()
t.test_backward_against_pytorch()

