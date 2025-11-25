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
