import numpy as np


class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = "lambda" if self.back_op and callable(self.back_op) else "None"
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'

    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):
        if deltas is not None:
            assert deltas.shape == self.value.shape, f'Expected gradient with shape {self.value.shape}, got {deltas.shape}'
            self.grad += deltas
            
            if self.back_op:
                self.back_op(self.grad)
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')

            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            self.grad = np.ones((1, 1))
            if self.back_op:
                self.back_op(self.grad)


def sui_sum(tensor):
    sum_tensor = Tensor(np.array([[tensor.value.sum()]]), back_op=None)
    back_op = lambda grad: tensor.backward(np.broadcast_to(grad, tensor.value.shape))
    sum_tensor.back_op = back_op
    return sum_tensor


def add(a, b):
    c = a.value + b.value
    back_op = lambda grad: (a.backward(grad), b.backward(grad))
    return Tensor(c, back_op=back_op)


def multiply(a, b):
    c = a.value * b.value
    back_op = lambda grad: (
        a.backward(grad * b.value),
        b.backward(grad * a.value)
    )
    return Tensor(c, back_op=back_op)


def relu(tensor):
    out = np.maximum(tensor.value, 0)
    back_op = lambda grad: tensor.backward(grad * np.where(tensor.value > 0, 1, 0))
    return Tensor(out, back_op=back_op)


def exp(tensor):
    out = np.exp(tensor.value)
    back_op = lambda grad: tensor.backward(grad * out)
    return Tensor(out, back_op=back_op)


def log(tensor):
    out = np.log(tensor.value + 1e-8) # Eps
    back_op = lambda grad: tensor.backward(grad / (tensor.value + 1e-8))
    return Tensor(out, back_op=back_op)


def softmax(tensor):
    max_val = np.max(tensor.value, axis=-1, keepdims=True)
    exp_input = np.exp(tensor.value - max_val)
    sum_exp = np.sum(exp_input, axis=-1, keepdims=True)
    out = exp_input / sum_exp
    
    def softmax_back_op(grad):
        grad_sum = np.sum(grad * out, axis=-1, keepdims=True)
        grad_input = out * (grad - grad_sum)
        tensor.backward(grad_input)
    
    return Tensor(out, back_op=softmax_back_op)


def cross_entropy_loss(logits, target_labels):
    batch_size = logits.value.shape[0]
    num_classes = logits.value.shape[1]
    
    probs = softmax(logits)
    log_probs = log(probs)
    
    # Vytvor one-hot encoding pro kazdou tridu (z pohledu derivace konstanta)
    one_hot = np.zeros((batch_size, num_classes))
    for i, label in enumerate(target_labels):
        one_hot[i, label] = 1.0
    one_hot_tensor = Tensor(one_hot)

    selected = multiply(one_hot_tensor, log_probs)
    
    total_sum = sui_sum(selected)
    neg_one = Tensor(np.array([[-1.0 / batch_size]]))
    loss = multiply(total_sum, neg_one)
    
    return loss


def dot_product(a, b):
    c = np.dot(a.value, b.value)
    back_op = lambda grad: (
        a.backward(np.dot(grad, b.value.T)),
        b.backward(np.dot(a.value.T, grad))
    )
    return Tensor(c, back_op=back_op)


def reshape(tensor, new_shape):
    reshaped = tensor.value.reshape(new_shape)
    
    def reshape_back_op(grad):
        original_shape = tensor.value.shape
        grad_reshaped = grad.reshape(original_shape)
        tensor.backward(grad_reshaped)
    
    return Tensor(reshaped, back_op=reshape_back_op)


def stack_tensors(tensors, axis=1):
    tensor_values = [t.value for t in tensors]
    stacked = np.stack(tensor_values, axis=axis)
    
    def stack_back_op(grad):
        splits = np.split(grad, len(tensors), axis=axis)
        for tensor, split_grad in zip(tensors, splits):
            split_grad = np.squeeze(split_grad, axis=axis)
            tensor.backward(split_grad)
    
    return Tensor(stacked, back_op=stack_back_op)

