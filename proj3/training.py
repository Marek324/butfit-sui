import argparse
import numpy as np
import matplotlib.pyplot as plt

import primitives as p
import sui_torch as st
import data_generation as dg


class LinearLayer:
    def __init__(self, out_features, in_features):
        self.weights = p.Tensor(np.random.randn(out_features, in_features))
        self.bias = p.Tensor(np.random.randn(out_features)[:, None])

    def forward(self, x):
        return p.add(p.dot_product(self.weights, x), self.bias)

    def parameters(self):
        return [self.weights, self.bias]


class SignalClassifier:
    def __init__(self, signal_length, num_classes=3):
        kernel_size = 5
        conv_out_channels = 8

        self.conv1 = st.Conv1DLayer(out_channels=conv_out_channels, kernel_size=kernel_size)
        pool_size = (signal_length - kernel_size + 1) // 2
        self.pool1 = st.MaxPool1DLayer(pool_size=pool_size, stride=1)
        self.fc1 = LinearLayer(num_classes, conv_out_channels * pool_size)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = p.relu(x)
        x = self.pool1.forward(x)
        x = p.relu(x)

        # Hack pro zjednoduseni implementace
        batch_size, channels, length = x.value.shape 
        x = p.reshape(x, (channels * length, batch_size))
        x = self.fc1.forward(x)
        x = p.reshape(x, (batch_size, x.value.shape[0]))

        return x

    def parameters(self):
        return self.conv1.parameters() + self.fc1.parameters()# + self.fc2.parameters()


class StochasticGradientDescent:
    def __init__(self, tensors, lr):
        self.tensors = tensors
        self.lr = lr

    def step(self):
        for tensor in self.tensors:
            tensor.value -= self.lr * tensor.grad

    def zero_grad(self):
        for tensor in self.tensors:
            tensor.grad.fill(0)


def train_model(model, train_signals, train_labels, optimizer, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        
        for signal, label in zip(train_signals, train_labels):
            optimizer.zero_grad()
            
            signal = signal.reshape(1, signal.shape[0])
            signal_tensor = p.Tensor(signal)
            logits = model.forward(signal_tensor)
            
            if np.any(np.isnan(logits.value)) or np.any(np.isinf(logits.value)):
                print("NaN or Inf in model output, you are probably doing something wrong.")
                continue
            
            loss = p.cross_entropy_loss(logits, [label])
            loss_scalar = float(loss.value.item() if hasattr(loss.value, "item") else loss.value)
            
            if np.isnan(loss_scalar) or np.isinf(loss_scalar):
                print("NaN or Inf in loss, you are probably doing something wrong.")
                continue
            
            epoch_loss += loss_scalar
            if np.argmax(logits.value, axis=-1)[0] == label:
                correct += 1
            
            loss.backward()
            optimizer.step()
        
        avg_loss = epoch_loss / len(train_signals) if len(train_signals) > 0 else 0.0
        accuracy = correct / len(train_signals) if len(train_signals) > 0 else 0.0
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")
    
    return losses


def evaluate_model(model, tep_signals, tep_labels):
    total_loss = 0.0
    correct = 0
    
    for signal, label in zip(tep_signals, tep_labels):
        signal = signal.reshape(1, signal.shape[0])
        signal_tensor = p.Tensor(signal)
        logits = model.forward(signal_tensor)
        loss = p.cross_entropy_loss(logits, [label])
        total_loss += float(loss.value.item())
        if np.argmax(logits.value, axis=-1)[0] == label:
            correct += 1
    
    avg_loss = total_loss / len(tep_signals) if len(tep_signals) > 0 else 0.0
    accuracy = correct / len(tep_signals) if len(tep_signals) > 0 else 0.0
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN for signal classification')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--signal-length', type=int, default=50)
    parser.add_argument('--do-plots', action='store_true')
    parser.add_argument('--save-plot', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    train_signals, train_labels = dg.generate_dataset(args.num_samples, args.signal_length)
    tep_signals, tep_labels = dg.generate_dataset(args.num_samples // 4, args.signal_length)
    
    print(f"Train samples: {len(train_signals)}")
    print(f"Tep samples: {len(tep_signals)}")
    
    model = SignalClassifier(signal_length=args.signal_length, num_classes=3)
    optimizer = StochasticGradientDescent(model.parameters(), lr=args.lr)
    
    print("\nTraining...")
    losses = train_model(model, train_signals, train_labels, optimizer, args.num_epochs)
    
    print("\nEvaluating...")
    eval_loss, eval_acc = evaluate_model(model, tep_signals, tep_labels)
    
    print(f"\nResults:")
    print(f"Tep Loss: {eval_loss:.4f} | Tep Acc: {eval_acc:.2%}")
    
    if args.do_plots or args.save_plot:
        class_names = ['Sin/Cos', 'Square', 'Noise']
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        # Generate one example from each class
        for class_idx in range(3):
            if class_idx == 0:
                # Sin/Cos
                signal = dg.generate_signal(np.cos, args.signal_length, phase=0.0)
            elif class_idx == 1:
                # Square wave
                square_fn = lambda t: np.sign(np.sin(t))
                signal = dg.generate_signal(square_fn, args.signal_length, phase=0.0)
            else:
                # Noise
                signal = dg.generate_noise(args.signal_length)
            
            signal = signal.reshape(1, signal.shape[0])
            signal_tensor = p.Tensor(signal)
            logits = model.forward(signal_tensor)
            predicted = np.argmax(logits.value, axis=-1)[0]
            
            plt.subplot(2, 2, class_idx + 2)
            plt.plot(signal[0])
            plt.title(f'True: {class_names[class_idx]}, Pred: {class_names[predicted]}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {args.save_plot}")
        
        if args.do_plots:
            plt.show()


if __name__ == '__main__':
    main()
