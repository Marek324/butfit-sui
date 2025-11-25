import numpy as np


def generate_signal(signal_function, length, phase=0.0):
    t = np.linspace(0, 4 * np.pi, length)
    signal = signal_function(t + phase)
    return signal


def generate_noise(length):
    # Chceme stejny rozsah jako u ostatnich signalu (tj. -1 az 1)
    return np.random.randn(length) * 2 - 1


# Vygeneruje dataset s 3 druhy signalu: sin/cos, square wave, sum
def generate_dataset(num_samples, signal_length):
    signals = []
    labels = []
    
    np.random.seed(42)
    samples_per_class = num_samples // 3
    
    # Generuj 1/3 cos
    for _ in range(samples_per_class):
        phase = np.random.uniform(0, 2 * np.pi)
        signal = generate_signal(np.cos, signal_length, phase=phase)
        signals.append(signal)
        labels.append(0)
    
    # Generuj 1/3 square wave
    for _ in range(samples_per_class):
        phase = np.random.uniform(0, 2 * np.pi)
        square_fn = lambda t: np.sign(np.sin(t))
        signal = generate_signal(square_fn, signal_length, phase=phase)
        signals.append(signal)
        labels.append(1)
    
    # Generuj 1/3 sumu
    for _ in range(samples_per_class):
        signal = generate_noise(signal_length)
        signals.append(signal)
        labels.append(2)
    
    indices = np.arange(len(signals))
    np.random.shuffle(indices)
    signals = [signals[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return signals, labels
