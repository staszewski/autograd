from autograd.loss import MSELoss
from autograd.mlp import MLP
from autograd.tensor import Tensor
import numpy as np
from autograd.operations.conv_2d_operation import Conv2DOperation, FlattenOperation

class SimpleTargetDetector:
    def __init__(self) -> None:
        self.conv_kernel = Tensor([[1, -1], [-1, 1]], requires_grad=True)
        self.mlp = MLP(input_size=16, hidden_size=5, output_size=1, activation="tanh")
        pass

    def forward(self, image):
        features = Conv2DOperation.apply(image, self.conv_kernel)
        flat_features = FlattenOperation.apply(features)
        prediction = self.mlp(flat_features)
        return prediction

    def zero_gradients(self):
        for param in self.mlp.parameters():
            param.zero_grad()
        self.conv_kernel.zero_grad()

def generate_tensor_examples(num_target=50, num_non_target=50, size=5):
    tensors = []
    labels = []
    
    for _ in range(num_target):
        array = np.zeros((size, size), dtype=np.float32)
        array[:, size // 2] = 1
        array[size // 2, :] = 1
        tensors.append(Tensor(array))
        labels.append(Tensor([[1.0]]))
    
    for _ in range(num_non_target):
        array = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3]).astype(np.float32)
        while (array[:, size // 2].sum() >= size or array[size // 2, :].sum() >= size):
            array = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3]).astype(np.float32)
        tensors.append(Tensor(array))
        labels.append(Tensor([[0.0]]))
    
    return tensors, labels

def train_detector(epochs=100, learning_rate = 0.01):
    detector = SimpleTargetDetector()
    tensors, labels = generate_tensor_examples()

    for epoch in range(epochs):
        total_loss = 0
        for image, target in zip (tensors, labels):
            predicition = detector.forward(image)
            loss = MSELoss.apply(predicition, target)

            total_loss += loss.data

            detector.zero_gradients()

            loss.backward()

            for param in detector.mlp.parameters():
                param._data -= learning_rate * param._grad

            detector.conv_kernel._data -= learning_rate * detector.conv_kernel._grad

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

    return detector            

def test_trained_detector(detector):
    test_target = Tensor(np.array([
        [0,0,1,0,0], 
        [0,0,1,0,0], 
        [1,1,1,1,1], 
        [0,0,1,0,0], 
        [0,0,1,0,0]
    ]).astype(np.float32))
    
    test_non_target = Tensor(np.random.choice([0,1], (5,5), p=[0.7, 0.3]).astype(np.float32))
    
    target_pred = detector.forward(test_target)
    non_target_pred = detector.forward(test_non_target)
    
    print(f"Target prediction: {target_pred.data[0][0]:.4f} (should be close to 1.0)")
    print(f"Non-target prediction: {non_target_pred.data[0][0]:.4f} (should be close to 0.0)")

def demo_detector():
    trained_detector = train_detector()
    test_trained_detector(trained_detector)

if __name__ == "__main__":
    demo_detector()