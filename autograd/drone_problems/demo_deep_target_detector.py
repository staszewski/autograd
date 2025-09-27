from autograd.drone_problems.demo_target_detector import generate_tensor_examples, train_detector as train_simple_detector
from autograd.loss import MSELoss
from autograd.mlp import MLP
from autograd.operations.conv_2d_operation import Conv2DOperation, FlattenOperation
from autograd.operations.max_pool_2d_operation import MaxPool2dOperation
from autograd.tensor import Tensor
import numpy as np

class DeepTargetDetector:
    def __init__(self):
        self.conv1_kernel = Tensor(np.random.randn(2, 2) * 0.01, requires_grad=True)
        self.mlp = MLP(input_size=1, hidden_size=8, output_size=1, activation="relu")

    def forward(self, image):
        # Conv + ReLU + MaxPool + Flatten + MLP
        features = Conv2DOperation.apply(image, self.conv1_kernel).relu()
        pooled = MaxPool2dOperation.apply(features, 3) 
        flat = FlattenOperation.apply(pooled)
        return self.mlp(flat)

    def zero_gradients(self):
        for param in self.mlp.parameters():
            param.zero_grad()
        self.conv1_kernel.zero_grad()

def train_deep_detector(epochs=200, learning_rate = 0.005):
    detector = DeepTargetDetector()
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

            detector.conv1_kernel._data -= learning_rate * detector.conv1_kernel._grad

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
                print(f"Conv1 gradient norm: {np.linalg.norm(detector.conv1_kernel._grad):.6f}")
                print(f"MLP W1 gradient norm: {np.linalg.norm(detector.mlp.W1._grad):.6f}")

    return detector 

def compare_detectors():
    simple_detector = train_simple_detector(epochs=100)
    deep_detector = train_deep_detector(epochs=100)


    test_images, test_labels = generate_tensor_examples(num_target=10, num_non_target=10)
    
    simple_correct = 0
    deep_correct = 0
    for image, label in zip(test_images, test_labels):
        simple_pred = simple_detector.forward(image)
        deep_pred = deep_detector.forward(image)
        
        simple_correct += int((simple_pred.data[0][0] > 0.5) == (label.data[0][0] > 0.5))
        deep_correct += int((deep_pred.data[0][0] > 0.5) == (label.data[0][0] > 0.5))
    
    print(f"Simple Detector Accuracy: {simple_correct/len(test_images)*100:.1f}%")
    print(f"Deep Detector Accuracy: {deep_correct/len(test_images)*100:.1f}%")

if __name__ == "__main__":
    compare_detectors()