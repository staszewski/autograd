from autograd import Tensor
from autograd.loss import CrossEntropyLoss, MSELoss
import numpy as np

class LinearClassifier:
    def __init__(self, input_size):
        self.W = Tensor(np.random.randn(1, input_size) * 0.1, requires_grad=True)
        self.b = Tensor(np.random.randn(1, 1), requires_grad=True)
    
    def __call__(self, x):
        return (self.W @ x + self.b).sigmoid()

    def parameters(self):
        return [self.W, self.b]

def generate_2d_clusters():
    cluster1_x = np.random.normal(2, 0.5, 10)
    cluster1_y = np.random.normal(2, 0.5, 10)

    cluster2_x = np.random.normal(-2, 0.5, 10)
    cluster2_y = np.random.normal(-2, 0.5, 10)

    X = np.column_stack([
        np.concatenate([cluster1_x, cluster2_x]),
        np.concatenate([cluster1_y, cluster2_y])
    ]).T

    y = np.concatenate([np.zeros(10), np.ones(10)]).reshape(1, -1)

    return Tensor(X, requires_grad=False), Tensor(y, requires_grad=False)

def train_classifier(classifier, X, y, loss_fn, learning_rate, epochs):
    for epoch in range(epochs):
        predictions = classifier(X)
        loss = loss_fn.apply(predictions, y)
        for param in classifier.parameters():
            param.zero_grad()
        loss.backward()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

        for param in classifier.parameters():
            param._data -= learning_rate * param._grad

    final_predicitions = classifier(X)
    final_loss = loss_fn.apply(final_predicitions, y)
    return final_loss.data

def calculate_accuracy(predictions, targets):
    binary_predictions = (predictions.data > 0.5).astype(int)
    binary_targets = targets.data.astype(int)
    accuracy = np.mean(binary_predictions == binary_targets)
    return accuracy



def compare_losses():
    X, y = generate_2d_clusters()
    ce_classifier = LinearClassifier(input_size=2)
    mse_classifier = LinearClassifier(input_size=2)
    ce_final_loss = train_classifier(ce_classifier, X, y, CrossEntropyLoss, 0.2, 500)
    mse_final_loss = train_classifier(mse_classifier, X, y, MSELoss, 0.2, 500)
    ce_accuracy = calculate_accuracy(ce_classifier(X), y)
    mse_accuracy = calculate_accuracy(mse_classifier(X), y)
    print(f"Cross-Entropy final loss: {ce_final_loss:.4f}")
    print(f"MSE final loss: {mse_final_loss:.4f}")
    print("--------------------------------")
    print(f"Cross-Entropy final accuracy: {ce_accuracy:.4f}")
    print(f"MSE final accuracy: {mse_accuracy:.4f}")

if __name__ == "__main__":
    compare_losses()