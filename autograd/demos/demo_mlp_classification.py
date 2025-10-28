from autograd import Tensor
from autograd.loss import CrossEntropyLoss, MSELoss
import numpy as np
import matplotlib.pyplot as plt

class LinearClassifier:
    def __init__(self, input_size):
        self.W = Tensor(np.random.randn(1, input_size) * 0.1, requires_grad=True)
        self.b = Tensor(np.random.randn(1, 1), requires_grad=True)
    
    def __call__(self, x):
        return (self.W @ x + self.b).sigmoid()

    def parameters(self):
        return [self.W, self.b]

def plot_decision_boundary(classifier, X, y, title):
    x_min, x_max = X.data[0].min() - 1, X.data[0].max() + 1
    y_min, y_max = X.data[1].min() - 1, X.data[1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  
    grid_tensor = Tensor(grid_points, requires_grad=False)
    predictions = classifier(grid_tensor)
    
    Z = predictions.data.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    class_0_mask = y.data[0] == 0
    class_1_mask = y.data[0] == 1
    plt.scatter(X.data[0][class_0_mask], X.data[1][class_0_mask], 
                c='red', marker='o', s=100, edgecolors='black', label='Class 0')
    plt.scatter(X.data[0][class_1_mask], X.data[1][class_1_mask], 
                c='blue', marker='s', s=100, edgecolors='black', label='Class 1')
    
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True, alpha=0.3)

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
    plot_decision_boundary(ce_classifier, X, y, "Cross-Entropy Decision Boundary")
    plot_decision_boundary(mse_classifier, X, y, "MSE Decision Boundary")
    plt.show()

if __name__ == "__main__":
    compare_losses()