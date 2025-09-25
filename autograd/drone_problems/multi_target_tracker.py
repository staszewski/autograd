from autograd.drone_problems.demo_target_detector import SimpleTargetDetector
from autograd.tensor import Tensor
from autograd.loss import MSELoss
import numpy as np

class MultiTargetDetector:
    PATCH_SIZE = 5
    THRESHOLD = 0.6
    def __init__(self):
        self.single_detector = SimpleTargetDetector()

    def extract_patch(self, image, start_i, start_j, patch_size=PATCH_SIZE):
        patch_data = image.data[start_i:start_i+patch_size, start_j:start_j+patch_size]
        return Tensor(patch_data)

    def detect_all_targets(self, image, patch_size=PATCH_SIZE, threshold=THRESHOLD):
        detected_positions = []
        image_size = image._data.shape[0]
        max_position = image_size - patch_size + 1

        for i in range(max_position):
            for j in range(max_position):
                patch = self.extract_patch(image, i, j)
                detected_position = self.single_detector.forward(patch)

                if detected_position.data > threshold:
                    detected_positions.append((i, j))
        
        return detected_positions

    def train(self, epochs=100, learning_rate=0.01):
        training_images, training_labels = self.generate_multi_target_training_data()
        
        for epoch in range(epochs):
            total_loss = 0
            for image, label in zip(training_images, training_labels):
                patches, patch_labels = self.extract_training_patches(image, label)
                
                for patch, patch_label in zip(patches, patch_labels):
                    prediction = self.single_detector.forward(patch)
                    loss = MSELoss.apply(prediction, patch_label)
                    total_loss += loss.data
                    
                    self.single_detector.zero_gradients()
                    loss.backward()
                    
                    for param in self.single_detector.mlp.parameters():
                        param._data -= learning_rate * param._grad
                    self.single_detector.conv_kernel._data -= learning_rate * self.single_detector.conv_kernel._grad
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Average Loss: {total_loss/len(training_images):.4f}")

    def extract_training_patches(self, image, label):
        patches = []
        patch_labels = []
        image_size = image._data.shape[0]
        max_position = image_size - self.PATCH_SIZE + 1
        
        for i in range(max_position):
            for j in range(max_position):
                patch = self.extract_patch(image, i, j)
                patches.append(patch)
                
                has_target = any(abs(i - ti) < 2 and abs(j - tj) < 2 for ti, tj in label)
                patch_labels.append(Tensor([[1.0 if has_target else 0.0]]))
        
        return patches, patch_labels

    def generate_multi_target_training_data(self, num_images=20, image_size=10):
        images = []
        labels = []
        
        for _ in range(num_images):
            image_array = np.zeros((image_size, image_size), dtype=np.float32)
            target_positions = []
            
            num_targets = np.random.randint(1, 4)
            for _ in range(num_targets):
                ti = np.random.randint(2, image_size - 3)
                tj = np.random.randint(2, image_size - 3)
                
                image_array[:, tj] = 0.3
                image_array[ti, :] = 0.3
                image_array[:, tj] = 1.0
                image_array[ti, :] = 1.0
                
                target_positions.append((ti, tj))
            
            images.append(Tensor(image_array))
            labels.append(target_positions)
        
        return images, labels

def demo_multi_target_detection():
    detector = MultiTargetDetector()
    
    print("Training multi-target detector...")
    detector.train(epochs=100, learning_rate=0.01)
    
    print("\nTesting trained detector:")
    test_image = np.zeros((10, 10), dtype=np.float32)
    test_image[:, 3] = 1.0
    test_image[2, :] = 1.0
    test_image[:, 7] = 1.0
    test_image[6, :] = 1.0
    
    test_tensor = Tensor(test_image)
    detected_positions = detector.detect_all_targets(test_tensor)
    print(f"Detected targets at positions: {detected_positions}")

if __name__ == "__main__":
    demo_multi_target_detection()