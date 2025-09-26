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
        image_height, image_width = image.data.shape
    
        end_i = start_i + patch_size
        end_j = start_j + patch_size
    
        # If patch goes beyond image, pad with zeros
        if end_i > image_height or end_j > image_width:
            patch_data = np.zeros((patch_size, patch_size), dtype=np.float32)
        
            valid_end_i = min(end_i, image_height)
            valid_end_j = min(end_j, image_width)
        
            valid_height = valid_end_i - start_i
            valid_width = valid_end_j - start_j
            patch_data[:valid_height, :valid_width] = image.data[start_i:valid_end_i, start_j:valid_end_j]
        else:
            patch_data = image.data[start_i:start_i+patch_size, start_j:start_j+patch_size]
    
        return Tensor(patch_data)

    def detect_all_targets(self, image, threshold=THRESHOLD):
        detected_positions = []
        image_size = image._data.shape[0]

        for i in range(image_size):
            for j in range(image_size):
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
    
    print("Training...")
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

    true_targets = [(2, 3), (6, 7)]

    print("True target positions:")
    for i, pos in enumerate(true_targets):
        print(f"  Target {i+1}: {pos}")

    for i, true_pos in enumerate(true_targets):
        print(f"\nTarget {i+1} at {true_pos}:")
        close_detections = []
        for det_pos in detected_positions:
            distance = abs(true_pos[0] - det_pos[0]) + abs(true_pos[1] - det_pos[1])
            if distance <= 2:
                close_detections.append((det_pos, distance))
    
        if close_detections:
            print(f"Found nearby: {close_detections}")
        else:
            print(f"MISSED - no detections within 2 pixels")

    # target 1 at (2, 3)
    patch_at_target1 = detector.extract_patch(test_tensor, 2, 3)
    prediction1 = detector.single_detector.forward(patch_at_target1)
    print(f"Prediction at target 1 location (2,3): {prediction1.data[0][0]:.4f}")
    
    # target 2 at (6, 7)
    patch_at_target2 = detector.extract_patch(test_tensor, 6, 7)
    prediction2 = detector.single_detector.forward(patch_at_target2)
    print(f"Prediction at target 2 location (6,7): {prediction2.data[0][0]:.4f}")
    
    print(f"Current threshold: {detector.THRESHOLD}")

    print("\nHigh-confidence patches (>0.6):")
    image_size = test_tensor._data.shape[0]
    for i in range(image_size):
        for j in range(image_size):
            patch = detector.extract_patch(test_tensor, i, j)
            prediction = detector.single_detector.forward(patch)
            if prediction.data[0][0] > 0.6:
                print(f"Position ({i},{j}): {prediction.data[0][0]:.4f}")

if __name__ == "__main__":
    demo_multi_target_detection()