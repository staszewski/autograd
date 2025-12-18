"""
Video capture module for USB video capture devices (RC832 receiver).

This module provides a simple interface to capture video frames from
USB video capture cards connected to FPV receivers.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import platform


class VideoCapture:
    """
    Handles video capture from USB video capture devices.
    
    Attributes:
        device_id: The video device ID (e.g., 0, 1, 2 for /dev/video0, /dev/video1, etc.)
        width: Desired frame width
        height: Desired frame height
        fps: Target frames per second
        cap: OpenCV VideoCapture object
    """
    
    def __init__(
        self, 
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        """
        Initialize video capture device.
        
        Args:
            device_id: Camera device index (0 for first device, 1 for second, etc.)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        
    def open(self) -> bool:
        """
        Open the video capture device.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # On macOS, use AVFoundation backend for better compatibility
            if platform.system() == 'Darwin':
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_AVFOUNDATION)
            else:
                # On Linux, try V4L2 first (better for USB capture cards)
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                print(f"Failed to open device {self.device_id}, trying default backend...")
                self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                print(f"ERROR: Could not open video device {self.device_id}")
                return False
            
            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # On macOS with USB capture cards, sometimes need to set FOURCC format
            # Try MJPG format which is common for USB capture devices
            if platform.system() == 'Darwin':
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Verify actual settings (device may not support requested values)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"Video device {self.device_id} opened successfully")
            print(f"  Requested: {self.width}x{self.height} @ {self.fps} FPS")
            print(f"  Actual:    {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            # Try to grab a test frame to verify device is actually working
            print(f"  Testing frame capture...")
            for attempt in range(5):
                ret = self.cap.grab()
                if ret:
                    print(f"  ✓ Frame capture working (attempt {attempt + 1})")
                    break
                import time
                time.sleep(0.1)
            else:
                print(f"  ⚠️  WARNING: Device opened but cannot grab frames")
                print(f"      This usually means no video signal is being received")
            
            self.is_open = True
            return True
            
        except Exception as e:
            print(f"ERROR: Exception while opening video device: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video capture device.
        
        Returns:
            Tuple of (success, frame) where:
                - success: True if frame was read successfully
                - frame: NumPy array of shape (height, width, 3) in BGR format,
                        or None if read failed
        """
        if not self.is_open or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("WARNING: Failed to read frame from video device")
            return False, None
        
        return True, frame
    
    def get_frame_stats(self, frame: np.ndarray) -> dict:
        """
        Analyze frame statistics to help diagnose video issues.
        
        Args:
            frame: Input frame as NumPy array
            
        Returns:
            Dictionary with statistics:
                - mean_brightness: Average pixel intensity (0-255)
                - std_brightness: Standard deviation of pixel intensity
                - min_value: Minimum pixel value
                - max_value: Maximum pixel value
                - is_dark: True if frame appears to be dark/black
        """
        if frame is None or frame.size == 0:
            return {
                'mean_brightness': 0,
                'std_brightness': 0,
                'min_value': 0,
                'max_value': 0,
                'is_dark': True
            }
        
        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        min_value = int(np.min(gray))
        max_value = int(np.max(gray))
        
        # Consider frame dark if mean brightness < 20 and std < 5
        is_dark = mean_brightness < 20 and std_brightness < 5
        
        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'min_value': min_value,
            'max_value': max_value,
            'is_dark': is_dark
        }
    
    def close(self):
        """Release the video capture device."""
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            print(f"Video device {self.device_id} closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def list_available_cameras() -> List[int]:
    """
    Enumerate all available video capture devices.
    
    Returns:
        List of device IDs that are available
    """
    available = []
    
    # Try first 10 device indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            # Get device name if possible
            backend = cap.getBackendName()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Device {i}: {backend} - {width}x{height}")
            cap.release()
    
    return available


def test_video_signal(device_id: int = 0, duration: int = 5) -> dict:
    """
    Test video signal quality for a specified duration.
    
    Args:
        device_id: Camera device index
        duration: Test duration in seconds
        
    Returns:
        Dictionary with test results
    """
    print(f"\nTesting video signal on device {device_id} for {duration} seconds...")
    
    with VideoCapture(device_id) as cap:
        if not cap.is_open:
            return {'error': 'Could not open device'}
        
        frame_count = 0
        dark_frames = 0
        brightness_values = []
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_count += 1
                stats = cap.get_frame_stats(frame)
                brightness_values.append(stats['mean_brightness'])
                
                if stats['is_dark']:
                    dark_frames += 1
        
        if frame_count == 0:
            return {'error': 'No frames captured'}
        
        return {
            'total_frames': frame_count,
            'dark_frames': dark_frames,
            'dark_frame_ratio': dark_frames / frame_count,
            'avg_brightness': np.mean(brightness_values),
            'brightness_std': np.std(brightness_values),
            'signal_detected': dark_frames / frame_count < 0.9  # If >90% dark, no signal
        }

