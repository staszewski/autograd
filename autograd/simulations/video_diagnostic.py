"""
Video diagnostic tool for troubleshooting USB video capture and FPV receiver setup.

This script helps you:
1. Find available video devices
2. Test video signal from RC832 receiver
3. Diagnose why the video feed might be dark
4. Display live video feed with statistics

Usage:
    python -m autograd.simulations.video_diagnostic
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autograd.drone_problems.video_capture import VideoCapture, list_available_cameras, test_video_signal


def print_troubleshooting_tips():
    """Print common troubleshooting steps for dark video."""
    print("\n" + "="*70)
    print("TROUBLESHOOTING: Dark Video Feed from RC832 Receiver")
    print("="*70)
    print("\n📡 VTX (Video Transmitter) Setup:")
    print("   1. Power ON your BetaFPV Meteor 75 drone")
    print("   2. Check if VTX LED is lit (indicates it's transmitting)")
    print("   3. Default VTX frequency: Check your drone's manual")
    print("      - Common: 5.8GHz, channels R1-R8 or F1-F8")
    print("      - BetaFPV Meteor 75 likely uses 25mW VTX")
    print()
    print("📺 RC832 Receiver Setup:")
    print("   1. Power ON the RC832 receiver (needs 5V power)")
    print("   2. Press CH button to cycle through channels")
    print("   3. Press FR button to cycle through frequency bands")
    print("   4. Match the band/channel to your drone's VTX")
    print("      - Example: If drone is on R1, set receiver to R1")
    print("   5. You should see the frequency on RC832's display")
    print()
    print("🔌 Physical Connections:")
    print("   - RC832 Video OUT → USB Capture Card IN")
    print("   - USB Capture Card → Computer USB port")
    print("   - RC832 powered (via USB or external 5V)")
    print()
    print("📶 Signal Strength:")
    print("   - RC832 should show signal strength indicator")
    print("   - If receiver shows signal but feed is dark:")
    print("     → Check video cable connection")
    print("     → Try different video input on capture card")
    print("   - If no signal indicator:")
    print("     → VTX and receiver are on different frequencies")
    print("     → VTX might not be powered or transmitting")
    print()
    print("🎯 Quick Test:")
    print("   - RC832 often has built-in screen - does IT show video?")
    print("   - If RC832 screen shows video but computer doesn't:")
    print("     → Issue is with USB capture card or connection")
    print("   - If RC832 screen is also dark:")
    print("     → Issue is with VTX/receiver pairing")
    print()
    print("="*70 + "\n")


def display_live_feed(device_id: int = 0):
    """
    Display live video feed with real-time statistics and diagnostics.
    
    Args:
        device_id: Video device index to display
    """
    print(f"\n🎥 Opening live video feed from device {device_id}...")
    print("   Press 'q' to quit")
    print("   Press 's' to save current frame")
    print()
    
    with VideoCapture(device_id, width=640, height=480, fps=30) as cap:
        if not cap.is_open:
            print("❌ ERROR: Could not open video device")
            return
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        saved_frames = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("⚠️  WARNING: Failed to read frame")
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
            
            # Get frame statistics
            stats = cap.get_frame_stats(frame)
            
            # Create display frame with text overlay
            display_frame = frame.copy()
            
            # Add dark overlay at top for text visibility
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            
            # Display statistics
            y_offset = 25
            line_height = 25
            
            # Status indicator
            if stats['is_dark']:
                status_text = "🔴 DARK - NO VIDEO SIGNAL"
                color = (0, 0, 255)  # Red
            else:
                status_text = "🟢 VIDEO SIGNAL DETECTED"
                color = (0, 255, 0)  # Green
            
            cv2.putText(display_frame, status_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height
            
            # FPS
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # Brightness stats
            cv2.putText(display_frame, 
                       f"Brightness: {stats['mean_brightness']:.1f} (std: {stats['std_brightness']:.1f})",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # Range
            cv2.putText(display_frame, 
                       f"Range: {stats['min_value']} - {stats['max_value']}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('FPV Video Feed - Diagnostic', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                filename = f'captured_frame_{saved_frames:03d}.jpg'
                cv2.imwrite(filename, frame)
                print(f"💾 Saved frame to {filename}")
                saved_frames += 1
        
        cv2.destroyAllWindows()
        print(f"\n✅ Captured {frame_count} frames total")


def main():
    """Main diagnostic routine."""
    print("\n" + "="*70)
    print("FPV Video Capture Diagnostic Tool")
    print("="*70)
    
    # Step 1: List available cameras
    print("\n📹 Step 1: Scanning for video devices...")
    available = list_available_cameras()
    
    if not available:
        print("\n❌ ERROR: No video devices found!")
        print("\nPossible issues:")
        print("  - USB capture card not connected")
        print("  - Driver not installed")
        print("  - Device permissions issue (try with sudo on Linux)")
        return
    
    print(f"\n✅ Found {len(available)} video device(s): {available}")
    
    # Step 2: Select device
    if len(available) == 1:
        device_id = available[0]
        print(f"\n📹 Using device {device_id}")
    else:
        print("\nMultiple devices detected. Which one is your RC832 receiver?")
        for dev_id in available:
            print(f"  {dev_id}: Device {dev_id}")
        
        try:
            device_id = int(input("\nEnter device number: "))
            if device_id not in available:
                print(f"Invalid device ID. Using {available[0]}")
                device_id = available[0]
        except:
            device_id = available[0]
            print(f"Invalid input. Using device {device_id}")
    
    # Step 3: Test signal
    print("\n📡 Step 2: Testing video signal...")
    test_results = test_video_signal(device_id, duration=3)
    
    if 'error' in test_results:
        print(f"\n❌ ERROR: {test_results['error']}")
        return
    
    print(f"\n📊 Signal Test Results:")
    print(f"   Frames captured: {test_results['total_frames']}")
    print(f"   Dark frames: {test_results['dark_frames']} ({test_results['dark_frame_ratio']*100:.1f}%)")
    print(f"   Average brightness: {test_results['avg_brightness']:.1f}")
    print(f"   Brightness variation: {test_results['brightness_std']:.1f}")
    
    if test_results['signal_detected']:
        print("\n   ✅ Video signal detected!")
    else:
        print("\n   ❌ NO VIDEO SIGNAL - Feed appears dark/black")
        print_troubleshooting_tips()
    
    # Step 4: Ask to display live feed
    print("\n" + "="*70)
    response = input("\nDisplay live video feed? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        display_live_feed(device_id)
    
    print("\n✅ Diagnostic complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()



