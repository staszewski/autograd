#!/usr/bin/env python3
"""
Quick test script for RC832 video feed.
Run this interactively to test your FPV video capture.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from autograd.drone_problems.video_capture import VideoCapture


def test_device(device_id):
    """Test a specific video device."""
    print(f"\n{'='*70}")
    print(f"Testing Device {device_id}")
    print(f"{'='*70}")
    print("\nPress 'q' to quit and try next device")
    print("Press 's' to save current frame")
    print()
    
    with VideoCapture(device_id, width=640, height=480, fps=30) as cap:
        if not cap.is_open:
            print(f"❌ Could not open device {device_id}")
            return
        
        frame_count = 0
        failed_reads = 0
        max_failed_reads = 30  # Exit after 30 consecutive failures
        
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads == 1:
                    print("⚠️  Failed to read frame - waiting for video signal...")
                elif failed_reads >= max_failed_reads:
                    print(f"\n❌ No frames received after {max_failed_reads} attempts")
                    print("   USB capture card is connected but NOT receiving video signal")
                    print("\n   Possible causes:")
                    print("   1. 🔋 Drone VTX not powered on")
                    print("   2. 📡 RC832 receiver on wrong frequency/channel")
                    print("   3. 🔌 RC832 not connected to USB capture card")
                    print("   4. 📺 RC832 needs 5V power supply")
                    break
                time.sleep(0.1)
                continue
            
            # Successfully read a frame!
            if failed_reads > 0:
                print(f"✅ Video signal detected after {failed_reads} failed attempts!")
            failed_reads = 0
            
            frame_count += 1
            
            # Get stats
            stats = cap.get_frame_stats(frame)
            
            # Create display
            display = frame.copy()
            
            # Add info overlay
            if stats['is_dark']:
                status = f"DARK - Brightness: {stats['mean_brightness']:.1f}"
                color = (0, 0, 255)
            else:
                status = f"Signal OK - Brightness: {stats['mean_brightness']:.1f}"
                color = (0, 255, 0)
            
            cv2.putText(display, f"Device {device_id} | {status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow(f'Device {device_id} - FPV Feed', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'device_{device_id}_frame.jpg'
                cv2.imwrite(filename, frame)
                print(f"💾 Saved to {filename}")
        
        cv2.destroyAllWindows()
        print(f"Captured {frame_count} frames from device {device_id}\n")


def main():
    print("\n" + "="*70)
    print("RC832 Video Feed Test")
    print("="*70)
    print("\n🎥 Your system has 2 video devices:")
    print("   Device 0: Likely MacBook camera")
    print("   Device 1: Likely USB capture card (RC832)")
    print()
    
    # Test both devices
    choice = input("Which device to test? (0, 1, or 'both'): ").strip().lower()
    
    if choice == 'both':
        devices = [0, 1]
    elif choice in ['0', '1']:
        devices = [int(choice)]
    else:
        print("Invalid choice. Testing device 1 (USB capture card)...")
        devices = [1]
    
    for device_id in devices:
        test_device(device_id)
    
    print("\n" + "="*70)
    print("🔍 TROUBLESHOOTING if Device 1 is dark:")
    print("="*70)
    print("\n1. ⚡ Power on your BetaFPV Meteor 75 drone")
    print("   - Check that VTX LED is lit")
    print()
    print("2. 📡 RC832 Receiver:")
    print("   - Make sure RC832 is powered (needs 5V)")
    print("   - Press CH button to change channel")
    print("   - Press FR button to change frequency band")
    print("   - Match the frequency to your drone's VTX")
    print()
    print("3. 🔌 Connections:")
    print("   - RC832 video OUT → USB capture card video IN")
    print("   - USB capture card → Mac USB port")
    print()
    print("4. 📺 RC832 Display:")
    print("   - Does the RC832's built-in screen show video?")
    print("   - If YES: problem is with USB capture card")
    print("   - If NO: VTX and receiver not paired")
    print()
    print("5. 📶 Common VTX Frequencies for BetaFPV:")
    print("   - Often ships on Band R (Raceband)")
    print("   - Try channels R1-R8 (5658-5917 MHz)")
    print("   - Check your drone's manual or Betaflight OSD")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        cv2.destroyAllWindows()

