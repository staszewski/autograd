# RC832 Video Receiver Setup Guide

## Quick Reference

### Your Hardware Setup
- **Device 0**: USB Video Capture Card (connected to RC832)
- **Device 1**: MacBook Camera
- **Drone**: BetaFPV Meteor 75 with ELRS
- **Receiver**: RC832 (5.8GHz 48CH)

## Current Issue

✅ USB capture card is **detected** by macOS
❌ No video frames are being received - **no signal from RC832**

This means:
- Hardware connections are OK
- RC832 is not receiving video from drone VTX
- **Most likely**: Frequency mismatch between VTX and RC832

## Solution Steps

### 1. Power On Drone
```bash
# Plug in battery to BetaFPV Meteor 75
# Check that VTX LED is lit (usually red/blue LED)
# Remove props if testing indoors!
```

### 2. Find Your VTX Frequency

**Option A: Check Betaflight** (most reliable)
1. Connect drone to computer via USB
2. Open Betaflight Configurator
3. Go to **Video Transmitter** tab
4. Note the Band and Channel (e.g., R3)

**Option B: Common BetaFPV Defaults**
- Band: **Raceband (R)**
- Channel: **R1** or **R3**

### 3. Configure RC832 Receiver

**Buttons:**
- **FR**: Cycle frequency bands (A, B, E, F, R)
- **CH**: Cycle channels (1-8)

**Steps:**
1. Press **FR** until display shows correct band (e.g., **R**)
2. Press **CH** until display shows correct channel (e.g., **1** for R1)
3. Look for signal strength indicator on RC832 display

### 4. Verify Signal

**RC832 has built-in screen:**
- ✅ If screen shows video → VTX is transmitting correctly
- ❌ If screen is blank/snowy → Frequency mismatch

## Common Raceband Frequencies

| Channel | Frequency |
|---------|-----------|
| R1 | 5658 MHz |
| R2 | 5695 MHz |
| R3 | 5732 MHz |
| R4 | 5769 MHz |
| R5 | 5806 MHz |
| R6 | 5843 MHz |
| R7 | 5880 MHz |
| R8 | 5917 MHz |

## Testing Scripts

### Quick Live Feed Test
```bash
python test_rc832.py
```
Opens live video feed with statistics overlay. Press 's' to save frames, 'q' to quit.

### Full Diagnostics (if needed)
```bash
python -m autograd.simulations.video_diagnostic
```
Complete diagnostic tool with troubleshooting tips.

## Troubleshooting

### RC832 Screen Shows Video, But Computer Doesn't
- Check video cable connection
- Try different input on capture card (if it has multiple)
- Capture card might be faulty

### RC832 Screen Is Blank/Snowy
- **VTX and receiver on different frequencies** (most common)
- VTX not powered or disabled in Betaflight
- VTX antenna disconnected
- Out of range (keep drone close, < 10m for initial test)

### Cannot Read Frames from Device 0
- This is what you're experiencing now
- Means capture card is connected but receiving no signal
- Follow steps 1-3 above to get VTX→RC832 connection working

## Physical Connections Checklist

```
BetaFPV Meteor 75 (VTX) 
    |
    | (5.8GHz wireless on matching frequency)
    |
    ↓
RC832 Receiver (powered with 5V)
    |
    | (RCA video cable)
    |
    ↓
USB Capture Card
    |
    | (USB cable)
    |
    ↓
MacBook (Device 0)
```

## Next Steps

Once video is working:
1. ✅ Phase 1: Video capture working
2. 🔄 Phase 2: Add person detection (using your SimpleCNN)
3. 🔄 Phase 3: Vision-to-control mapping
4. 🔄 Phase 4: CRSF protocol for ELRS control

## Files

- `autograd/drone_problems/video_capture.py` - Core video capture module
- `autograd/simulations/video_diagnostic.py` - Diagnostic tool
- `test_rc832.py` - Live feed viewer with stats
- `RC832_SETUP.md` - This reference guide

## Need Help?

**Most likely issue**: VTX frequency ≠ RC832 frequency

**Quick fix**: Try all Raceband channels (R1-R8) on RC832 using the CH button. When you hit the right channel, you'll see:
1. Signal strength indicator lights up on RC832
2. Video appears on RC832's built-in screen
3. Video appears on your computer

