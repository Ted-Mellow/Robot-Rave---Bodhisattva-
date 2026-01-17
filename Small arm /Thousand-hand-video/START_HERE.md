# START HERE 🚀

## One Command to See Green Joint Overlays

```bash
./run_live.sh --live --end 10
```

This will:
1. Show video with **green dots** on body joints
2. Display robot simulation
3. Show real-time joint angles

## Important: Your conda environment has wrong MediaPipe

Your conda environment intercepts `python3` and uses MediaPipe 0.10.31 (incompatible). Use the `run_live.sh` wrapper which uses the system Python with MediaPipe 0.10.9.

## Why You Didn't See Green Dots

Green dots only appear in **live mode** (`--live` flag).

Your command:
```bash
python run_pipeline.py --end 10 --greyscale  # ❌ No green dots
```

Correct command:
```bash
./run_live.sh --live --end 10                  # ✅ Green dots!
```

## Quick Reference

| Command | What It Does | Green Dots? |
|---------|-------------|-------------|
| `./run_live.sh --live` | Live comparison mode | ✅ YES |
| `./run_live.sh --greyscale` | Batch processing | ❌ NO |

## Files to Read

1. **START_HERE.md** ← You are here
2. **README_SIMPLE.md** - Quick guide
3. **README.md** - Full documentation (if needed)

---

**Just run this:**
```bash
./run_live.sh --live --end 10
```

You'll see the green dots! 🟢

**Note:** The `run_live.sh` script uses the absolute path to system Python to bypass conda's environment.

## Technical Note: Greyscale Display

The video is now displayed in greyscale (with histogram equalization for better contrast), but pose detection runs on the **color** frames. This is because MediaPipe's neural network requires color information for accurate detection:

- **Color processing:** ~94% detection rate ✅
- **Greyscale processing:** ~0% detection rate ❌

You'll see:
- Greyscale video background (high contrast, easy to see)
- Colored pose skeleton overlay (MediaPipe's output)
- Green dots showing robot joint positions
- Colored text and overlays
