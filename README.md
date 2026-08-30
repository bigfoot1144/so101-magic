```
SO-101 MJCF
   ↓
MuJoCo
   ↓
BAM STS3215 actuator model
   ↓
mjlab + MuJoCo-Warp
   ↓
thousands of parallel simulated SO-101s
   ↓
domain randomization
- friction
- voltage
- latency
- backlash
- joint offsets
- camera/target noise
   ↓
rsl_rl PPO
   ↓
small control policy
inputs:
- joint state
- target person ray
- previous action
outputs:
- joint target deltas
   ↓
Microduck-style ONNX export
- observation normalization embedded
   ↓
real perception stack
camera
   ↓
DINOv3
   ↓
person center / torso center
   ↓
camera calibration
   ↓
3D direction ray in robot frame
   ↓
ONNX PPO policy
   ↓
safety + target integrator
   ↓
rustypot
   ↓
real STS3215 servos
   ↓
SO-101
```

Training happens primarily on the **RTX 5090**. Deployment runs on the **Jetson Orin Nano**, ideally with DINOv3 in TensorRT and the tiny control policy in ONNX Runtime/TensorRT.

The key Microduck-style separation is:

```text
simulation physics → BAM
parallel training → mjlab/MJWarp
RL → rsl_rl PPO
deployment artifact → ONNX
real motor I/O → rustypot
```
