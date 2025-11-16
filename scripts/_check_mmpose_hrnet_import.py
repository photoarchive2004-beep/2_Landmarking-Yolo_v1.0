import numpy, traceback

print("=== MMPose HRNet import check v2 ===")
print("numpy version:", numpy.__version__)
try:
    from mmpose.models.backbones import HRNet as MMPoseHRNet
    print("MMPoseHRNet imported OK:", MMPoseHRNet)
except Exception as e:
    print("IMPORT_ERROR:", repr(e))
    traceback.print_exc()
