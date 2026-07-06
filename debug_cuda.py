import os
import glob
import ctypes

nvidia_base = r"C:\Users\PC\idiscovr-vr-characters\venv\Lib\site-packages\nvidia"

for subdir in glob.glob(os.path.join(nvidia_base, "*", "bin")):
    print("Adding:", subdir)
    os.add_dll_directory(subdir)

print("\nPATH:")
print(os.environ["PATH"])

print("\nTrying to load cuBLAS...")
ctypes.CDLL("cublas64_12.dll")
print("SUCCESS")