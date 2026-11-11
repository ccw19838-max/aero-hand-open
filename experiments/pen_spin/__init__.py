"""CPU-first upright Aero thumb-around experiment, using simulator truth."""
import os
from pathlib import Path

# This experiment is explicitly CPU-only. Keep caches out of read-only home dirs.
os.environ['JAX_PLATFORMS'] = 'cpu'
_cache = Path(__file__).parent / 'artifacts' / 'cache'
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(_cache / 'jax'))
os.environ.setdefault('MPLCONFIGDIR', str(_cache / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(_cache))
os.environ.setdefault('MESA_SHADER_CACHE_DIR', str(_cache / 'mesa'))

# MuJoCo chooses its GL backend at import time. Automatic evaluation imports
# the model before record_video(), so select headless rendering here instead
# of relying on environment changes made after MuJoCo has already loaded.
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('EGL_PLATFORM', 'surfaceless')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
