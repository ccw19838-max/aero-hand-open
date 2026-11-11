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
