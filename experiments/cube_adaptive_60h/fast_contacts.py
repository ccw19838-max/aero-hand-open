"""Read-only C contact aggregation; all dynamics and monitors remain inherited."""
from pathlib import Path
import ctypes,fcntl,hashlib,os,subprocess
import numpy as np
import mujoco
from .env import AdaptiveEnv

_LIBRARY=None


def descriptor():
    source=Path(__file__).with_name('native_contacts.c')
    return dict(name='c_contacts_equivalent_v1',mujoco_version=mujoco.__version__,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compile_flags='-std=c99 -O2 -ffp-contract=off -shared -fPIC',
        changes='read-only contact aggregation; inherited physics, timestep, action, observation and monitor')


def library():
    global _LIBRARY
    if _LIBRARY is not None:return _LIBRARY
    source=Path(__file__).with_name('native_contacts.c');package=Path(mujoco.__file__).parent
    libmujoco=next(package.glob('libmujoco.so.*'))
    token=hashlib.sha256((source.read_text()+mujoco.__version__+'-O2-ffp-contract=off').encode()).hexdigest()[:20]
    cache=Path(__file__).parent/'artifacts/native_contact_cache'/token;cache.mkdir(parents=True,exist_ok=True)
    path=cache/'contacts.so'
    with (cache/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            temporary=cache/f'contacts_{os.getpid()}.so'
            subprocess.run(['cc','-std=c99','-O2','-ffp-contract=off','-shared','-fPIC',str(source),
                '-I',str(package/'include'),str(libmujoco),'-Wl,-rpath,'+str(package),'-o',str(temporary)],
                check=True,capture_output=True,text=True)
            os.replace(temporary,path)
    lib=ctypes.CDLL(str(path));ptr=ctypes.c_void_p
    lib.cube_forbidden.argtypes=[ptr,ptr,ctypes.c_int,ptr,ctypes.c_double];lib.cube_forbidden.restype=ctypes.c_int
    lib.cube_contacts.argtypes=[ptr,ptr,ctypes.c_int,ptr,ptr,ptr,ctypes.c_double,ptr];lib.cube_contacts.restype=None
    _LIBRARY=lib;return lib


class FastContactEnv(AdaptiveEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._contact_library=library();self._finger_map=np.ascontiguousarray(self.ids['geom_finger'],dtype=np.int32)
        self._palm_mask=np.ascontiguousarray(self.palm_geoms,dtype=np.uint8)
        self._forbidden_mask=np.ascontiguousarray(self.ids['geom_forbidden'],dtype=np.uint8)
        self._output=np.zeros(8,dtype=np.float64)
        self._model_address=self.model._address;self._data_address=self.data._address
        self._finger_pointer=self._finger_map.ctypes.data;self._palm_pointer=self._palm_mask.ctypes.data
        self._forbidden_pointer=self._forbidden_mask.ctypes.data;self._output_pointer=self._output.ctypes.data
    def _forbidden(self):
        return bool(self._contact_library.cube_forbidden(self._model_address,self._data_address,self.ids['cube_geom'],
            self._forbidden_pointer,self.config.contact_force))
    def _contacts(self):
        self._contact_library.cube_contacts(self._model_address,self._data_address,self.ids['cube_geom'],
            self._finger_pointer,self._palm_pointer,self._forbidden_pointer,self.config.contact_force,self._output_pointer)
        forces=self._output[:5].copy()
        return dict(finger_forces=forces,hand_force=float(sum(forces)+self._output[5]),
            max_penetration=float(self._output[6]),forbidden_support=bool(self._output[7]))
