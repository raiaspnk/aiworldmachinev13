from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='monster_core',
    version='1.0.0',
    description='Zero-Copy C++/CUDA Engine for AAA World-to-Mesh Pipeline',
    ext_modules=[
        CUDAExtension(
            name='monster_core',
            sources=['monster_core.cpp', 'monster_core_kernels.cu'],
            libraries=['gomp'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++20', '-march=native', '-fopenmp'],
                'nvcc': ['-O3', '--use_fast_math'],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)
