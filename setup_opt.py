#!/usr/bin/env python
"""
Setup script for building optimized CUDA extensions
This script compiles the optimized deformable attention kernels
without modifying the original mmcv build.
"""

import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_extensions():
    """Build optimized CUDA extensions"""
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'mmcv', 'ops', 'csrc')
    
    # PyBind file (独立的绑定文件)
    main_source = os.path.join(extensions_dir, 'pytorch', 'pybind_opt.cpp')
    
    # C++ sources
    sources = [
        main_source,
        os.path.join(extensions_dir, 'pytorch', 'ms_deform_attn_opt.cpp'),
    ]
    
    # CUDA sources (设备注册在 .cu 文件末尾)
    cuda_sources = [
        os.path.join(extensions_dir, 'pytorch', 'cuda', 'ms_deform_attn_cuda_opt.cu'),
    ]
    
    sources += cuda_sources
    
    include_dirs = [
        os.path.join(this_dir, 'mmcv', 'ops', 'csrc'),
        os.path.join(this_dir, 'mmcv', 'ops', 'csrc', 'common'),
        os.path.join(this_dir, 'mmcv', 'ops', 'csrc', 'common', 'cuda'),
        os.path.join(this_dir, 'mmcv', 'ops', 'csrc', 'pytorch'),
    ]
    
    define_macros = []
    extra_compile_args = {
        'cxx': ['-g', '-O3'],
        'nvcc': [
            '-O3',
            '-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
            '--use_fast_math',
        ]
    }
    
    ext_modules = [
        CUDAExtension(
            name='mmcv._ext_opt',
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    
    return ext_modules


if __name__ == '__main__':
    setup(
        name='mmcv_deform_attn_opt',
        version='0.1.0',
        description='Optimized Multi-Scale Deformable Attention for mmcv',
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )
