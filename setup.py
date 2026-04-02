from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# C++/CUDA extensions are already compiled via CMake
# setup.py is only used to register the Python package
setup(
    name="vae_morphosyntax",
    version="1.0.0",
    description="Unsupervised Morphological and Syntactic Structure Induction via VAE",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)

'''setup(
    name="vae_morphosyntax",
    version="1.0.0",
    description="Unsupervised Morphological and Syntactic Structure Induction via VAE",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    ext_modules=[
        CppExtension(
            name="fast_edit_distance",
            sources=["cpp_extensions/fast_edit_distance.cpp"],
        ),
        CppExtension(
            name="morpheme_segmenter",
            sources=["cpp_extensions/morpheme_segmenter.cpp"],
        ),
        CUDAExtension(
            name="attention_kernel",
            sources=["cpp_extensions/cuda_kernels/attention_kernel.cu"],
        ),
        CUDAExtension(
            name="elbo_kernel",
            sources=["cpp_extensions/cuda_kernels/elbo_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)'''