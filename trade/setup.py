from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="trade_cpp",
    ext_modules=[CppExtension("trade_cpp", ["cpp/pybind_torch.cpp"], extra_compile_args={'cxx': ['-g']})],
    cmdclass={"build_ext": BuildExtension},
)
