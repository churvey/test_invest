// This should be in $HOME/.local/lib/python3.8/site-packages/torch/include/
#include <pybind11/pybind11.h>
#include "numpy_iterator.h"
#include <iostream>

// void Greet() { std::cout << "Hello Pytorch from C++" << std::endl; }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NumpyDictSampler>(m, "NumpyDictSampler")
        .def(py::init<py::object, int, std::vector<int>>())
        .def("__iter__", &NumpyDictSampler::__iter__);
    
    py::class_<NumpyDictSampler::Iterator>(m, "Iterator")
        .def("__next__", &NumpyDictSampler::Iterator::next);
    
    //  m.def("get_array_ptr", &get_array_ptr);
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("greet", &Greet); }
