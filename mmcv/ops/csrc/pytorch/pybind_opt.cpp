/*!
**************************************************************************************************
* Deformable DETR - Optimized Version Python Bindings
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Standalone Python bindings for optimized deformable attention operators
* This file is compiled separately to avoid modifying the original mmcv codebase
**************************************************************************************************
*/

#include <torch/extension.h>
#include "pytorch_cpp_helper.hpp"

// Forward declarations for optimized functions
Tensor ms_deform_attn_forward_opt(const Tensor &value, const Tensor &spatial_shapes,
                                  const Tensor &level_start_index,
                                  const Tensor &sampling_loc,
                                  const Tensor &attn_weight, const int im2col_step);

void ms_deform_attn_backward_opt(const Tensor &value, const Tensor &spatial_shapes,
                                 const Tensor &level_start_index,
                                 const Tensor &sampling_loc,
                                 const Tensor &attn_weight,
                                 const Tensor &grad_output, Tensor &grad_value,
                                 Tensor &grad_sampling_loc,
                                 Tensor &grad_attn_weight, const int im2col_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward_opt", &ms_deform_attn_forward_opt,
        "forward function of multi-scale deformable attention (optimized for FP32)",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("im2col_step"));
  
  m.def("ms_deform_attn_backward_opt", &ms_deform_attn_backward_opt,
        "backward function of multi-scale deformable attention (optimized for FP32)",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("grad_output"),
        py::arg("grad_value"), py::arg("grad_sampling_loc"),
        py::arg("grad_attn_weight"), py::arg("im2col_step"));
}
