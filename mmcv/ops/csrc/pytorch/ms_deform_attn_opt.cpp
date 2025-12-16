/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "pytorch_cpp_helper.hpp"

// Forward declarations for CUDA implementations
Tensor ms_deform_attn_cuda_forward_opt(const Tensor &value,
                                       const Tensor &spatial_shapes,
                                       const Tensor &level_start_index,
                                       const Tensor &sampling_loc,
                                       const Tensor &attn_weight,
                                       const int im2col_step);

void ms_deform_attn_cuda_backward_opt(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight,
    const int im2col_step);

// C++ wrappers that call CUDA implementations
Tensor ms_deform_attn_forward_opt(const Tensor &value, const Tensor &spatial_shapes,
                                  const Tensor &level_start_index,
                                  const Tensor &sampling_loc,
                                  const Tensor &attn_weight,
                                  const int im2col_step) {
  at::DeviceGuard guard(value.device());
  return ms_deform_attn_cuda_forward_opt(value, spatial_shapes, level_start_index,
                                         sampling_loc, attn_weight, im2col_step);
}

void ms_deform_attn_backward_opt(const Tensor &value, const Tensor &spatial_shapes,
                                 const Tensor &level_start_index,
                                 const Tensor &sampling_loc,
                                 const Tensor &attn_weight,
                                 const Tensor &grad_output, Tensor &grad_value,
                                 Tensor &grad_sampling_loc,
                                 Tensor &grad_attn_weight, const int im2col_step) {
  at::DeviceGuard guard(value.device());
  ms_deform_attn_cuda_backward_opt(value, spatial_shapes, level_start_index,
                                   sampling_loc, attn_weight, grad_output,
                                   grad_value, grad_sampling_loc, grad_attn_weight,
                                   im2col_step);
}
