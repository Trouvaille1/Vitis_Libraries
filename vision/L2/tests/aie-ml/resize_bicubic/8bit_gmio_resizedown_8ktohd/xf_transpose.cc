/*
 * Copyright 2021 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels.h"
#include "imgproc/xf_transpose_resize.hpp"

void transpose_api(adf::input_buffer<uint8_t>& input_metadata,
                   adf::input_buffer<uint8_t>& input_data,
                   adf::output_buffer<uint8_t>& output,
                   int outputStride) {
    xf::cv::aie::Transpose1D transpose = xf::cv::aie::Transpose1D();
    transpose.runImpl(input_metadata, input_data, output, outputStride, 4);
}
