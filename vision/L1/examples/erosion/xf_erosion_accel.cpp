/*
 * Copyright (C) 2019-2022, Xilinx, Inc.
 * Copyright (C) 2022-2023, Advanced Micro Devices, Inc.
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

#include "xf_erosion_accel_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPPCX)) / 8) / (INPUT_PTR_WIDTH / 8);

void erosion_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                   unsigned char* process_shape,
                   ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                   int height,
                   int width) {
// clang-format off
     #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0  depth=__XF_DEPTH
     #pragma HLS INTERFACE m_axi      port=process_shape offset=slave  bundle=gmem1  depth=9
	 #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2  depth=__XF_DEPTH
     #pragma HLS INTERFACE s_axilite  port=return 			          bundle=control
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPPCX, XF_CV_DEPTH_IN_1> imgInput(height, width);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPPCX, XF_CV_DEPTH_OUT_1> imgOutput(height, width);

//     // Copy the shape data:
//     unsigned char _kernel[FILTER_SIZE * FILTER_SIZE];
//     for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; ++i) {
// // clang-format off
//         #pragma HLS PIPELINE
//         // clang-format on
//         _kernel[i] = process_shape[i];
//     }

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPPCX, XF_CV_DEPTH_IN_1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::erode<XF_BORDER_CONSTANT, IN_TYPE, HEIGHT, WIDTH, KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS, NPPCX,
                  XF_CV_DEPTH_IN_1, XF_CV_DEPTH_OUT_1>(imgInput, imgOutput, process_shape);

    // Convert imgOutput xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPPCX, XF_CV_DEPTH_OUT_1>(imgOutput, img_out);

    return;
} // End of kernel
