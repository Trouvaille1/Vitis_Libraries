/*
 * Copyright 2022 Xilinx, Inc.
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
#ifndef _XF_CUSTOM_CONV_CONFIG_H_
#define _XF_CUSTOM_CONV_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_custom_convolution.hpp"

#define HEIGHT 2160
#define WIDTH 3840

#define XF_CV_DEPTH_IN_1 2
#define XF_CV_DEPTH_OUT_1 2

#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3
#define SHIFT 15

/*  specify the shift parameter */

#define OUT_8U 0
#define OUT_16S 1

#define GRAY 0
#define RGB 1

#define NPPCX XF_NPPC8

#define IN_TYPE XF_8UC3
#define OUT_TYPE XF_16SC3

#define CV_IN_TYPE CV_8UC3
#define CV_OUT_TYPE CV_16SC3

#define INPUT_PTR_WIDTH 256
#define OUTPUT_PTR_WIDTH 512

#endif
// end of _XF_CUSTOM_CONV_CONFIG_H_
