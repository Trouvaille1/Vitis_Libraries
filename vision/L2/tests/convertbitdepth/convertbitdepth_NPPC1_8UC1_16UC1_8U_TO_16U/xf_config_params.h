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
#ifndef _XF_CONVERT_BITDEPTH_CONFIG_H_
#define _XF_CONVERT_BITDEPTH_CONFIG_H_

#include "ap_int.h"
#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "core/xf_convert_bitdepth.hpp"

// CONFIGURABLE
#define HEIGHT 2160
#define WIDTH 3840

#define XF_CV_DEPTH_IN_1 2
#define XF_CV_DEPTH_OUT_1 2

// APP SEPCIFIC
#define CONVERT_TYPE XF_CONVERT_8U_TO_16U
#define NPPCX XF_NPPC1

#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_16UC1

#define CV_IN_TYPE CV_8UC1
#define CV_OUT_TYPE CV_16UC1

#define IN_BYTE 1
#define OUT_BYTE 2

#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 16

#endif
// _XF_CONVERT_BITDEPTH_CONFIG_H_
