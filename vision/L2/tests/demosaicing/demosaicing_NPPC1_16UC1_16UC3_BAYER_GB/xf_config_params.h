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

#define WIDTH 3840
#define HEIGHT 2160

#ifndef _XF_DEMOSIACING_CONFIG_H_
#define _XF_DEMOSIACING_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_demosaicing.hpp"

#define WIDTH 3840
#define HEIGHT 2160

#define XF_CV_DEPTH_IN_1 2
#define XF_CV_DEPTH_OUT_1 2

#define NPPCX XF_NPPC1

#define T_8U 0
#define T_16U 1

#define BPATTERN XF_BAYER_GB

#define XF_USE_URAM 0

#define IN_TYPE XF_16UC1
#define OUT_TYPE XF_16UC3

#define CV_IN_TYPE CV_16UC1
#define CV_OUT_TYPE CV_16UC3

#define INPUT_PTR_WIDTH 16
#define OUTPUT_PTR_WIDTH 64

#define ERROR_THRESHOLD 1

#endif
// _XF_DEMOSAICING_CONFIG_H_
