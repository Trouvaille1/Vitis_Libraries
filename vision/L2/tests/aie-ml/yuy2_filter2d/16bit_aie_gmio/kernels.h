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

#ifndef _KERNELS_H_
#define _KERNELS_H_

#include <adf/window/types.h>
#include <adf/stream/types.h>
#include "adf.h"
//#include "config.h"

#define SRS_SHIFT 10 // SRS shift used can be increased if input data likewise adjusted)

void yuy2_filter2D(adf::input_buffer<int16_t>& input, const int16_t (&coeff)[16], adf::output_buffer<int16_t>& output);

#endif
