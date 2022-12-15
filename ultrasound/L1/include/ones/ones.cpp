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

#include "kernels.hpp"

// void ones(output_window<float>* out){
//
//	aie::vector<float, SPACE_DIMENSION> res = aie::broadcast<float, SPACE_DIMENSION>(1);
//
//	for(unsigned i = 0; i < N_SAMPLES; i+=SPACE_DIMENSION){
//		window_writeincr(out, res);
//	}
//
// };

namespace us {
namespace L1 {

template <typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
void ones(output_window<T>* out) {
    aie::vector<T, VECDIM> res = aie::broadcast<float, VECDIM>(1);

    for (unsigned i = 0; i < LEN; i += INCREMENT) {
        window_writeincr(out, res);
    }
};

} // namespace L1
} // namespace us
