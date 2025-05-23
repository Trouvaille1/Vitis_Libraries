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

#ifndef ADF_GRAPH_RESIZE_NORM_H
#define ADF_GRAPH_RESIZE_NORM_H

#include <adf.h>
#include <common/xf_aie_const.hpp>
#include "kernels.h"
#include "config.h"

using namespace adf;

/*
 * ADF graph to compute weighted moving average of
 * the last 8 samples in a stream of numbers
 */

class resizeNormGraph : public adf::graph {
   private:
    kernel k;

   public:
    input_plio in1;
    output_plio out1;
    port<input> scalex;
    port<input> scaley;
    port<input> coeff;

    resizeNormGraph() {
        k = kernel::create_object<ResizeNormRunner>();

        in1 = input_plio::create("DataIn0", adf::plio_128_bits, "data/input.txt");
        out1 = output_plio::create("DataOut0", adf::plio_128_bits, "data/output.txt");

        // create nets to connect kernels and IO ports
        connect<>(in1.out[0], k.in[0]);
        connect<>(k.out[0], out1.in[0]);
        connect<parameter>(scalex, async(k.in[1]));
        connect<parameter>(scaley, async(k.in[2]));
        connect<parameter>(coeff, async(k.in[3]));

        adf::dimensions(k.in[0]) = {TILE_WINDOW_SIZE_IN};
        adf::dimensions(k.out[0]) = {TILE_WINDOW_SIZE_OUT};

        // specify kernel sources
        source(k) = "xf_resize_normalize.cc";

        runtime<ratio>(k) = 1.0;
    }
};

#endif
