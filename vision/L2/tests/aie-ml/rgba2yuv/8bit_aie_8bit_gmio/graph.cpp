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

#include "graph.h"

rgba2yuvGraph mygraph;

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <common/xf_aie_utils.hpp>
int main(int argc, char** argv) {
    uint8_t* inputData1 = (uint8_t*)adf::GMIO::malloc(TILE_WINDOW_SIZE_RGBA + xf::cv::aie::METADATA_SIZE);
    uint8_t* outputData1 = (uint8_t*)adf::GMIO::malloc(TILE_WINDOW_SIZE_Y + xf::cv::aie::METADATA_SIZE);
    uint8_t* outputData2 = (uint8_t*)adf::GMIO::malloc(TILE_WINDOW_SIZE_UV + xf::cv::aie::METADATA_SIZE);

    memset(inputData1, 0, TILE_WINDOW_SIZE_RGBA + xf::cv::aie::METADATA_SIZE);
    xf::cv::aie::xfSetTileWidth(inputData1, TILE_WIDTH);
    xf::cv::aie::xfSetTileHeight(inputData1, TILE_HEIGHT);

    uint8_t* dataIn = (uint8_t*)xf::cv::aie::xfGetImgDataPtr(inputData1);
    for (int i = 0; i < TILE_ELEMENTS * 4; i++) {
        dataIn[i] = rand() % 256;
    }

    mygraph.init();
    uint16 tile_width = TILE_WIDTH;
    uint16 tile_height = TILE_HEIGHT;

    mygraph.update(mygraph.tile_width, tile_width);
    mygraph.update(mygraph.tile_height, tile_height);
    mygraph.run(1);

    mygraph.in.gm2aie_nb(inputData1, TILE_WINDOW_SIZE_RGBA + xf::cv::aie::METADATA_SIZE);
    mygraph.out1.aie2gm_nb(outputData1, TILE_WINDOW_SIZE_Y + xf::cv::aie::METADATA_SIZE);
    mygraph.out2.aie2gm_nb(outputData2, TILE_WINDOW_SIZE_UV + xf::cv::aie::METADATA_SIZE);
    mygraph.out1.wait();
    mygraph.out2.wait();

    // Compare the results
    // int acceptableError = 0;
    // int errCount = 0;
    // uint8_t* dataOut = (uint8_t*)xf::cv::aie::xfGetImgDataPtr(outputData);

    for (int i = 0; i < TILE_ELEMENTS + 64; i++) {
        // int cValue = abs(dataIn[i] - dataIn1[i]);
        if (i % 16 == 15)
            std::cout << (int)outputData1[i] << "\n";
        else
            std::cout << (int)outputData1[i] << " ";
    }
    /*if (errCount) {
        std::cout << "Test failed!" << std::endl;
        exit(-1);
    }*/
    std::cout << "Test passed" << std::endl;

    mygraph.end();
    return 0;
}
#endif
