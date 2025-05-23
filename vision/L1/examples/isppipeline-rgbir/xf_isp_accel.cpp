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

#include "xf_isp_types.h"
#include <iostream>
#include <fstream>
#include <strstream>

static bool flag = 0;

static uint32_t hist0_awb[3][HIST_SIZE] = {0};
static uint32_t hist1_awb[3][HIST_SIZE] = {0};

static int igain_0[3] = {0};
static int igain_1[3] = {0};

/************************************************************************************
 * Function:    AXIVideo2BayerMat
 * Parameters:  Multiple bayerWindow.getval AXI Stream, User Stream, Image Resolution
 * Return:      None
 * Description: Read data from multiple pixel/clk AXI stream into user defined stream
 ************************************************************************************/
template <int TYPE, int ROWS, int COLS, int NPPC, int XFCVDEPTH_IN>
void AXIVideo2BayerMat(InVideoStrm_t& bayer_strm, xf::cv::Mat<TYPE, ROWS, COLS, NPPC, XFCVDEPTH_IN>& bayer_mat) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    InVideoStrmBus_t axi;

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    int rows = bayer_mat.rows;
    int cols = bayer_mat.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    bool start = false;
    bool last = false;

loop_start_hunt:
    while (!start) {
// clang-format off
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount avg=0 max=0
        // clang-format on

        bayer_strm >> axi;
        start = axi.user.to_bool();
    }

loop_row_axi2mat:
    for (int i = 0; i < rows; i++) {
        last = false;
// clang-format off
#pragma HLS loop_tripcount avg=ROWS max=ROWS
    // clang-format on
    loop_col_zxi2mat:
        for (int j = 0; j < cols; j++) {
// clang-format off
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount avg=COLS/NPPC max=COLS/NPPC
            // clang-format on

            if (start || last) {
                start = false;
            } else {
                bayer_strm >> axi;
            }

            last = axi.last.to_bool();

            bayer_mat.write(idx++, axi.data(m_pix_width - 1, 0));
        }

    loop_last_hunt:
        while (!last) {
// clang-format off
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount avg=0 max=0
            // clang-format on

            bayer_strm >> axi;
            last = axi.last.to_bool();
        }
    }

    return;
}
template <int TYPE, int ROWS, int COLS, int NPPC>
void ColorMat2AXIvideo(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& color_mat, OutVideoStrm_t& color_strm) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    OutVideoStrmBus_t axi;

    int rows = color_mat.rows;
    int cols = color_mat.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    XF_TNAME(TYPE, NPPC) srcpixel;

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    int depth = XF_DTPIXELDEPTH(XF_LTM_T, NPPCX);

    bool sof = true; // Indicates start of frame

loop_row_mat2axi:
    for (int i = 0; i < rows; i++) {
// clang-format off
#pragma HLS loop_tripcount avg=ROWS max=ROWS
    // clang-format on
    loop_col_mat2axi:
        for (int j = 0; j < cols; j++) {
// clang-format off
#pragma HLS loop_flatten off
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount avg=COLS/NPPC max=COLS/NPPC
            // clang-format on
            if (sof) {
                axi.user = 1;
            } else {
                axi.user = 0;
            }

            if (j == cols - 1) {
                axi.last = 1;
            } else {
                axi.last = 0;
            }

            axi.data = 0;

            srcpixel = color_mat.read(idx++);

            for (int npc = 0; npc < NPPC; npc++) {
                for (int rs = 0; rs < 3; rs++) {
#if XF_AXI_GBR == 1
                    int kmap[3] = {1, 0, 2}; // GBR format
#else
                    int kmap[3] = {0, 1, 2}; // GBR format
#endif

                    int start = (rs + npc * 3) * depth;

                    int start_format = (kmap[rs] + npc * 3) * depth;

                    axi.data(start + (depth - 1), start) = srcpixel.range(start_format + (depth - 1), start_format);
                }
            }

            axi.keep = -1;
            color_strm << axi;

            sof = false;
        }
    }

    return;
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int XFCVDEPTH_IN, int XFCVDEPTH_OUT>
void fifo_copy(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& demosaic_out,
               xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& ltm_in,
               unsigned short height,
               unsigned short width) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    ap_uint<13> row, col;
    int readindex = 0, writeindex = 0;

    ap_uint<13> img_width = width >> XF_BITSHIFT(NPC);

Row_Loop:
    for (row = 0; row < height; row++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
#pragma HLS LOOP_FLATTEN off
    // clang-format on
    Col_Loop:
        for (col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline
            // clang-format on
            XF_TNAME(SRC_T, NPC) tmp_src;
            tmp_src = demosaic_out.read(readindex++);
            ltm_in.write(writeindex++, tmp_src);
        }
    }
}
template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int XFCVDEPTH_IN, int XFCVDEPTH_OUT>
void fifo_awb(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& demosaic_out,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& ltm_in,
              uint32_t hist0[3][HIST_SIZE],
              uint32_t hist1[3][HIST_SIZE],
              int gain0[3],
              int gain1[3],
              unsigned short height,
              unsigned short width,
              float thresh) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on	
	xf::cv::Mat<OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XFCVDEPTH_OUT> impop(height, width);
    uint32_t awb_config = (int)(thresh * 256); // thresh_awb int Q24_8 format change to Q16_16 format

	float inputMin = 0.0f;
    float inputMax = (1 << (XF_DTPIXELDEPTH(IN_TYPE, NPPCX))) - 1; // 65535.0f;
    float outputMin = 0.0f;
    float outputMax = (1 << (XF_DTPIXELDEPTH(IN_TYPE, NPPCX))) - 1; // 65535.0f;
	
	// clang-format off
#pragma HLS DATAFLOW
    // clang-format on
    if (WB_TYPE) {
        xf::cv::AWBhistogram<OUT_TYPE, OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_USE_URAM, WB_TYPE, HIST_SIZE,
                             XFCVDEPTH_IN, XFCVDEPTH_OUT>(demosaic_out, impop, hist0, awb_config, inputMin, inputMax,
                                                          outputMin, outputMax);
        xf::cv::AWBNormalization<OUT_TYPE, OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, WB_TYPE, HIST_SIZE, XFCVDEPTH_OUT,
                                 XFCVDEPTH_OUT>(impop, ltm_in, hist1, awb_config, inputMin, inputMax, outputMin,
                                                outputMax);
    } else {
        xf::cv::AWBChannelGain<OUT_TYPE, OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, 0, XFCVDEPTH_IN, XFCVDEPTH_OUT>(
            demosaic_out, impop, awb_config, gain0);
        xf::cv::AWBGainUpdate<OUT_TYPE, OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, 0, XFCVDEPTH_OUT, XFCVDEPTH_OUT>(
            impop, ltm_in, awb_config, gain1);
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int XFCVDEPTH_IN, int XFCVDEPTH_OUT>
void function_awb(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& demosaic_out,
                  xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& ltm_in,
                  uint32_t hist0[3][HIST_SIZE],
                  uint32_t hist1[3][HIST_SIZE],
                  int gain0[3],
                  int gain1[3],
                  unsigned short height,
                  unsigned short width,
                  unsigned char mode_reg,
                  float thresh) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    ap_uint<8> mode = (ap_uint<8>)mode_reg;
    ap_uint<1> mode_flg = mode.range(0, 0);

    if (mode_flg) {
        fifo_awb<OUT_TYPE, OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XFCVDEPTH_IN, XFCVDEPTH_OUT>(
            demosaic_out, ltm_in, hist0, hist1, gain0, gain1, height, width, thresh);
    } else {
        fifo_copy<OUT_TYPE, OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XFCVDEPTH_IN, XFCVDEPTH_OUT>(demosaic_out, ltm_in,
                                                                                               height, width);
    }
}

template <int TYPE, int ROWS, int COLS, int NPPC>
void ColorMat2AXIvideo_yuv(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& color_mat, OutVideoStrm_t& color_strm) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    OutVideoStrmBus_t axi;

    int rows = color_mat.rows;
    int cols = color_mat.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    XF_TNAME(TYPE, NPPC) srcpixel;

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    int depth = XF_DTPIXELDEPTH(TYPE, NPPCX);

    bool sof = true; // Indicates start of frame

loop_row_mat2axi:
    for (int i = 0; i < rows; i++) {
// clang-format off
#pragma HLS loop_tripcount avg=ROWS max=ROWS
    // clang-format on
    loop_col_mat2axi:
        for (int j = 0; j < cols; j++) {
// clang-format off
#pragma HLS loop_flatten off
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount avg=COLS/NPPC max=COLS/NPPC
            // clang-format on
            if (sof) {
                axi.user = 1;
            } else {
                axi.user = 0;
            }

            if (j == cols - 1) {
                axi.last = 1;
            } else {
                axi.last = 0;
            }

            axi.data = 0;

            srcpixel = color_mat.read(idx++);

            for (int npc = 0; npc < NPPC; npc++) {
                for (int rs = 0; rs < 1; rs++) {
                    int start = (rs + npc) * depth;

                    axi.data(start + (depth - 1), start) = srcpixel.range(start + (depth - 1), start);
                }
            }

            axi.keep = -1;
            color_strm << axi;

            sof = false;
        }
    }

    return;
}
void ISPpipeline(unsigned short height,
                 unsigned short width,
                 InVideoStrm_t& in_axis_video,
                 OutVideoStrm_t& out_axis_video,
                 // InVideoStrm_t& out_axis_video,
                 OutVideoStrmIR_t& ir_axis_video,
                 char R_IR_C1_wgts[25],
                 char R_IR_C2_wgts[25],
                 char B_at_R_wgts[25],
                 char IR_at_R_wgts[9],
                 char IR_at_B_wgts[9],
                 char sub_wgts[4],
                 uint32_t hist0[3][HIST_SIZE],
                 uint32_t hist1[3][HIST_SIZE],
                 int gain0[3],
                 int gain1[3],
                 uint16_t rgain,
                 uint16_t bgain,
                 unsigned char gamma_lut[256 * 3],
                 unsigned char mode_reg,
                 uint16_t pawb,
                 uint16_t ggain,
                 uint16_t bformat) {
#pragma HLS INLINE OFF

    xf::cv::Mat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_IN> imgInput(height, width);
    xf::cv::Mat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_IN_COPY1> imgInputCopy1(height, width);
    xf::cv::Mat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_6XWIDTH> imgInputCopy2(height, width);
    xf::cv::Mat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_FULLIR_OUT> fullir_out(height, width);
    xf::cv::Mat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_RRGB_OUT> rggb_out(height, width);
    xf::cv::Mat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_GAIN_OUT> gain_out(height, width);
    xf::cv::Mat<OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_DEMOSAIC_OUT> demosaic_out(height, width);
    xf::cv::Mat<OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_DEMOOUT_FINAL> demoOut_final(height, width);
    xf::cv::Mat<OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_LTM_IN> ltm_in(height, width);
    xf::cv::Mat<XF_LTM_T, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_DST> _dst(height, width);
    xf::cv::Mat<XF_LTM_T, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_AEC_IN> aecin(height, width);
    xf::cv::Mat<XF_16UC1, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_OUT> _imgOutput(height, width);

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    const int Q_VAL = 1 << (XF_DTPIXELDEPTH(IN_TYPE, NPPCX));

    float thresh = (float)pawb / 256;
    float inputMax = (1 << (XF_DTPIXELDEPTH(IN_TYPE, NPPCX))) - 1; // 65535.0f;

    float mul_fact = (inputMax / (inputMax - BLACK_LEVEL));
    AXIVideo2BayerMat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_IN>(in_axis_video, imgInput);

    xf::cv::rgbir2bayer<FILTERSIZE1, FILTERSIZE2, XF_BAYER_PATTERN, IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX,
                        XF_BORDER_CONSTANT, XF_USE_URAM, XF_CV_DEPTH_IN, XF_CV_DEPTH_RRGB_OUT, XF_CV_DEPTH_FULLIR_OUT,
                        XF_CV_DEPTH_3XWIDTH>(imgInput, R_IR_C1_wgts, R_IR_C2_wgts, B_at_R_wgts, IR_at_R_wgts,
                                             IR_at_B_wgts, sub_wgts, rggb_out, fullir_out);

    xf::cv::gaincontrol<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_RRGB_OUT, XF_CV_DEPTH_GAIN_OUT>(
        rggb_out, gain_out, rgain, bgain, ggain, bformat);

    xf::cv::demosaicing<IN_TYPE, OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, 0, XF_CV_DEPTH_GAIN_OUT,
                        XF_CV_DEPTH_DEMOSAIC_OUT>(gain_out, demosaic_out, bformat);

    function_awb<OUT_TYPE, OUT_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_DEMOSAIC_OUT, XF_CV_DEPTH_LTM_IN>(
        demosaic_out, ltm_in, hist0, hist1, gain0, gain1, height, width, mode_reg, thresh);

    if (OUT_TYPE == XF_8UC3) {
        fifo_copy<OUT_TYPE, XF_LTM_T, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_LTM_IN, XF_CV_DEPTH_AEC_IN>(
            ltm_in, aecin, height, width);
    } else {
        xf::cv::xf_QuatizationDithering<OUT_TYPE, XF_LTM_T, XF_HEIGHT, XF_WIDTH, 256, Q_VAL, NPPCX, XF_USE_URAM,
                                        XF_CV_DEPTH_LTM_IN, XF_CV_DEPTH_AEC_IN>(ltm_in, aecin);
    }
    xf::cv::gammacorrection<XF_LTM_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_AEC_IN, XF_CV_DEPTH_DST>(
        aecin, _dst, gamma_lut);
    xf::cv::rgb2yuyv<XF_LTM_T, XF_YUV_T, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_DST, XF_CV_DEPTH_OUT>(_dst,
                                                                                                       _imgOutput);

    xfMat2AXIvideo<AXI_WIDTH_OUT, XF_YUV_T, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_OUT>(_imgOutput, out_axis_video);
    xfMat2AXIvideo<AXI_WIDTH_IN, IN_TYPE, XF_HEIGHT, XF_WIDTH, NPPCX, XF_CV_DEPTH_FULLIR_OUT>(fullir_out,
                                                                                              ir_axis_video);
}

/*********************************************************************************
 * Function:    ISPPipeline_accel
 * Parameters:  Stream of input/output pixels, image resolution
 * Return:
 * Description:
 **********************************************************************************/

void ISPPipeline_accel(uint16_t height,
                       uint16_t width,
                       InVideoStrm_t& in_axis_video,
                       OutVideoStrm_t& out_axis_video,
                       // InVideoStrm_t& out_axis_video,
                       OutVideoStrmIR_t& ir_axis_video,
                       char R_IR_C1_wgts[25],
                       char R_IR_C2_wgts[25],
                       char B_at_R_wgts[25],
                       char IR_at_R_wgts[9],
                       char IR_at_B_wgts[9],
                       char sub_wgts[4],
                       uint16_t rgain,
                       uint16_t bgain,
                       unsigned char gamma_lut[256 * 3],
                       unsigned char mode_reg,
                       uint16_t pawb,
                       uint16_t ggain,
                       uint16_t bformat) {
// Create AXI Streaming Interfaces for the core
// clang-format off
#pragma HLS INTERFACE axis port=in_axis_video register
#pragma HLS INTERFACE axis port=out_axis_video register
#pragma HLS INTERFACE axis port=ir_axis_video register

#pragma HLS INTERFACE s_axilite port=R_IR_C1_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=R_IR_C2_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=B_at_R_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=IR_at_R_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=IR_at_B_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=sub_wgts bundle=CTRL

#pragma HLS INTERFACE s_axilite port=width bundle=CTRL
#pragma HLS INTERFACE s_axilite port=height bundle=CTRL
#pragma HLS INTERFACE s_axilite port=rgain bundle=CTRL
#pragma HLS INTERFACE s_axilite port=bgain bundle=CTRL
#pragma HLS INTERFACE s_axilite port=mode_reg bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pawb bundle=CTRL
#pragma HLS INTERFACE s_axilite port=gamma_lut bundle=CTRL
#pragma HLS INTERFACE s_axilite port=bformat bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ggain bundle=CTRL



#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

#pragma HLS ARRAY_PARTITION variable=hist0_awb complete dim=1
#pragma HLS ARRAY_PARTITION variable=hist1_awb complete dim=1
    // clang-format on

    if (!flag) {
        ISPpipeline(height, width, in_axis_video, out_axis_video, ir_axis_video, R_IR_C1_wgts, R_IR_C2_wgts,
                    B_at_R_wgts, IR_at_R_wgts, IR_at_B_wgts, sub_wgts, hist0_awb, hist1_awb, igain_0, igain_1, rgain,
                    bgain, gamma_lut, mode_reg, pawb, ggain, bformat);
        flag = 1;

    } else {
        ISPpipeline(height, width, in_axis_video, out_axis_video, ir_axis_video, R_IR_C1_wgts, R_IR_C2_wgts,
                    B_at_R_wgts, IR_at_R_wgts, IR_at_B_wgts, sub_wgts, hist1_awb, hist0_awb, igain_1, igain_0, rgain,
                    bgain, gamma_lut, mode_reg, pawb, ggain, bformat);
        flag = 0;
    }
}
