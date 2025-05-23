#
# Copyright (C) 2019-2022, Xilinx, Inc.
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ifeq (, $(shell which doxygen 2>/dev/null))
$(error "No doxygen in PATH.")
endif
ifeq (, $(shell which doxyrest 2>/dev/null))
$(error "No doxyrest in PATH.")
endif
ifeq (, $(shell which sphinx-build 2>/dev/null))
$(error "No sphinx-build in PATH.")
endif
ifeq (, $(HTML_DEST_DIR))
$(error "HTML_DEST_DIR is not set.")
endif

.PHONY: all install xml rst html clean cleanhtml cleanall

all: install

install: .stamp/sphinx
	rm -rf $(HTML_DEST_DIR)/*
	rsync -av build/html/ $(HTML_DEST_DIR)

.stamp/doxygen:
	rm -rf xml_L1 xml_L2 xml_L3
	doxygen Doxyfile_L1
	doxygen Doxyfile_L2
	doxygen Doxyfile_L3
	mkdir -p .stamp
	touch .stamp/doxygen

DOXYREST_FRAME = $(dir $(shell which doxyrest))/../doxyrest_frame

.stamp/doxyrest: .stamp/doxygen
	rm -rf src/rst_L1 src/rst_L2 src/rst_L3
	mkdir src/rst_L1 src/rst_L2 src/rst_L3
	doxyrest xml_L1/index.xml -o src/rst/index.rst \
		-F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock
	doxyrest xml_L2/index.xml -o src/rst_L2/index.rst \
		-F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock
	doxyrest xml_L3/index.xml -o src/rst_L3/index.rst \
		-F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock
	touch .stamp/doxyrest

.stamp/sphinx: .stamp/doxyrest Makefile.sphinx
	make -f Makefile.sphinx html
	touch .stamp/sphinx

xml:
	rm -f .stamp/doxygen
	make .stamp/doxygen

rst:
	rm -f .stamp/doxyrest
	make .stamp/doxyrest

html:
	rm -rf .stamp/sphinx build/doctrees
	make .stamp/sphinx

clean:
	rm -rf .stamp

cleanhtml:
	rm -rf .stamp/sphinx build/html build/doctrees

cleanall:
	rm -rf .stamp xml_L1 xml_L2 xml_L3 src/rst_L1 src/rst_L2 src/rst_L3 build/html build/doctrees
