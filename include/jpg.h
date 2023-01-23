/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#ifndef __JPG_H__
#define __JPG_H__
#pragma once

#include <cstdlib>
#include <cstring>

extern "C" {
    #include <jpeglib.h>
}

#include "cujpg.h"


void JPG2BGR(cujpg::imgBuffer* src, cujpg::imgBuffer* dst)
{
    struct jpeg_error_mgr jerr;
	struct jpeg_decompress_struct cinfo;
	cinfo.err = jpeg_std_error(&jerr);
    //1创建解码对象并且初始化
	jpeg_create_decompress(&cinfo);
	//2.装备解码的数据
	//jpeg_stdio_src(&cinfo, infile);
	jpeg_mem_src(&cinfo, src->start, src->length);
    //3.获取jpeg图片文件的参数
	(void) jpeg_read_header(&cinfo, TRUE);
	/* Step 4: set parameters for decompression */
	//5.开始解码
	(void) jpeg_start_decompress(&cinfo);
    //6.申请存储一行数据的内存空间
	int row_stride = cinfo.output_width * cinfo.output_components;
    unsigned char *buffer = (unsigned char*)malloc(row_stride);
    dst->width = cinfo.output_width;
    dst->height = cinfo.output_height;
    dst->length = cinfo.output_width * cinfo.output_height * cinfo.output_components;
    dst->start = new uint8_t[dst->length];

	int i = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
		(void) jpeg_read_scanlines(&cinfo, &buffer, 1); 
		memcpy(dst->start + i * dst->width * 3, buffer, row_stride);
		i++;
	}
	//7.解码完成
	(void) jpeg_finish_decompress(&cinfo);
    //8.释放解码对象
	jpeg_destroy_decompress(&cinfo);
    free(buffer);
}

#endif // __JPG_H__
