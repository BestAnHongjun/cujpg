/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#include "cujpg.h"
#include "../cuda/cujpg_kernel.cuh"
using namespace cujpg;

#include <fstream>
#include <iostream>


cuJpgDecoder::cuJpgDecoder(memDevice dstDev)
{
    this->dstDev = dstDev;
    this->src_buffer = NULL;
    this->dst_buffer = NULL;
    this->tmp_buffer = NULL;

    memset(&oFrameHeader,0,sizeof(FrameHeader));
    memset(aQuantizationTables,0, 4 * sizeof(QuantizationTable));
    memset(aHuffmanTables,0, 4 * sizeof(HuffmanTable));
    CHECK(cudaMalloc(&pdQuantizationTables, 64 * 4));
    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));
}

cuJpgDecoder::~cuJpgDecoder()
{
    if (this->dstMallocTag)
    {
        if (this->dstDev == HOST) 
        {
            delete[] this->dst_buffer->start;
            delete this->dst_buffer;
        }
        else 
        {
            CHECK(cudaFree(this->dst_buffer->start));
            delete this->dst_buffer;
        }
    }

    if (!this->tmp_buffer) delete[] this->tmp_buffer;
    CHECK(cudaFree(pdQuantizationTables));
    NPP_CHECK_NPP(nppiDCTFree(pDCTState));
}

void cuJpgDecoder::setSrcBuffer(imgBuffer* src)
{
    this->src_buffer = src;
}

void cuJpgDecoder::decode(imgType type)
{
    int nPos = 0;
    uint8_t nMarker = nextMarker(src_buffer->start, nPos, src_buffer->length);

    // Check if this is a valid JPEG buffer
    if (nMarker != 0x0d8u)
    {
        std::cerr << "Invalid Jpeg Image" << std::endl;
        exit(-1);
    }

    nMarker = nextMarker(src_buffer->start, nPos, src_buffer->length);

    while (nMarker != -1)
    {
        if (nMarker == 0x0d9u)
        {   // EOF
            break;
        }
        
        if (nMarker == 0x0dbu)
        {   // DQT
            readQuantizationTables(src_buffer->start + nPos, aQuantizationTables);
        }

        if ((nMarker == 0x0c0u) | (nMarker == 0x0c2u))
        {   // SOF0
            if (nMarker == 0x0c2)
            {
                std::cerr << "CuJPG Library only support basline JPEG images!" << std::endl;
                exit(-1);
            }

            readFrameHeader(src_buffer->start + nPos, oFrameHeader);
            #if CUJPG_DEBUG_OUTPUT == 1
            std::cout << "Image Size: " << this->oFrameHeader.nWidth;
            std::cout << "x" << this->oFrameHeader.nHeight;
            std::cout << "x" << static_cast<int>(this->oFrameHeader.nComponents) << std::endl;
            #endif

            //Assert 3-Channel Image for this Sample
            if (oFrameHeader.nComponents != 3)
            {
                cerr << "The sample does only support color JPEG images" << endl;
                return exit(-1);
            }

            // Compute channel sizes as stored in the JPEG (8x8 blocks & MCU block layout)
            for (int i = 0; i < oFrameHeader.nComponents; ++ i)
            {
                this->nMCUBlocksV = max(this->nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f);
                this->nMCUBlocksH = max(this->nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4);
            }
            #if CUJPG_DEBUG_OUTPUT == 1
            std::cout << "nMCUBlocksV:" << this->nMCUBlocksH << std::endl;
            std::cout << "nMCUBlocksH:" << this->nMCUBlocksV << std::endl;
            #endif

            for (int i = 0; i < oFrameHeader.nComponents; ++ i)
            {
                NppiSize oBlocks;
                NppiSize oBlocksPerMCU = {oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};

				aSrcActualSize[i].width = DivUp(this->oFrameHeader.nWidth * oBlocksPerMCU.width, nMCUBlocksH);
				aSrcActualSize[i].height = DivUp(this->oFrameHeader.nHeight * oBlocksPerMCU.height, nMCUBlocksV);
				
                oBlocks.width = (int)ceil((oFrameHeader.nWidth + 7) / 8  *
                                          static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
                oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

                oBlocks.height = (int)ceil((oFrameHeader.nHeight + 7) / 8 *
                                           static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
                oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

                aSrcSize[i].width = oBlocks.width * 8;
                aSrcSize[i].height = oBlocks.height * 8;

                // Allocate Memory
                size_t nPitch;
                NPP_CHECK_CUDA(cudaMallocPitch(&apdDCT[i], &nPitch, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height));
                aDCTStep[i] = static_cast<Npp32s>(nPitch);

                NPP_CHECK_CUDA(cudaMallocPitch(&apSrcImage[i], &nPitch, aSrcSize[i].width, aSrcSize[i].height));
                aSrcImageStep[i] = static_cast<Npp32s>(nPitch);

                NPP_CHECK_CUDA(cudaHostAlloc(&aphDCT[i], aDCTStep[i] * oBlocks.height, cudaHostAllocDefault));
            }
        }

        if (nMarker == 0x0c4u)
        {   // Huffman Tables
            readHuffmanTables(src_buffer->start + nPos, aHuffmanTables);
        }

        if (nMarker == 0x0DA)
        {
            // Scan
            readScanHeader(src_buffer->start + nPos, oScanHeader);
            nPos += 6 + oScanHeader.nComponents * 2;

            int nAfterNextMarkerPos = nPos;
            int nAfterScanMarker = nextMarker(src_buffer->start, nAfterNextMarkerPos, src_buffer->length);

            if (nRestartInterval > 0)
            {
                while (nAfterScanMarker >= 0x0D0 && nAfterScanMarker <= 0x0D7)
                {
                    // This is a restart marker, go on
                    nAfterScanMarker = nextMarker(src_buffer->start, nAfterNextMarkerPos, src_buffer->length);
                }
            }

            NppiDecodeHuffmanSpec *apDecHuffmanDCTable[3];
            NppiDecodeHuffmanSpec *apDecHuffmanACTable[3];

            for (int i = 0; i < 3; ++i)
            {
                nppiDecodeHuffmanSpecInitAllocHost_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &apDecHuffmanDCTable[i]);
                nppiDecodeHuffmanSpecInitAllocHost_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &apDecHuffmanACTable[i]);
            }

            NPP_CHECK_NPP(nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R(src_buffer->start + nPos, nAfterNextMarkerPos - nPos - 2,
                                                                   nRestartInterval, oScanHeader.nSs, oScanHeader.nSe, 
                                                                   oScanHeader.nA >> 4, oScanHeader.nA & 0x0f,
                                                                   aphDCT, aDCTStep,
                                                                   apDecHuffmanDCTable,
                                                                   apDecHuffmanACTable,
                                                                   aSrcSize));

            for (int i = 0; i < 3; ++i)
            {
                nppiDecodeHuffmanSpecFreeHost_JPEG(apDecHuffmanDCTable[i]);
                nppiDecodeHuffmanSpecFreeHost_JPEG(apDecHuffmanACTable[i]);
            }
        }

        nMarker = nextMarker(src_buffer->start, nPos, src_buffer->length);
    }

    // Copy DCT coefficients and Quantization Tables from host to device 
    Npp8u aZigzag[] = {
            0,  1,  5,  6, 14, 15, 27, 28,
            2,  4,  7, 13, 16, 26, 29, 42,
            3,  8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
    };

    for (int i = 0; i < 4; ++i)
    {
        Npp8u temp[64];

        for(int k = 0 ; k < 32 ; ++ k)
        {
            temp[2 * k + 0] = aQuantizationTables[i].aTable[aZigzag[k +  0]];
            temp[2 * k + 1] = aQuantizationTables[i].aTable[aZigzag[k + 32]];
        }
        NPP_CHECK_CUDA(cudaMemcpyAsync((unsigned char *)pdQuantizationTables + i * 64, temp, 64, cudaMemcpyHostToDevice));          
    }
        

    for (int i = 0; i < 3; ++ i)
    {
        NPP_CHECK_CUDA(cudaMemcpyAsync(apdDCT[i], aphDCT[i], aDCTStep[i] * aSrcSize[i].height / 8, cudaMemcpyHostToDevice));
    }

    // Inverse DCT
    for (int i = 0; i < 3; ++ i)
    {
        NPP_CHECK_NPP(nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW(apdDCT[i], aDCTStep[i],
                                                              apSrcImage[i], aSrcImageStep[i],
                                                              pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
                                                              aSrcSize[i],
                                                              pDCTState));
    }

    #if CUJPG_DEBUG_OUTPUT == 1
    std::cout << "Y:" << aSrcSize[0].width << "x" << aSrcSize[0].height << std::endl;
    std::cout << "Cr:" << aSrcSize[1].width << "x" << aSrcSize[1].height << std::endl;
    std::cout << "Cb:" << aSrcSize[2].width << "x" << aSrcSize[2].height << std::endl;
    std::cout << "YStep:" << aSrcImageStep[0] << std::endl;
    std::cout << "CrStep:" << aSrcImageStep[1] << std::endl;
    std::cout << "CbStep:" << aSrcImageStep[2] << std::endl;
    #endif

    size_t mPitch;
    imgBuffer* res_buffer_d = new imgBuffer;
    int32_t pwidth = res_buffer_d->width = aSrcSize[0].width;
    int32_t pheight = res_buffer_d->height = aSrcSize[0].height;
    int32_t pdim = type == TYPE_GRAY ? 1 : 3;
    res_buffer_d->length = pwidth * pheight * pdim;
    NPP_CHECK_CUDA(cudaMallocPitch(&(res_buffer_d->start), &mPitch, pwidth * pdim, pheight));

    int (*kernel_func)(uint8_t*, uint8_t*, uint8_t*, int, int, int, int, int, uint8_t*, int, int);
    if (type == TYPE_RGB) kernel_func = YCrCb2RGB;
    else if (type == TYPE_BGR) kernel_func = YCrCb2BGR;
    else if (type == TYPE_GRAY) kernel_func = YCrCb2Gray;
    else
    {
        std::cerr << "Unsupport image type! at" << __FILE__ << "," << __LINE__ << std::endl;
        exit(-1);
    }
    kernel_func(apSrcImage[0], apSrcImage[1], apSrcImage[2],
        pwidth, pheight, aSrcImageStep[0], aSrcImageStep[1], aSrcImageStep[2],
        res_buffer_d->start, nMCUBlocksV, nMCUBlocksH);

    if (dstDev == HOST)
    {
        if (dstMallocTag) 
        {
            delete[] dst_buffer->start;
            delete dst_buffer;
        }
        dstMallocTag = true;
        dst_buffer = new imgBuffer;
        dst_buffer->start = new uint8_t[pwidth * pheight * pdim];
        dst_buffer->width = pwidth;
        dst_buffer->height = pheight;
        dst_buffer->length = pwidth * pheight * pdim;
        NPP_CHECK_CUDA(cudaMemcpy(dst_buffer->start, res_buffer_d->start, pwidth * pheight * pdim, cudaMemcpyDeviceToHost));
        CHECK(cudaFree(res_buffer_d));
    }
    else
    {
        if (dstMallocTag)
        {
            CHECK(cudaFree(dst_buffer->start));
            delete dst_buffer;
        }
        dstMallocTag = true;
        dst_buffer = res_buffer_d;
        dst_buffer->width = pwidth;
        dst_buffer->height = pheight;
        dst_buffer->length = pwidth * pheight * pdim;
    }

    for (int i = 0; i < 3; ++i)
    {
        cudaFree(apdDCT[i]);
        cudaFreeHost(aphDCT[i]);
        cudaFree(apSrcImage[i]);
    }
}

imgBuffer* cuJpgDecoder::getBufferResult()
{
    if (dst_buffer == NULL)
    {
        std::cerr << "You have not decode any JPEG! At" << __FILE__ << "," << __LINE__ << std::endl;
        exit(-1);
    }
    return dst_buffer;
}

cv::Mat cuJpgDecoder::getMatResult()
{
    if (dst_buffer == NULL)
    {
        std::cerr << "You have not decode any JPEG! At" << __FILE__ << "," << __LINE__ << std::endl;
        exit(-1);
    }
    cv::Mat res;
    int32_t pwidth = dst_buffer->width;
    int32_t pheight = dst_buffer->height;
    int32_t pdim = (dst_buffer->length) / (pwidth * pheight);
    int32_t type = pdim == 1 ? CV_8UC1 : CV_8UC3;
    if (dstDev == GPU)
    {
        if (tmp_buffer != NULL) delete[] tmp_buffer; 
        tmp_buffer = new uint8_t[dst_buffer->length];
        NPP_CHECK_CUDA(cudaMemcpy(tmp_buffer, dst_buffer->start, dst_buffer->length, cudaMemcpyDeviceToHost));
        res = cv::Mat(pheight, pwidth, type, (void*)tmp_buffer, pwidth * pdim);
    }
    else
    {
        res = cv::Mat(pheight, pwidth, type, (void*)dst_buffer->start);
    }
    return res;
}