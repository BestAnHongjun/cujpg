/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#include "cujpg.h"
#include "../cuda/cujpg_kernel.cuh"

#include <fstream>
#include <iostream>


cuJpgDecoder::cuJpgDecoder(const char* jpgFileName)
{
    std::ifstream stream(jpgFileName, std::ifstream::binary);
    if (!stream.good())
    {
        std::cerr << "Error! Can not open jpg file:" << jpgFileName << std::endl;
        exit(-1);
    }

    stream.seekg(0, std::ios::end);
    this->nDataLength = (int64_t)stream.tellg();
    stream.seekg(0, std::ios::beg);

    this->pJpgData = new uint8_t[this->nDataLength];
    stream.read((char*)this->pJpgData, this->nDataLength);

    init();
}

cuJpgDecoder::cuJpgDecoder(const uint8_t* pJpgData, int64_t nDataLength)
{
    this->pJpgData = new uint8_t[nDataLength];
    memcpy(this->pJpgData, pJpgData, nDataLength);
    init();
}

cuJpgDecoder::~cuJpgDecoder()
{
    delete[] this->pJpgData;
    cudaFree(pdQuantizationTables);
    if (this->resBuffer) delete[] this->resBuffer;
}

void cuJpgDecoder::init()
{
    memset(&oFrameHeader,0,sizeof(FrameHeader));
    memset(aQuantizationTables,0, 4 * sizeof(QuantizationTable));
    memset(aHuffmanTables,0, 4 * sizeof(HuffmanTable));
    cudaMalloc(&pdQuantizationTables, 64 * 4);
    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));
}

void cuJpgDecoder::decode(imgType type)
{
    int nPos = 0;
    uint8_t nMarker = nextMarker(this->pJpgData, nPos, this->nDataLength);

    // Check if this is a valid JPEG buffer
    if (nMarker != 0x0d8u)
    {
        std::cerr << "Invalid Jpeg Image" << std::endl;
        exit(-1);
    }

    nMarker = nextMarker(this->pJpgData, nPos, this->nDataLength);

    while (nMarker != -1)
    {
        if (nMarker == 0x0d9u)
        {   // EOF
            break;
        }
        
        if (nMarker == 0x0dbu)
        {   // DQT
            readQuantizationTables(this->pJpgData + nPos, this->aQuantizationTables);
        }

        if ((nMarker == 0x0c0u) | (nMarker == 0x0c2u))
        {   // SOF0
            if (nMarker == 0x0c2)
            {
                std::cerr << "CuJPG Library only support basline JPEG images!" << std::endl;
                exit(-1);
            }

            readFrameHeader(this->pJpgData + nPos, this->oFrameHeader);
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
            readHuffmanTables(this->pJpgData + nPos, aHuffmanTables);
        }

        if (nMarker == 0x0DA)
        {
            // Scan
            readScanHeader(this->pJpgData + nPos, oScanHeader);
            nPos += 6 + oScanHeader.nComponents * 2;

            int nAfterNextMarkerPos = nPos;
            int nAfterScanMarker = nextMarker(this->pJpgData, nAfterNextMarkerPos, this->nDataLength);

            if (nRestartInterval > 0)
            {
                while (nAfterScanMarker >= 0x0D0 && nAfterScanMarker <= 0x0D7)
                {
                    // This is a restart marker, go on
                    nAfterScanMarker = nextMarker(this->pJpgData, nAfterNextMarkerPos, this->nDataLength);
                }
            }

            NppiDecodeHuffmanSpec *apDecHuffmanDCTable[3];
            NppiDecodeHuffmanSpec *apDecHuffmanACTable[3];

            for (int i = 0; i < 3; ++i)
            {
                nppiDecodeHuffmanSpecInitAllocHost_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &apDecHuffmanDCTable[i]);
                nppiDecodeHuffmanSpecInitAllocHost_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &apDecHuffmanACTable[i]);
            }

            NPP_CHECK_NPP(nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R(this->pJpgData + nPos, nAfterNextMarkerPos - nPos - 2,
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

        nMarker = nextMarker(this->pJpgData, nPos, this->nDataLength);
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

    uint8_t* resBuffer_d;
    pwidth = aSrcSize[0].width;
    pheight = aSrcSize[0].height;
    
    size_t mPitch;
    size_t img_step = aSrcImageStep[0] * 3;
    if (!this->resBuffer)
        this->resBuffer = new uint8_t[pheight * img_step];
    NPP_CHECK_CUDA(cudaMallocPitch(&resBuffer_d, &mPitch, img_step, pheight));

    YCrCb2BGR(apSrcImage[0], apSrcImage[1], apSrcImage[2],
        pwidth, pheight, aSrcImageStep[0], aSrcImageStep[1], aSrcImageStep[2],
        resBuffer_d, nMCUBlocksV, nMCUBlocksH);

    NPP_CHECK_CUDA(cudaMemcpy(resBuffer, resBuffer_d, pheight * img_step, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 3; ++i)
    {
        cudaFree(apdDCT[i]);
        cudaFreeHost(aphDCT[i]);
        cudaFree(apSrcImage[i]);
    }
    cudaFree(resBuffer_d);

    cudaDeviceReset();
}

uint8_t* getBufferResult()
{

}

cv::Mat cuJpgDecoder::getMatResult()
{
    cv::Mat res = cv::Mat(pheight, pwidth, CV_8UC3, (void*)this->resBuffer, 4096 * 3);
    //cv::Mat res = cv::Mat(pheight, pwidth, CV_8UC1, (void*)this->resBuffer, 4096);
    return res;
}