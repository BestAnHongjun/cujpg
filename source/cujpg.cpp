/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-20
*/

#include <npp.h>
#include <cuda_runtime.h>

#include <math.h>
#include <cmath>
#include <string.h>
#include <fstream>
#include <iostream>
#include <cstdio>

#include "Exceptions.h"
#include "Endianess.h"
#include "helper_string.h"
#include "helper_cuda.h"

#include "cuda/cujpg_kernel.cuh"


struct FrameHeader
{
    unsigned char nSamplePrecision;
    unsigned short nHeight;
    unsigned short nWidth;
    unsigned char nComponents;
    unsigned char aComponentIdentifier[3];
    unsigned char aSamplingFactors[3];
    unsigned char aQuantizationTableSelector[3];
};

struct ScanHeader
{
    unsigned char nComponents;
    unsigned char aComponentSelector[3];
    unsigned char aHuffmanTablesSelector[3];
    unsigned char nSs;
    unsigned char nSe;
    unsigned char nA;
};

struct QuantizationTable
{
    unsigned char nPrecisionAndIdentifier;
    unsigned char aTable[64];
};

struct HuffmanTable
{
    unsigned char nClassAndIdentifier;
    unsigned char aCodes[16];
    unsigned char aTable[256];
};

enum teComponentSampling
{
	YCbCr_444,
	YCbCr_440,
	YCbCr_422,
	YCbCr_420,
	YCbCr_411,
	YCbCr_410,
	YCbCr_UNKNOWN
};


int DivUp(int x, int d)
{
    return (x + d - 1) / d;
}


int nextMarker(const unsigned char *pData, int &nPos, int nLength)
{
    unsigned char c = pData[nPos++];

    do
    {
        while (c != 0xffu && nPos < nLength)
        {
            c =  pData[nPos++];
        }

        if (nPos >= nLength)
            return -1;

        c =  pData[nPos++];
    }
    while (c == 0 || c == 0x0ffu);

    return c;
}


template<typename T>
T readAndAdvance(const unsigned char *&pData)
{
    T nElement = readBigEndian<T>(pData);
    pData += sizeof(T);
    return nElement;
}


void readRestartInterval(const unsigned char *pData, int &nRestartInterval)
{
    readAndAdvance<unsigned short>(pData);
    nRestartInterval = readAndAdvance<unsigned short>(pData);
}


void readFrameHeader(const unsigned char *pData, FrameHeader &header)
{
    readAndAdvance<unsigned short>(pData);
    header.nSamplePrecision = readAndAdvance<unsigned char>(pData);
    header.nHeight = readAndAdvance<unsigned short>(pData);
    header.nWidth = readAndAdvance<unsigned short>(pData);
    header.nComponents = readAndAdvance<unsigned char>(pData);

    for (int c=0; c<header.nComponents; ++c)
    {
        header.aComponentIdentifier[c] = readAndAdvance<unsigned char>(pData);
        header.aSamplingFactors[c] = readAndAdvance<unsigned char>(pData);
        header.aQuantizationTableSelector[c] = readAndAdvance<unsigned char>(pData);
    }

}


void readQuantizationTables(const unsigned char *pData, QuantizationTable *pTables)
{
    unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

    while (nLength > 0)
    {
        unsigned char nPrecisionAndIdentifier = readAndAdvance<unsigned char>(pData);
        int nIdentifier = nPrecisionAndIdentifier & 0x0f;

        pTables[nIdentifier].nPrecisionAndIdentifier = nPrecisionAndIdentifier;
        memcpy(pTables[nIdentifier].aTable, pData, 64);
        pData += 64;

        nLength -= 65;
    }
}


void readHuffmanTables(const unsigned char *pData, HuffmanTable *pTables)
{
    unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

    while (nLength > 0)
    {
        unsigned char nClassAndIdentifier = readAndAdvance<unsigned char>(pData);
        int nClass = nClassAndIdentifier >> 4; // AC or DC
        int nIdentifier = nClassAndIdentifier & 0x0f;
        int nIdx = nClass * 2 + nIdentifier;
        pTables[nIdx].nClassAndIdentifier = nClassAndIdentifier;

        // Number of Codes for Bit Lengths [1..16]
        int nCodeCount = 0;

        for (int i = 0; i < 16; ++i)
        {
            pTables[nIdx].aCodes[i] = readAndAdvance<unsigned char>(pData);
            nCodeCount += pTables[nIdx].aCodes[i];
        }

        memcpy(pTables[nIdx].aTable, pData, nCodeCount);
        pData += nCodeCount;

        nLength -= (17 + nCodeCount);
    }
}


void readScanHeader(const unsigned char *pData, ScanHeader &header)
{
    readAndAdvance<unsigned short>(pData);

    header.nComponents = readAndAdvance<unsigned char>(pData);

    for (int c=0; c<header.nComponents; ++c)
    {
        header.aComponentSelector[c] = readAndAdvance<unsigned char>(pData);
        header.aHuffmanTablesSelector[c] = readAndAdvance<unsigned char>(pData);
    }

    header.nSs = readAndAdvance<unsigned char>(pData);
    header.nSe = readAndAdvance<unsigned char>(pData);
    header.nA = readAndAdvance<unsigned char>(pData);
}


void jpgToYCrCb(unsigned char* pJpegData, int nInputLength, 
    unsigned char* Y_d, unsigned char* &Cr_d, unsigned char* &Cb_d,
    int &pwidth, int &pheight, int &YStep, int &CrStep, int &CbStep,
    int &nMCUBlocksV, int &nMCUBlocksH)
{
    NppiDCTState *pDCTState;
    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));
    
    // Check if this is a valid JPEG buffer
    int nPos = 0;
    int nMarker = nextMarker(pJpegData, nPos, nInputLength);

    if (nMarker != 0x0D8)
    {
        cerr << "Invalid Jpeg Image" << endl;
        return exit(-1);
    }

    nMarker = nextMarker(pJpegData, nPos, nInputLength);

    // Parsing and Huffman Decoding (on host)
    FrameHeader oFrameHeader;
    QuantizationTable aQuantizationTables[4];
    Npp8u *pdQuantizationTables;
    cudaMalloc(&pdQuantizationTables, 64 * 4);

    HuffmanTable aHuffmanTables[4];
    HuffmanTable *pHuffmanDCTables = aHuffmanTables;
    HuffmanTable *pHuffmanACTables = &aHuffmanTables[2];
    ScanHeader oScanHeader;
    memset(&oFrameHeader,0,sizeof(FrameHeader));
    memset(aQuantizationTables,0, 4 * sizeof(QuantizationTable));
    memset(aHuffmanTables,0, 4 * sizeof(HuffmanTable));
	NppiSize aSrcActualSize[3];
	teComponentSampling eComponentSampling = YCbCr_UNKNOWN;
	
    int nRestartInterval = -1;

    NppiSize aSrcSize[3];
    Npp16s *aphDCT[3] = {0,0,0};
    Npp16s *apdDCT[3] = {0,0,0};
    Npp32s aDCTStep[3];

    Npp8u *apSrcImage[3] = {0,0,0};
    Npp32s aSrcImageStep[3];

    while (nMarker != -1)
    {
        if (nMarker == 0x0D8)
        {
            // Embedded Thumbnail, skip it
            int nNextMarker = nextMarker(pJpegData, nPos, nInputLength);

            while (nNextMarker != -1 && nNextMarker != 0x0D9)
            {
                nNextMarker = nextMarker(pJpegData, nPos, nInputLength);
            }
        }

        if (nMarker == 0x0DD)
        {
            readRestartInterval(pJpegData + nPos, nRestartInterval);
        }

        if ((nMarker == 0x0C0) | (nMarker == 0x0C2))
        {
            //Assert Baseline for this Sample
            //Note: NPP does support progressive jpegs for both encode and decode
            if (nMarker != 0x0C0)
            {
                cerr << "The sample does only support baseline JPEG images" << endl;
                return exit(-1);
            }

            // Baseline or Progressive Frame Header
            readFrameHeader(pJpegData + nPos, oFrameHeader);
            cout << "Image Size: " << oFrameHeader.nWidth << "x" << oFrameHeader.nHeight << "x" << static_cast<int>(oFrameHeader.nComponents) << endl;

            //Assert 3-Channel Image for this Sample
            if (oFrameHeader.nComponents != 3)
            {
                cerr << "The sample does only support color JPEG images" << endl;
                return exit(-1);
            }

            // Compute channel sizes as stored in the JPEG (8x8 blocks & MCU block layout)
            for (int i = 0; i < oFrameHeader.nComponents; ++ i)
            {
                nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f );
                nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4 );
            }

            printf("2:%d %d\n", nMCUBlocksV, nMCUBlocksH);

            for (int i = 0; i < oFrameHeader.nComponents; ++ i)
            {
                NppiSize oBlocks;
                // NppiSize oBlocksPerMCU = {oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};
                NppiSize oBlocksPerMCU = {oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};

				aSrcActualSize[i].width = DivUp(oFrameHeader.nWidth * oBlocksPerMCU.width, nMCUBlocksH);
				aSrcActualSize[i].height = DivUp(oFrameHeader.nHeight * oBlocksPerMCU.height, nMCUBlocksV);
				
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

        if (nMarker == 0x0DB)
        {
            // Quantization Tables
            readQuantizationTables(pJpegData + nPos, aQuantizationTables);
        }

        if (nMarker == 0x0C4)
        {
            // Huffman Tables
            readHuffmanTables(pJpegData + nPos, aHuffmanTables);
        }

        if (nMarker == 0x0DA)
        {
            // Scan
            readScanHeader(pJpegData + nPos, oScanHeader);
            nPos += 6 + oScanHeader.nComponents * 2;

            int nAfterNextMarkerPos = nPos;
            int nAfterScanMarker = nextMarker(pJpegData, nAfterNextMarkerPos, nInputLength);

            if (nRestartInterval > 0)
            {
                while (nAfterScanMarker >= 0x0D0 && nAfterScanMarker <= 0x0D7)
                {
                    // This is a restart marker, go on
                    nAfterScanMarker = nextMarker(pJpegData, nAfterNextMarkerPos, nInputLength);
                }
            }

            NppiDecodeHuffmanSpec *apDecHuffmanDCTable[3];
            NppiDecodeHuffmanSpec *apDecHuffmanACTable[3];

            for (int i = 0; i < 3; ++i)
            {
                nppiDecodeHuffmanSpecInitAllocHost_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &apDecHuffmanDCTable[i]);
                nppiDecodeHuffmanSpecInitAllocHost_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &apDecHuffmanACTable[i]);
            }

            NPP_CHECK_NPP(nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R(pJpegData + nPos, nAfterNextMarkerPos - nPos - 2,
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

        nMarker = nextMarker(pJpegData, nPos, nInputLength);
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

    Y_d = apSrcImage[0];
    Cr_d = apSrcImage[1];
    Cb_d = apSrcImage[2];
    YStep = aSrcImageStep[0];
    CrStep = aSrcImageStep[1];
    CbStep = aSrcImageStep[2];

    printf("1:%d %d %d %d %d %d\n", aSrcSize[0].width, aSrcSize[0].height,aSrcSize[1].width, aSrcSize[1].height,aSrcSize[2].width, aSrcSize[2].height);

    pwidth = aSrcSize[0].width;
    pheight = aSrcSize[0].height;

    delete [] pJpegData;
    cudaFree(pdQuantizationTables);
    nppiDCTFree(pDCTState);
    for (int i = 0; i < 3; ++i)
    {
        cudaFree(apdDCT[i]);
        cudaFreeHost(aphDCT[i]);
        // cudaFree(apSrcImage[i]);
    }

    return;
}

unsigned char* jpgToRgb(unsigned char* pJpegData, int nInputLength, int &pwidth, int &pheight)
{
    Npp8u *Y_d, *Cr_d, *Cb_d;
    Npp32s YStep, CrStep, CbStep;
    int nMCUBlocksV, nMCUBlocksH;

    jpgToYCrCb(pJpegData, nInputLength, Y_d, Cr_d, Cb_d, pwidth, pheight, YStep, CrStep, CbStep, nMCUBlocksV, nMCUBlocksH);

    Npp8u *rgb_h;
    // Npp8u *rgb_d;
    // size_t mPitch = pwidth * pheight * 3;

    // NPP_CHECK_CUDA(cudaMalloc(&rgb_d, mPitch));

    // YCrCb2RGB(Y_d, Cr_d, Cb_d, pwidth, pheight, YStep, CrStep, CbStep, rgb_d, nMCUBlocksV, nMCUBlocksH);

    rgb_h = (unsigned char*)malloc(pwidth * pheight * 3);
    printf("%d, %d\n", pwidth, pheight);
    // NPP_CHECK_CUDA(cudaHostAlloc(&rgb_h, pwidth*pheight * 3, cudaHostAllocDefault));
    // NPP_CHECK_CUDA(cudaMemcpy(rgb_h, rgb_d, pwidth*pheight * 3, cudaMemcpyDeviceToHost));

    // memcpy(res, rgb_h, pwidth * pheight * 3);

    cudaFree(Y_d);
    cudaFree(Cr_d);
    cudaFree(Cb_d);
    // cudaFree(rgb_d);
    // cudaFreeHost(rgb_h);

    cudaDeviceReset();

    return rgb_h;
}