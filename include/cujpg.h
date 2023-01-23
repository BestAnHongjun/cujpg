/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#ifndef __CUJPG_H__
#define __CUJPG_H__

#include <npp.h>
#include <cstdlib>
#include <unistd.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>

#include "cujpg_utils.h"
#include "cujpg_config.h"
#include "Exceptions.h"
#include "Endianess.h"
#include "helper_string.h"
#include "helper_cuda.h"

namespace cujpg
{

struct imgBuffer{
    uint8_t* start = NULL;
    uint32_t width;
    uint32_t height;
    uint64_t length;
};

typedef struct imgBuffer imgBuffer;


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

enum imgType
{
    TYPE_RGB     = 0,
    TYPE_BGR     = 1,
    TYPE_GRAY    = 2
};

enum memDevice
{
    HOST    = 0,
    GPU     = 1
};


class cuJpgDecoder
{
private:
    imgBuffer* src_buffer;
    imgBuffer* dst_buffer;
    memDevice dstDev;
    bool dstMallocTag = false;

public:
    cuJpgDecoder(memDevice dstDev);
    ~cuJpgDecoder();

public:
    void setSrcBuffer(imgBuffer* src);
    void decode(imgType type);
    imgBuffer* getBufferResult();
    cv::Mat getMatResult();


private:
    uint8_t* tmp_buffer;
    NppiSize aSrcActualSize[3];

    NppiSize aSrcSize[3];
    
    Npp32s aDCTStep[3];
    Npp16s *aphDCT[3] = {0,0,0};
    Npp16s *apdDCT[3] = {0,0,0};
    Npp8u *apSrcImage[3] = {0,0,0};
    Npp32s aSrcImageStep[3];
    
    HuffmanTable aHuffmanTables[4];
    HuffmanTable* pHuffmanDCTables = aHuffmanTables;
    HuffmanTable *pHuffmanACTables = &aHuffmanTables[2];

    FrameHeader oFrameHeader;
    int nMCUBlocksV = 0, nMCUBlocksH = 0;

    NppiDCTState *pDCTState;
    QuantizationTable aQuantizationTables[4];
    Npp8u *pdQuantizationTables;

    ScanHeader oScanHeader;
    int nRestartInterval = -1;

private:
    static uint8_t nextMarker(const uint8_t* pData, int &nPos, int nLength);
    static void readQuantizationTables(const unsigned char *pData, QuantizationTable *pTables);
    static void readFrameHeader(const uint8_t* pData, FrameHeader& header);
    static void readHuffmanTables(const unsigned char *pData, HuffmanTable *pTables);
    static void readScanHeader(const unsigned char *pData, ScanHeader &header);
};

} // namespace cujpg

#endif // __CUJPG_H__
