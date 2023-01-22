/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#include "cujpg.h"


void cuJpgDecoder::readQuantizationTables(const unsigned char *pData, QuantizationTable *pTables)
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

void cuJpgDecoder::readFrameHeader(const uint8_t* pData, FrameHeader& header)
{
    readAndAdvance<unsigned short>(pData);
    header.nSamplePrecision = readAndAdvance<unsigned char>(pData);
    header.nHeight = readAndAdvance<unsigned short>(pData);
    header.nWidth = readAndAdvance<unsigned short>(pData);
    header.nComponents = readAndAdvance<unsigned char>(pData);

    for (int c = 0; c<header.nComponents; ++c)
    {
        header.aComponentIdentifier[c] = readAndAdvance<unsigned char>(pData);
        header.aSamplingFactors[c] = readAndAdvance<unsigned char>(pData);
        header.aQuantizationTableSelector[c] = readAndAdvance<unsigned char>(pData);
    }
}

void cuJpgDecoder::readHuffmanTables(const unsigned char *pData, HuffmanTable *pTables)
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

void cuJpgDecoder::readScanHeader(const unsigned char *pData, ScanHeader &header)
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