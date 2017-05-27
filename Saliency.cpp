#pragma once
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include "imgLib.h"
using namespace cv;
using namespace std;


#define theta 0.6f	//intial mark
#define alpha 0.05f	//control enhance
#define beta  0.2f	//control enhance
#define wBndCon 3.0f	//boundary connectivity weight


//Raw saliency computation via MDC
void SaliencyRaw(IplImage *pImgSrc, Mat_<float> &salImg)
{
	if(pImgSrc->nChannels != 3)
		return;

	int x, y;
	int W = pImgSrc->width;
	int H = pImgSrc->height;
	int W1 = W + 1;
	int H1 = H + 1;

#define MAP_SQR 0	//sum of sqr(ch0) + sqr(ch1) +sqr(ch2)
#define MAP_CH0 1	//sum of ch0
#define MAP_CH1 2	//sum of ch1
#define MAP_CH2 3	//sum of ch2
#define MAP_GROUP 4
#define MapType unsigned int
	MapType *IntMap = new MapType[W1 * H1 * MAP_GROUP];
	MapType *pMap;

	MapType Sum_C;
	MapType Sum_D0;
	MapType Sum_D1;
	MapType Sum_D2;
	unsigned char *pBuf;

	//top border
	memset(IntMap, 0x0, W1 * MAP_GROUP * sizeof(MapType));

	for(y = 1; y <= H1 - 1; y ++)
	{
		x = 0;
		pBuf = (unsigned char *)&pImgSrc->imageData[(y - 1) * pImgSrc->widthStep + x * pImgSrc->nChannels]; 
		Sum_C = 0;
		Sum_D0 = 0;
		Sum_D1 = 0;
		Sum_D2 = 0;

		pMap = &IntMap[y * W1 * MAP_GROUP];
		pMap[MAP_SQR] = 0;
		pMap[MAP_CH0] = 0;
		pMap[MAP_CH1] = 0;
		pMap[MAP_CH2] = 0;
		pMap += MAP_GROUP;

		for(x = 1; x <= W1 - 1; x ++)
		{
			Sum_C	+= pBuf[0] * pBuf[0] + pBuf[1] * pBuf[1] + pBuf[2] * pBuf[2];
			Sum_D0	+= pBuf[0];
			Sum_D1	+= pBuf[1];
			Sum_D2	+= pBuf[2];

			pMap[MAP_SQR] = pMap[MAP_SQR - W1 * MAP_GROUP] + Sum_C;
			pMap[MAP_CH0] = pMap[MAP_CH0 - W1 * MAP_GROUP] + Sum_D0;
			pMap[MAP_CH1] = pMap[MAP_CH1 - W1 * MAP_GROUP] + Sum_D1;
			pMap[MAP_CH2] = pMap[MAP_CH2 - W1 * MAP_GROUP] + Sum_D2;

			pBuf += pImgSrc->nChannels;
			pMap += MAP_GROUP;
		}

	}

	MapType cTbl0, cTbl1, cTbl2, cTbl3;
	//from all pixels
	for(y = 1; y <= H1 - 1; y ++)
	{
		x = 0;
		pBuf = (unsigned char *)&pImgSrc->imageData[(y - 1) * pImgSrc->widthStep + x * pImgSrc->nChannels]; 
		float* pSal = (float *)salImg.ptr(y - 1);
		for(x = 1; x <= W1 - 1; x ++)
		{
			MapType C = pBuf[0] * pBuf[0] + pBuf[1] * pBuf[1] + pBuf[2] * pBuf[2];
			MapType d0 = 2 * pBuf[0];
			MapType d1 = 2 * pBuf[1];
			MapType d2 = 2 * pBuf[2];

			//left top
			pMap = &IntMap[(y * W1 + x) * MAP_GROUP];
			cTbl0 = pMap[MAP_SQR] - pMap[MAP_CH0] * d0 - pMap[MAP_CH1] * d1 - pMap[MAP_CH2] * d2 + (x) * (y) * C;

			//right top
			pMap = &IntMap[(y * W1 + W1 - 1) * MAP_GROUP];
			cTbl1 = pMap[MAP_SQR] - pMap[MAP_CH0] * d0 - pMap[MAP_CH1] * d1 - pMap[MAP_CH2] * d2 + (W1 - 1) * (y) * C;

			//left bottom
			pMap = &IntMap[((H1 -1) * W1 + x) * MAP_GROUP];
			cTbl2 = pMap[MAP_SQR] - pMap[MAP_CH0] * d0 - pMap[MAP_CH1] * d1 - pMap[MAP_CH2] * d2 + (x) * (H1 - 1) * C;

			//right bottom
			pMap = &IntMap[((H1 -1) * W1 + (W1 -1)) * MAP_GROUP];
			cTbl3 = pMap[MAP_SQR] - pMap[MAP_CH0] * d0 - pMap[MAP_CH1] * d1 - pMap[MAP_CH2] * d2 + (W1 - 1) * (H1 - 1) * C;

			cTbl1 -= cTbl0;
			cTbl2 -= cTbl0;
			cTbl3 -= (cTbl0 + cTbl1 + cTbl2);

			*pSal = (min(min(cTbl0, cTbl1), min(cTbl2, cTbl3)));

			pSal ++;
			pBuf += 3;
		}
	}

	for(y = 0; y < H; y ++)
	{
		float* pSal = (float *)salImg.ptr(y);
		for(x = 0; x < W; x ++)
		{
			*pSal = sqrt(*pSal);
			pSal ++;
		}
	}
	normalize(salImg, salImg, 0, 1, NORM_MINMAX);

	delete [] IntMap;
}






#define N 24
int borderCntTbl[N*N*N];
int colorCntTbl[N*N*N];
float salSmoothTbl[N*N*N];

void SaliencySmooth(IplImage *pImgSrc, Mat_<float> &salImg)
{
	int r = 20;
	int x, y;
	int *pIdxMap = new int[pImgSrc->height * pImgSrc->width];


	memset(borderCntTbl, 0x0, sizeof(colorCntTbl));
	memset(colorCntTbl, 0x0, sizeof(colorCntTbl));
	memset(salSmoothTbl, 0x0, sizeof(salSmoothTbl));
	for(y = 0; y < pImgSrc->height; y ++)
	{
		unsigned char *pBuf = (unsigned char *)&pImgSrc->imageData[y * pImgSrc->widthStep];
		float *pSal = (float *)salImg.ptr(y);
		for(x = 0; x < pImgSrc->width; x ++)
		{
			int a = pBuf[0] * N / 256;
			int b = pBuf[1] * N / 256;
			int c = pBuf[2] * N / 256;
			int idx = a * N * N + b * N + c;
			pIdxMap[y * pImgSrc->width + x] = idx;
			colorCntTbl[idx] ++;
			salSmoothTbl[idx] += *pSal;

			if(y < r || x < r || x > pImgSrc->width - r)
				borderCntTbl[idx] ++;
			pBuf += 3;
			pSal ++;
		}
	}

	for(int i = 0; i < N * N * N; i ++)
	{
		if(colorCntTbl[i] > 0)
		{
			salSmoothTbl[i] /= colorCntTbl[i];
			float ratioBd = (float)borderCntTbl[i] / r / sqrt(colorCntTbl[i]);
			if(ratioBd > 0.01)
				salSmoothTbl[i] *= exp(-wBndCon * ratioBd);
		}
	}

	for(y = 0; y < pImgSrc->height; y ++)
	{
		float *pSal = (float *)salImg.ptr(y);
		for(x = 0; x < pImgSrc->width; x ++)
		{
			int idx = pIdxMap[y * pImgSrc->width + x];

			(*pSal) = ((*pSal)  + salSmoothTbl[idx]) / 2;
			pSal ++;
		}
	}

	delete pIdxMap;
	return;
}
#undef N

float OtsuThre(int nWidth, int nHeight, Mat_<float> &salImg)
{
	float r = 0.03f;
	int pixelCount[256], totalCount;  
    float pixelPro[256];  
    int i, j;
	int threshold = 0; 
 
//	normalize(data, data, 0, 1, NORM_MINMAX);
  
    for(i = 0; i < 256; i++)  
    {  
        pixelCount[i] = 0;  
        pixelPro[i] = 0;  
    }  
  
    //
	totalCount = 0;
    for(i = nHeight * r; i < nHeight * (1 - r); i++)
		for(j = nWidth * r; j < nWidth * (1 - r); j ++)
	    {  
			pixelCount[(int)(salImg.at<float>(i,j) * 255)] ++;  
			totalCount ++;
	    }  
  
    //
	float utmp = 0, usum = 0;
    for(i = 0; i < 256; i++)  
    {  
        pixelPro[i] = (float)pixelCount[i] / totalCount;  
		utmp += i * pixelPro[i];  
		usum += i * i * pixelPro[i];
    }  
  
    //
    float w0, w1, u0tmp, u0sum, u1sum, u1tmp, u0, u1, u,deltaTmp, deltaMax = 0;  
	w0 = u0tmp = u0sum = 0;
    for(i = 0; i < 256; i++)  
    {  
		w0 += pixelPro[i];
		u0tmp += i * pixelPro[i];  
		u0sum += i * i * pixelPro[i];

		w1 = 1 - w0;
		u1tmp = utmp - u0tmp;
		u1sum = usum - u0sum;

        u0 = u0tmp / w0;        //average gray of class 1
        u1 = u1tmp / w1;        //average gray of class 2
        u = u0tmp + u1tmp;      //average gray of image
        //cov
        deltaTmp = w0 * (u0 - u)*(u0 - u) + w1 * (u1 - u)*(u1 - u);
        if(deltaTmp > deltaMax)  
        {     
            deltaMax = deltaTmp;  
            threshold = i;  
        }  
    }

    return (float)threshold / 255;  
}  



void SaliencyEnhance(IplImage *pImgSrc, Mat_<float> &salImg)
{
	int i, j;
	float thTop, thBottom;
	float thres_val = OtsuThre(pImgSrc->width, pImgSrc->height, salImg);

	Mat markers(pImgSrc->height, pImgSrc->width, CV_8U);
	thTop = max(min(0.9f, thres_val * (1 + theta)), 0.3f);
	thBottom = min(max(0.1f, thres_val * (1 -  theta)), 0.3f);

	int cntFg = 0;
	for( i = 0; i < markers.rows; i++ )
	{
		uchar *pDataBg = markers.data + i * markers.step;
		float *pSal = (float *)salImg.ptr(i);
		for( j = 0; j < markers.cols; j++ )
		{
			if(*pSal > thTop)
			{
				*pDataBg = m_fg;
				cntFg ++;
			}
			else if(*pSal < thBottom)
				*pDataBg = m_bg;
			else
				*pDataBg = 0;
			pDataBg ++;
			pSal ++;
		}
	}
	int __bgErode = min((int)(sqrt(cntFg) * 0.05), 3);
	if(__bgErode > 0)
	{
		Mat element = getStructuringElement(MORPH_RECT, Size(1+__bgErode*2,1+__bgErode*2));
		erode(markers, markers, element);
		dilate(markers, markers, element);
	}


	Mat img(pImgSrc); 

	markerWatershed( &img, &markers);

	for( i = 0; i < markers.rows; i++ )
	{
		uchar *pDataMk = markers.data + i * markers.step;
		float *pSal = (float *)salImg.ptr(i);
		for( j = 0; j < markers.cols; j++ )
		{
			char index = *pDataMk;
			if(index == m_bg)
			{
				(*pSal) = (*pSal) * beta;
			}
			else if(index == m_fg)
			{
				(*pSal) = (1 - alpha) + (*pSal) * alpha;
			}

			pDataMk ++;
			pSal ++;
		}
	}

}

int SaliencyMDC(IplImage *pImgSrc, Mat_<float> &salImg)
{
	cvSmooth(pImgSrc, pImgSrc, CV_MEDIAN, 3);
	RGB2LAB(pImgSrc);
	SaliencyRaw(pImgSrc, salImg);
	SaliencySmooth(pImgSrc, salImg);
	SaliencyEnhance(pImgSrc, salImg);
	return 0;
}
