
#pragma once
#include <cv.h>
#include <highgui.h>
#include <io.h>
#include "imgLib.h"
using namespace cv;



extern int SaliencyMDC(IplImage *pImgSrc, Mat_<float> &salImg);
int tAll = 0;


int SaliencyDetection(char *inputFileName, char *salFileName)
{
	int64 t1, t2;

	IplImage *pImgSrc_full = cvLoadImage(inputFileName);
	if(pImgSrc_full == NULL)
	{
		printf("Input file %s open failed\n", inputFileName);
		return -1;
	}

	t1 = cvGetTickCount();

	double ratio = (double)300 / max(pImgSrc_full->width, pImgSrc_full->height);
	if(ratio > 1)
		ratio = 1;
	int w = ((int)(ratio * pImgSrc_full->width + 3)) / 4 * 4;
	int h = ratio * pImgSrc_full->height;

	IplImage *pImgSrc = cvCreateImage(cvSize(w, h), 8, 3);

	ImgResize(pImgSrc_full, pImgSrc);

	Mat_<float> salImg(pImgSrc->height, pImgSrc->width);
	SaliencyMDC(pImgSrc, salImg);

	t2 = cvGetTickCount();


	Mat_<float> salImg_full;
	resize(salImg, salImg_full, cvSize(pImgSrc_full->width, pImgSrc_full->height));

	tAll += (t2 - t1);

	char buf[256];
	sprintf(buf, "%s_Ours.png", salFileName);
	imwrite( buf, salImg_full * 255);


	cvReleaseImage(&pImgSrc);
	cvReleaseImage(&pImgSrc_full);
	return 0;
}


int main( int argc, char * argv[] ) 
{
	if (argc < 2) {
		printf("Usage: %s image\n", argv[0] );
		printf("Usage: %s dirName\n", argv[0] );
		return -1;
	}

	if(strstr(argv[1], ".jpg"))
	{
		printf("Input image %s:\n", argv[1]);
		SaliencyDetection(argv[1], "sal");
		return 0;
	}
	
	printf("Input directory %s:\n", argv[1]);

	long handle;
	struct _finddata_t filestruct;

	int nFileNum = 0;
	char* szDirName = argv[1];
	char path_search[_MAX_PATH];
	sprintf(path_search, "%s\\*.jpg", szDirName);

	handle = _findfirst(path_search, &filestruct);
	while(handle != -1)
	{
		char inputFileName[256];
		char salFileName[256];
		printf("Processing image %d: %-70s\r", nFileNum + 1, filestruct.name);
		sprintf(inputFileName, "%s\\%s", szDirName, filestruct.name);
		strcpy(salFileName, inputFileName);
		while(salFileName[strlen(salFileName) - 1] != '.')
			salFileName[strlen(salFileName) - 1] = 0;
		salFileName[strlen(salFileName) - 1] = 0;
		SaliencyDetection(inputFileName, salFileName);

		nFileNum ++;
		if(_findnext(handle, &filestruct) == -1)
			break;
	}

	double freq = cvGetTickFrequency() * 1000;
	if(nFileNum > 0)
		tAll = tAll / nFileNum;
	printf("\nSaliency detection finished, average time (file I/O time excluded): %.3fms\n", tAll / freq);


	return 0;

}


