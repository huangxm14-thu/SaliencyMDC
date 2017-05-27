#pragma once
#include <cv.h>
#include <highgui.h>
#include <io.h>
using namespace cv;


#define m_fg 0x40
#define m_bg 0x20

void ImgResize(IplImage* pImgSrc, IplImage* pImgDst);
void RGB2LAB(IplImage *pImgSrc);
void markerWatershed( Mat* src, Mat* dst);
