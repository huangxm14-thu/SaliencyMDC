#pragma once
#include <cv.h>
#include <highgui.h>
#include <io.h>
#include "imgLib.h"
using namespace cv;


void ImgResize(IplImage* pImgSrc, IplImage* pImgDst)
{
	if(pImgSrc->nChannels != 3 || pImgDst->nChannels != 3)
		return;

	if(pImgSrc->width == pImgDst->width && pImgSrc->height == pImgDst->height)
	{
		cvCopy(pImgSrc, pImgDst);
		return;
	}

	int x, y;
	int* posX = new int[pImgDst->width];
	int* posY = new int[pImgDst->height];
	
	for(x = 0; x < pImgDst->width; x ++)
	{
		float p = (x + 0.5) * (pImgSrc->width - 1) / pImgDst->width;
		posX[x] = p + 0.5;
	}
	for(y = 0; y < pImgDst->height; y ++)
	{
		float p = (y + 0.5) * (pImgSrc->height - 1) / pImgDst->height;
		posY[y] = p + 0.5;
	}

	for(y = 0; y < pImgDst->height; y ++)
	{
		int pos = posY[y];
		unsigned char *pBufDst  = (unsigned char *)pImgDst->imageData + y * pImgDst->widthStep;
		unsigned char *pBufSrc0  = (unsigned char *)pImgSrc->imageData + (pos + 0) * pImgSrc->widthStep;
		unsigned char *pBufSrc1  = (unsigned char *)pImgSrc->imageData + (pos + 1) * pImgSrc->widthStep;
		for(x = 0; x < pImgDst->width; x ++)
		{
			int xOffset = 3 * posX[x];
			pBufDst[0] = pBufSrc0[xOffset + 0];
			pBufDst[1] = pBufSrc0[xOffset + 1];
			pBufDst[2] = pBufSrc0[xOffset + 2];
			pBufDst  += 3;
		}
	}
	delete [] posX;
	delete [] posY;
}




#define GAMMA_ZOOM 256
#define FTBL_SIZE 10240
#define FTBL_BITS 8
#include "RGB2LAB.h"


void RGB2LAB(IplImage *pImgSrc)
{
const int M[] = {
	FTBL_SIZE / GAMMA_ZOOM * 0.412453 / 0.95047, 
	FTBL_SIZE / GAMMA_ZOOM * 0.357580 / 0.95047, 
	FTBL_SIZE / GAMMA_ZOOM * 0.180423 / 0.95047,
	FTBL_SIZE / GAMMA_ZOOM * 0.212671, 
	FTBL_SIZE / GAMMA_ZOOM * 0.715160, 
	FTBL_SIZE / GAMMA_ZOOM * 0.072169,
	FTBL_SIZE / GAMMA_ZOOM * 0.019334 / 1.08883, 
	FTBL_SIZE / GAMMA_ZOOM * 0.119193 / 1.08883, 
	FTBL_SIZE / GAMMA_ZOOM * 0.950227 / 1.08883
};

	for(int y = 0; y <= pImgSrc->height - 1; y ++)
	{
		unsigned char *pBuf = (unsigned char *)pImgSrc->imageData +y * pImgSrc->widthStep;
		for(int x = 0; x <= pImgSrc->width - 1; x ++)
		{
			int B = gammaTbl[pBuf[0]];
			int G = gammaTbl[pBuf[1]];
			int R = gammaTbl[pBuf[2]];

			int FX = fTbl[M[0] * R + M[1] * G + M[2] * B];
			int FY = fTbl[M[3] * R + M[4] * G + M[5] * B];
			int FZ = fTbl[M[6] * R + M[7] * G + M[8] * B];

			pBuf[0] = (116 * FY >> FTBL_BITS);// - 16;		//LÊä³ö·¶Î§ÊÇ16-116
		    pBuf[1] = (500 * (FX - FY) >> FTBL_BITS) + 128;
		    pBuf[2] = (200 * (FY - FZ) >> FTBL_BITS) + 128;

			pBuf += 3;
		}

	}
}



/****************************************************************************************\
*                                       Watershed                                        *
\****************************************************************************************/

typedef struct CvWSNode
{
    struct CvWSNode* next;
    int mask_ofs;
}
CvWSNode;


typedef struct CvWSQueue
{
    CvWSNode* first;
    CvWSNode* last;
}
CvWSQueue;

static CvWSNode*
icvAllocWSNodes( CvMemStorage* storage )
{
    CvWSNode* n = 0;

    int i, count = (storage->block_size - sizeof(CvMemBlock))/sizeof(*n) - 1;

    n = (CvWSNode*)cvMemStorageAlloc( storage, count*sizeof(*n) );
    for( i = 0; i < count-1; i++ )
        n[i].next = n + i + 1;
    n[count-1].next = 0;

    return n;
}


#define m_def (m_fg | m_bg)

void markerWatershed( Mat* src, Mat* dst)
{
    const int IN_QUEUE = 0x2;
    const int WSHED = 0x1;
    const int NQ = 256;
    cv::Ptr<CvMemStorage> storage;

    CvSize size;
    CvWSNode* free_node = 0, *node;
    CvWSQueue q[NQ];
    int active_queue;
    int i, j;
    int db, dg, dr;
    char* mask;
    uchar* img;
    int mstep, istep;
    int subs_tab[513];

    // MAX(a,b) = b + MAX(a-b,0)
    #define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
    // MIN(a,b) = a - MAX(a-b,0)
    #define ws_min(a,b) min(a,b)/*((a) - subs_tab[(a)-(b)+NQ])*/

    #define ws_push(idx,mofs)  \
    {                               \
        if( !free_node )            \
            free_node = icvAllocWSNodes( storage );\
        node = free_node;           \
        free_node = free_node->next;\
        node->next = 0;             \
        node->mask_ofs = mofs;      \
        if( q[idx].last )           \
            q[idx].last->next=node; \
        else                        \
            q[idx].first = node;    \
        q[idx].last = node;         \
    }

    #define ws_pop(idx,mofs)   \
    {                               \
        node = q[idx].first;        \
        q[idx].first = node->next;  \
        if( !node->next )           \
            q[idx].last = 0;        \
        node->next = free_node;     \
        free_node = node;           \
        mofs = node->mask_ofs;      \
    }

    #define c_diff(ptr1,ptr2,diff)      \
    {                                   \
        db = abs((ptr1)[0] - (ptr2)[0]);\
        dg = abs((ptr1)[1] - (ptr2)[1]);\
        dr = abs((ptr1)[2] - (ptr2)[2]);\
        diff = (db + dg + dr) / 3;         \
        assert( 0 <= diff && diff <= 255 ); \
    }

	size = cvSize(src->cols, src->rows);
    storage = cvCreateMemStorage();

    istep = src->step;
	img = src->ptr(0);
    mstep = dst->step / sizeof(mask[0]);
    mask = (char *)dst->ptr(0);

    memset( q, 0, NQ*sizeof(q[0]) );

    for( i = 0; i < 256; i++ )
        subs_tab[i] = 0;
    for( i = 256; i <= 512; i++ )
        subs_tab[i] = i - 256;

    // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
    for( j = 0; j < size.width; j++ )
        mask[j] = mask[j + mstep*(size.height-1)] = WSHED;

    // initial phase: put all the neighbor pixels of each marker to the ordered queue -
    // determine the initial boundaries of the basins
    for( i = 1; i < size.height-1; i++ )
    {
        img += istep; mask += mstep;
        mask[0] = mask[size.width-1] = WSHED;

        for( j = 1; j < size.width-1; j++ )
        {
            char* m = mask + j;
            if( m[0] < 0 ) m[0] = 0;
            if( m[0] == 0 && ((m[-1] & m_def) || (m[1] & m_def) || (m[-mstep] & m_def) || (m[mstep] & m_def)) )
            {
                uchar* ptr = img + j*3;
                int idx = 256, t;
                if( m[-1] & m_def )
				{
                    c_diff( ptr, ptr - 3, idx );
				}
                if( m[1] & m_def )
                {
                    c_diff( ptr, ptr + 3, t );
                    idx = ws_min( idx, t );
                }
                if( m[-mstep] & m_def )
                {
                    c_diff( ptr, ptr - istep, t );
                    idx = ws_min( idx, t );
                }
                if( m[mstep] & m_def )
                {
                    c_diff( ptr, ptr + istep, t );
                    idx = ws_min( idx, t );
                }
                assert( 0 <= idx && idx <= 255 );
                ws_push( idx, i*mstep + j);
                m[0] = IN_QUEUE;
            }
        }
    }

    // find the first non-empty queue
    for( i = 0; i < NQ; i++ )
        if( q[i].first )
            break;

    // if there is no markers, exit immediately
    if( i == NQ )
        return;

    active_queue = i;
    img = src->ptr(0);
    mask = (char *)dst->ptr(0);

    // recursively fill the basins
    for(;;)
    {
        int mofs;
        int lab = 0, t;
        char* m;
        uchar* ptr;

        if( q[active_queue].first == 0 )
        {
            for( i = active_queue+1; i < NQ; i++ )
                if( q[i].first )
                    break;
            if( i == NQ )
                break;
            active_queue = i;
        }

        ws_pop( active_queue, mofs);

        m = mask + mofs;
        ptr = img + 3 * mofs;
		/*
        t = m[-1];
        if( t & m_def ) lab = t;
        t = m[1];
        if( t & m_def )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }
        t = m[-mstep];
        if( t & m_def )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }
        t = m[mstep];
        if( t & m_def )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }
		*/
		lab = m[-1] | m[1] | m[-mstep] | m[mstep];
		lab &= m_def;
		if(lab == m_def)
			lab = WSHED;
        assert( lab != 0 );
        m[0] = lab;
        if( lab == WSHED )
            continue;

        if( m[-1] == 0 )
        {
            c_diff( ptr, ptr - 3, t );
            ws_push( t, mofs - 1);
            active_queue = ws_min( active_queue, t );
            m[-1] = IN_QUEUE;
        }
        if( m[1] == 0 )
        {
            c_diff( ptr, ptr + 3, t );
            ws_push( t, mofs + 1);
            active_queue = ws_min( active_queue, t );
            m[1] = IN_QUEUE;
        }
        if( m[-mstep] == 0 )
        {
            c_diff( ptr, ptr - istep, t );
            ws_push( t, mofs - mstep);
            active_queue = ws_min( active_queue, t );
            m[-mstep] = IN_QUEUE;
        }
        if( m[mstep] == 0 )
        {
            c_diff( ptr, ptr + istep, t );
            ws_push( t, mofs + mstep);
            active_queue = ws_min( active_queue, t );
            m[mstep] = IN_QUEUE;
        }
    }
}


