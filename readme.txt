================================================================================================
               Xiaoming Huang, and Yu-Jin Zhang,
               300 FPS Salient Object Detection via Minimum Directional Contrast,
               IEEE Trans. on Image Processing
================================================================================================
* Copyright(c)2015-2017 Xiaoming Huang, <huangxm14@mails.tsinghua.edu.cn>.
* Last update:26-05-2017. 

This is the code of above work. The source code is developed with visual studio 2012 and opencv 2.4.9.
We have test the program on a 64-bit PC with Win7 OS,achieve 300 FPS speed performance on Intel Core i5-4590 CPU @ 3.3 GHz and 8GB RAM.
If you have any problem, please don't hesitate to contact with huangxm14@mails.tsinghua.edu.cn.

Note that we find that time cost of program fluctuate due to some reason, we suggest run 5 times and take average time cost.


1.Directory introduction
--Demo: One demo program
--SaliencyMDC.sln: solution file (visual studio 2012)
--opencv2.4.9:header files and lib of opencv
--test.cpp: test file of our method
--Saliency.cpp: detail implementation of our method

2.Build Project
Open solution file saliencyMDC.sln with VS2012, select Release @ x64 mode, build.

3.Run
You need set directory of input images, e.g., .\Demo\MSRA, then run the program, the final result will be also saved in the directory.



