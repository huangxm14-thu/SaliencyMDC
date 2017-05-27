========================================================================
     300 FPS Salient Object Detection via Minimum Directional Contrast
========================================================================
* Copyright(c)2015-2017 Xiaoming Huang, <huangxm14@mails.tsinghua.edu.cn>.
* Last update:26-05-2017. 

This is the demo of our work "300 FPS Salient Object Detection via Minimum Directional Contrast",
Note that our demo program developed with visual studio 2012 and opencv 2.4.9, we have test the program on a 64-bit PC with Win7 OS,
We achieve 300 FPS speed performance on Intel Core i5-4590 CPU @ 3.3 GHz and 8GB RAM.
If you have any problem, please don't hesitate to contact with huangxm14@mails.tsinghua.edu.cn.

Note that we find that time cost of demo program fluctuate due to some reason, we suggest run 5 times and take average time cost.



1.Directory introduction
--ECCSD: 100 images and ground-truth selected from ECCSD dataset, saliency result of ours, RC [18] and MB+ [40];
--MSRA: 100 images and ground-truth selected from MSRA10K dataset, saliency result of ours, RC [18] and MB+ [40];
--Demo.bat: one .bat program to detect saliency of ECCSD amd MSRA directory
--SaliencyMDC.exe: our program

2.Run Demo Program
You can run Demo.bat, input and output files are in directory Dataset.


