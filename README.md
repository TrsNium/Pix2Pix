# Pix2Pix
If you haven't known about UNet, check out [here](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)   
I will use this experience for CycleGAN.   

# DataSet
I got dataset from pixiv and safebooru.    
if you wanna get them from safebooru, please check out [this repo](https://github.com/TrsNium/scraping_safebooru)
    
# Set Up
Command line is like following.
```
$ git clone git@github.com:TrsNium/scraping_safebooru.git
$ git clone git@github.com:TrsNium/Pix2Pix__.git
$ cd Pix2Pix__/data
$ mkdir RGB_LineDraw
$ cd ../../
$ cd scraping_safebooru
$ python scraping.py
$ mv ./download/ ../Pix2Pix__/data/RGB_LineDraw
$ mv ./linedraw/ ../Pix2Pix__/data/RGB_LineDraw
$ cd ../Pix2_Pix/data
$ python resize.py
$ cd ..
$ python train.py
```     
And I use some library.    
Please install tensorflow,pillow, BeutifulSoup4 and cv2
