from PIL import Image
import PIL
import os

rgb_dir = './RGB_LineDraw/download/'
linedraw_dir = './RGB_LineDraw/linedraw/'

rgb572_path = './rgb572/'
linedraw572_path = './linedraw572/'

rgb398_path = './rgb388/'
linedraw398_path = './linedraw388/'

if not os.path.exists(rgb572_path):
    os.mkdir(rgb572_path)

if not os.path.exists(rgb398_path):
    os.mkdir(rgb398_path)

if not os.path.exists(linedraw572_path):
    os.mkdir(linedraw572_path)

if not os.path.exists(linedraw398_path):
    os.mkdir(linedraw398_path)

for n in os.listdir(rgb_dir):
    file_name = n.split('.')[0]

    img_rgb = Image.open(rgb_dir+n)
    img_rgb = img_rgb.convert('RGB')

    img_linedraw = Image.open(linedraw_dir+n)
    img_linedraw = img_linedraw.convert('RGB')

    width, height = img_rgb.size
    if (width < 572) and (height < 572):
        os.remove(rgb_dir+n)
        os.remove(linedraw_dir+n)
        continue;

    resize_rgb = img_rgb.resize((572,572))
    resize_linedraw = img_linedraw.resize((572,572))

    resize_rgb.save(rgb572_path+file_name+'.jpg', 'JPEG', quality=100, optimize=True)
    resize_linedraw.save(linedraw572_path+file_name+'.jpg', 'JPEG',quality=100, optimize=True)

for n in os.listdir(rgb572_path):
    img_rgb = Image.open(rgb572_path+n)
    img_linedraw = Image.open(linedraw572_path+n)

    resize_rgb = img_rgb.resize((388,388))
    resize_linedraw = img_linedraw.resize((388,388))
    resize_rgb.save(rgb398_path+n, quality=100, optimize=True)
    resize_linedraw.save(linedraw398_path+n, quality=100, optimize=True)
