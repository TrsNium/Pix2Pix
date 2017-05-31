from PIL import Image
import PIL
import os

rgb_dir = './RGB_LineDraw/download/'
linedraw_dir = './RGB_LineDraw/linedraw/'

rgb572_path = './rgb572/'
linedraw572_path = './linedraw572/'

rgb512_path = './rgb512/'
linedraw512_path = './linedraw512/'

rgb388_path = './rgb388/'
linedraw388_path = './linedraw388/'

rgb388_crop_path = './crop_rgb388/'
linedraw388_crop_path = './crop_linedraw388/'

if not os.path.exists(rgb572_path):
    os.mkdir(rgb572_path)

if not os.path.exists(rgb388_path):
    os.mkdir(rgb388_path)

if not os.path.exists(linedraw572_path):
    os.mkdir(linedraw572_path)

if not os.path.exists(rgb512_path):
    os.mkdir(rgb512_path)

if not os.path.exists(linedraw512_path):
    os.mkdir(linedraw512_path)

if not os.path.exists(linedraw388_path):
    os.mkdir(linedraw388_path)

if not os.path.exists(rgb388_crop_path):
    os.mkdir(rgb388_crop_path)

if not os.path.exists(linedraw388_crop_path):
    os.mkdir(linedraw388_crop_path)

for n in os.listdir(rgb_dir):
    try:
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
    
    except:
        continue

for n in os.listdir(rgb572_path):
    img_rgb = Image.open(rgb572_path+n)
    img_linedraw = Image.open(linedraw572_path+n)

    resize_rgb = img_rgb.resize((512,512))
    resize_linedraw = img_linedraw.resize((512,512))
    resize_rgb.save(rgb512_path+n, quality=100, optimize=True)
    resize_linedraw.save(linedraw512_path+n, quality=100, optimize=True)

for n in os.listdir(rgb572_path):
    img_rgb = Image.open(rgb572_path+n)
    img_linedraw = Image.open(linedraw572_path+n)

    resize_rgb = img_rgb.resize((388,388))
    resize_linedraw = img_linedraw.resize((388,388))
    resize_rgb.save(rgb388_path+n, quality=100, optimize=True)
    resize_linedraw.save(linedraw388_path+n, quality=100, optimize=True)

for n in os.listdir(rgb572_path):
    img_rgb = Image.open(rgb572_path+n)
    img_linedraw = Image.open(linedraw572_path+n)

    crop_rgb = img_rgb.crop((92,92,572-92,572-92))
    crop_linedraw = img_linedraw.crop((92,92,572-92,572-92))

    crop_rgb.save(rgb388_crop_path+n, 'JPEG', quality=100, optimize=True) 
    crop_linedraw.save(linedraw388_crop_path+n,'JPEG',quality=100, optimize=True)  
