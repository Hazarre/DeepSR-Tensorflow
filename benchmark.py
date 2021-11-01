# Std import 
import glob 
import os 

# Third party imports
import tensorflow_hub as hub
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr, ycbcr2rgb
import numpy as np
from PIL import Image

# Local import 
from common import resolve_single
from srgan import generator
from mcdn import mcdn

def concat_imgs_h(imgs):
    '''
    Assume square images 
    '''
    n = len(imgs)
    dst = Image.new('RGB', (imgs[0].width*n, imgs[0].height))
    for i in range(n): 
        dst.paste(imgs[i], (imgs[0].width*i, 0))
    return dst

def postprocess(img):
    '''
    rgb to y channel
    crop off 4-pixel wide strip
    '''
    img = img.crop((4, 4, img.size[0]-4, img.size[1]-4)) 
    # crop of 4-pixel wide border
    img = rgb2ycbcr(img)
    return img[..., 0]

def eval_dataset(data_dir, upscaler = None):
    '''
    Computer the average psnr and ssim given a dataset and upscaling method
    data_dir : the path containing images of a dataset
    upscaler (Image) -> (Image) : the model that upscales the image in format rgb 256 -> rgb 256
    '''
    total_psnr = 0.0
    total_ssim = 0.0  
    fnames = glob.glob(data_dir+"/*.*")
    n_images = len(fnames)
    assert n_images > 0, "no image read"
    for f in fnames: 
        hr = Image.open(f)
        lr = hr.resize((hr.size[0]//4, hr.size[1]//4), Image.BICUBIC)
        sr = upscaler(lr)
        hr = hr.crop( (0,0, sr.size[0], sr.size[1]  )   )
        hr = postprocess(hr)
        sr = postprocess(sr)
        psnr = peak_signal_noise_ratio(hr,sr,data_range=255)
        ssim = structural_similarity(hr, sr, data_range=255)
        total_psnr += psnr
        total_ssim += ssim

    return {"psnr":total_psnr/n_images, "ssim": total_ssim/n_images}

def eval_image(img_dir, upscalers, output_dir):
    '''
    Function: 
        Save image stacks of lr, hr, and sr of an image upscaled by methods in upscalers into output_dir 
    Args: 
        img_dir: the img to stack 
        upscalers: a list of upscaling methods 
        output_dir: the output directory 
    '''
    n = len(upscalers) # two for bicubic and original image
    hr = Image.open(img_dir)
    # crop image size to the max multiple 4
    hr = hr.crop( (0,0, hr.size[0]//4 *4, hr.size[1]//4*4  )   ) 
    lr = hr.resize((hr.size[0]//4, hr.size[1]//4), Image.BICUBIC)
    bi = lr.resize(hr.size, Image.BICUBIC)
    dst = Image.new('RGB', (hr.width*(n+2), hr.height))
    dst.paste(bi, (0, 0))
    for i in range(n):
        upscaler = upscalers[i]
        sr = upscaler(lr)
        dst.paste(sr, (hr.width*(i+1), 0))
        
        np_sr = postprocess(sr)
        np_hr = postprocess(hr)
        psnr = peak_signal_noise_ratio(np_hr,np_sr,data_range=255)
        ssim = structural_similarity(np_hr, np_sr, data_range=255)
        print("%.2f / %.2f" % (psnr, ssim))

    dst.paste(hr, (hr.width*(n+1), 0))
    dst.save(output_dir)
    print("Image saved at: " + output_dir)

def srgan_upscaler(): 
    # load srgan generator 
    weights_dir = 'weights/srgan'
    weights_file = lambda filename: os.path.join(weights_dir, filename)
    gan_generator = generator()
    gan_generator.load_weights(weights_file('gan_generator.h5'))
    upscaler = lambda lr: Image.fromarray(  resolve_single(gan_generator, np.array(lr)).numpy())
    return upscaler

def esrgan_upscaler(): 
    model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
    upscaler = lambda lr: Image.fromarray(
        tf.cast(tf.clip_by_value(
            model(tf.cast( tf.expand_dims(lr, axis=0), tf.float32))
            , 0, 255), tf.uint8)[0].numpy()
    )
    return upscaler
 
def mcdn_upscaler(): 
    def upscale(lr):
        mcdn.load_weights("weights/mcdn/model6_valpsnr") 
        # normalized yuv
        yuv = (rgb2ycbcr(lr) - 16 )/ 219
        yuv = tf.cast(tf.expand_dims(yuv, axis=0), tf.float32)

        y_sr = mcdn(yuv[...,0]) * 219 + 16
    # y_sr[0,...,0].numpy().astype(np.uint8)
        w, h = yuv.shape[1], yuv.shape[2]
        
        hr = tf.image.resize(yuv, size=[w*4,h*4], method='bicubic', antialias=True) * 219 + 16

        hr = tf.concat( (y_sr, hr[...,1][...,None], hr[...,2][...,None]  ) , axis= -1)[0].numpy()
        hr = np.clip(  hr, 16, 235 )

        rgb = np.clip( ycbcr2rgb(hr)*255, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    return lambda lr: upscale(lr)



dataset_name ="Set14/original"
upscalers = [srgan_upscaler(), esrgan_upscaler(), mcdn_upscaler()]

for us in upscalers:
    result = eval_dataset("data/" + dataset_name, us) 
    print(result)



# fnames = glob.glob("data/" + dataset_name + "/*.*")
# for f in fnames: 
#     name = f.split("\\")[-1]
#     eval_image(f, upscalers, output_dir="output/BSDS100/" + name)





# print(fnames)
# hr = Image.open(fnames[0])
# lr = hr.resize((hr.size[0]//4, hr.size[1]//4), Image.BICUBIC)
# bi = lr.resize(hr.size, Image.BICUBIC)
# sr = Image.fromarray(  resolve_single(gan_generator, np.array(lr)).numpy() )

# imgs = concat_imgs_h( [hr, bi, sr])
# imgs.show()


