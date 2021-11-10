# Std import 
import glob 
import os 
# turn off gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

# Third party imports
import tensorflow as tf
# import tensorflow_hub as hub
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr, ycbcr2rgb
import numpy as np
from PIL import Image
import cv2

# Local import 
from common import resolve_single
from srgan import generator
from mcdn import mcdn
from FSRCNN.model import FSRCNN

R = 4 
SIZE = 100


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

def fsrcnn_upscaler():
    # load model 
    IS_FSRCNN_S = False 
    # prep the model 
    fsrcnn = FSRCNN(d=32, s=5, m=1, r=4) if IS_FSRCNN_S else FSRCNN()
    fsrcnn.build((100, None, None, 1))
    name_model = "fsrcnn_s" if IS_FSRCNN_S else "fsrcnn"
    weights_dir = "FSRCNN/weights_" + name_model + ".h5"
    fsrcnn.load_weights(weights_dir)

    def upscale(lr):

        yuv = (rgb2ycbcr(lr) - 16 )/ 219
        yuv = tf.cast(tf.expand_dims(yuv, axis=0), tf.float32)
        y_lr= yuv[...,0][..., None]

        y_sr = fsrcnn.predict(y_lr) * 219 + 16
        w, h = yuv.shape[1], yuv.shape[2]
        yuv_hr = tf.image.resize(yuv, size=[w*R,h*R], method='bicubic', antialias=True) * 219 + 16

        yuv_sr = tf.concat( (y_sr, yuv_hr[...,1][...,None], yuv_hr[...,2][...,None]  ) , axis= -1)[0].numpy()
        yuv_sr = np.clip( yuv_sr, 16, 235 )
        rgb_sr    = np.clip( ycbcr2rgb(yuv_sr)*255, 0, 255 ).astype(np.uint8)
        rgb_sr = Image.fromarray(rgb_sr)
        return rgb_sr
    return lambda lr: upscale(lr)
        


dataset_name ="Set14"
upscalers = [fsrcnn_upscaler()]


for us in upscalers:
    result = eval_dataset("data/" + dataset_name, us) 
    print(result)





def upscale_video(vid_in_path, vid_out_path, upscaler):
    cap = cv2.VideoCapture(vid_in_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width*4, frame_height*4)

    fps = int(cap.get(5))
    frame_count = cap.get(7)
    print("frame_count: ", frame_count)
    output_writer = cv2.VideoWriter(vid_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret: 
            # cover to PIL image 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lr = Image.fromarray(frame)
            sr = lr.resize(frame_size, Image.BOX)

            # sr = upscaler(lr)
    
            # sr = sr.resize(frame_size, Image.BICUBIC)
            sr = cv2.cvtColor(np.array(sr), cv2.COLOR_RGB2BGR)
            output_writer.write(sr)
            print("writing")
        else: 
            print('Stream disconnted')
            break

    cap.release()
    output_writer.release()



# upscalers = {"srgan": srgan_upscaler()} # "mcdn": mcdn_upscaler()
# vid_name = 'vid2_480p_4.mp4'
# vid_in_path  = 'data/videos/' + vid_name
# vid_out_path = 'output/videos/upscaled_box_' + vid_name

# for model_name in upscalers.keys():
#     print(model_name)
#     upscaler = upscalers[model_name]
#     upscale_video(vid_in_path, vid_out_path, upscaler)



