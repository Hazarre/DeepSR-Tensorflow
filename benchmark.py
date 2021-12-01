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
from SRGAN.model import generator
from FSRCNN.model import FSRCNN
from ASRCNN.model import ASRCNN
from MCDN.model import MCDN

R = 2
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
    hr = hr.crop( (0,0, hr.size[0]//R *R, hr.size[1]//R*R  )   ) 
    lr = hr.resize((hr.size[0]//R, hr.size[1]//R), Image.BICUBIC)
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
    def resolve_single(model, lr):
        return resolve(model, tf.expand_dims(lr, axis=0))[0]

    def resolve(model, lr_batch):
        lr_batch = tf.cast(lr_batch, tf.float32)
        sr_batch = model(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch
    # load srgan generator 
    gan_generator = generator()
    gan_generator.load_weights('weights/srgan_generator.h5')
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
 
def mcdn_upscaler(r=R): 
    def upscale(lr):
        mcdn = MCDN(if_train=False, scale=r, name="MCDN_X%d" % r)
        mcdn.load_weights("weights/mcdn_x%d.h5" % r, by_name=True) 
        # normalized yuv
        yuv = (rgb2ycbcr(lr) - 16 )/ 219
        yuv = tf.cast(tf.expand_dims(yuv, axis=0), tf.float32)

        y_sr = mcdn(yuv[...,0]) * 219 + 16
    # y_sr[0,...,0].numpy().astype(np.uint8)
        w, h = yuv.shape[1], yuv.shape[2]
        
        hr = tf.image.resize(yuv, size=[w*R,h*R], method='bicubic', antialias=True) * 219 + 16

        hr = tf.concat( (y_sr, hr[...,1][...,None], hr[...,2][...,None]  ) , axis= -1)[0].numpy()
        hr = np.clip(  hr, 16, 235 )

        rgb = np.clip( ycbcr2rgb(hr)*255, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    return lambda lr: upscale(lr)

def fsrcnn_upscaler(r=R):
    # load model 
    IS_FSRCNN_S = False 
    # prep the model 
    fsrcnn = FSRCNN(d=32, s=5, m=1, r=r) if IS_FSRCNN_S else FSRCNN(r=r)
    fsrcnn.build((11, None, None, 1))
    fsrcnn.summary()
    name_model = "fsrcnn_s" if IS_FSRCNN_S else "fsrcnn"
    weights_dir = "FSRCNN/weights_" + name_model + "x%d.h5" % R
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
        rgb_sr = np.clip( ycbcr2rgb(yuv_sr)*255, 0, 255 ).astype(np.uint8)
        rgb_sr = Image.fromarray(rgb_sr)
        return rgb_sr
    return lambda lr: upscale(lr)
        
def bicubic_upscaler(r=R):
    return lambda lr: lr.resize( (lr.size[0]*r, lr.size[1]*r) , Image.BICUBIC)

def bilinear_upscaler(r=R):
    return lambda lr: lr.resize( (lr.size[0]*r, lr.size[1]*r) , Image.BILINEAR)

def asrcnn_upscaler():
    # load model 
    IS_ASRCNN_S = False 
    # prep the model 
    asrcnn = ASRCNN(d=32, s=5, m=1, r=R) if IS_ASRCNN_S else ASRCNN()
    asrcnn.build((100, None, None, 1))
    name_model = "asrcnn_s" if IS_ASRCNN_S else "asrcnn"
    weights_dir = "ASRCNN/weights_" + name_model + ".h5"
    asrcnn.load_weights(weights_dir)

    def upscale(lr):
        yuv = (rgb2ycbcr(lr) - 16 )/ 219
        yuv = tf.cast(tf.expand_dims(yuv, axis=0), tf.float32)
        y_lr= yuv[...,0][..., None]

        y_sr = asrcnn.predict(y_lr) * 219 + 16
        w, h = yuv.shape[1], yuv.shape[2]
        yuv_hr = tf.image.resize(yuv, size=[w*R,h*R], method='bicubic', antialias=True) * 219 + 16

        yuv_sr = tf.concat( (y_sr, yuv_hr[...,1][...,None], yuv_hr[...,2][...,None]  ) , axis= -1)[0].numpy()
        yuv_sr = np.clip( yuv_sr, 16, 235 )
        rgb_sr    = np.clip( ycbcr2rgb(yuv_sr)*255, 0, 255 ).astype(np.uint8)
        rgb_sr = Image.fromarray(rgb_sr)
        return rgb_sr
    return lambda lr: upscale(lr)

def upscale_video(vid_in_path, vid_out_path, upscaler, r=R, max_frame = 30*15):
    '''
    Upscales Image by r times. 

    Args:
        r: ratio to upsacle  
        vid_in_path:
        vid_out_path: 
        upscaler: 
        max_frame: 
        r: 
    '''
    cap = cv2.VideoCapture(vid_in_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    sr_frame_size = (frame_width*r, frame_height*r)
    fps = int(cap.get(5))
    frame_count = cap.get(7)
    print("frame_count: ", frame_count)
    output_writer = cv2.VideoWriter(vid_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, sr_frame_size)

    frame_count = 0 
    while cap.isOpened():
        frame_count += 1 
        ret, frame = cap.read()
        if ret: 
            # cover to PIL image 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lr = Image.fromarray(frame)
            sr = upscaler(lr)
            sr = cv2.cvtColor(np.array(sr), cv2.COLOR_RGB2BGR)
            output_writer.write(sr)
            print("writing")
        else: 
            print('Stream disconnted')
            break
        if frame_count >= max_frame:
            break

    cap.release()
    output_writer.release()

def enhance_video(vid_in_path, vid_out_path, upscaler, r=R, max_frame = 30*15):
    '''
    Upscales then bicubic downscale vidoes (frame by frame) to its original size. 

    Args:

    r: ratio to upsacle  
    vid_in_path:
    vid_out_path: 
    upscaler: 
    max_frame: 
    r: 
    '''
    cap = cv2.VideoCapture(vid_in_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    fps = int(cap.get(5))
    frame_count = cap.get(7)
    print("frame_count: ", frame_count)
    output_writer = cv2.VideoWriter(vid_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    frame_count = 0 
    while cap.isOpened():
        frame_count += 1 
        ret, frame = cap.read()
        if ret: 
            # cover to PIL image 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lr = Image.fromarray(frame)
            sr = upscaler(lr).resize(frame_size, Image.BICUBIC)
            sr = cv2.cvtColor(np.array(sr), cv2.COLOR_RGB2BGR)
            output_writer.write(sr)
            print("writing")
        else: 
            print('Stream disconnted')
            break
        if frame_count >= max_frame:
            break

    cap.release()
    output_writer.release()

def restore_video(vid_in_path, vid_out_path, upscaler, r=R, max_frame = 30*15):
    '''
    Bicubic downscales then upscales vidoes (frame by frame) to its original size. 
    
    Args:
        r: ratio to upsacle  
        vid_in_path:
        vid_out_path: 
        upscaler: 
        max_frame: 
        r: 
    '''
    cap = cv2.VideoCapture(vid_in_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    fps = int(cap.get(5))
    frame_count = cap.get(7)
    print("frame_count: ", frame_count)
    output_writer = cv2.VideoWriter(vid_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    frame_count = 0 
    while cap.isOpened():
        frame_count += 1 
        ret, frame = cap.read()
        if ret: 
            # cover to PIL image 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hr = Image.fromarray(frame)
            lr = hr.resize((hr.size[0]//R, hr.size[1]//R), Image.BICUBIC)
            lr.show()
            sr = upscaler(lr)
            sr = cv2.cvtColor(np.array(sr), cv2.COLOR_RGB2BGR)
            output_writer.write(sr)
            print("writing")
        else: 
            print('Stream disconnted')
            break
        if frame_count >= max_frame:
            break

    cap.release()
    output_writer.release()

# dataset_name ="BSDS100"
# upscalers = [mcdn_upscaler()]
# output_dir = 'data/output/' + dataset_name +"/"

# for img_name in os.listdir("data/" + dataset_name):
#     eval_image("data/" + dataset_name +"/" + img_name, upscalers, output_dir + img_name)

# for us in upscalers:
#     result = eval_dataset("data/" + dataset_name, us) 
#     print(result)



upscalers={ "mcdn": mcdn_upscaler()}
            "bilinear": bilinear_upscaler(), 
            "bicubic": bicubic_upscaler(), 
            "fsrcnn": fsrcnn_upscaler() } 

for model_name in upscalers.keys():
    for i in [1,2,3,4,5]:
        upscaler = upscalers[model_name]
        vid_in_path  = "/home/henrychang/Desktop/vid%d_cropped_1_2th.mp4" % i

        # vid_out_path = "/home/henrychang/Desktop/Enhanced/vid%d_"  % i + model_name + ".mp4"
        # enhance_video(vid_in_path, vid_out_path, upscaler, r=R, max_frame = 30*60*10)

        # vid_out_path = "/home/henrychang/Desktop/Restoredx2/vid%d_"  % i + model_name + ".mp4"
        # restore_video(vid_in_path, vid_out_path, upscaler, r=R, max_frame = 30*60*10)

        vid_in_path  = "/home/henrychang/Desktop/vid%d_cropped_1_4th.mp4" % i
        vid_out_path = "/home/henrychang/Desktop/Upscaledx2/vid%d_"  % i + model_name + ".mp4"
        upscale_video(vid_in_path, vid_out_path, upscaler, r=R, max_frame = 30*60*10)





 