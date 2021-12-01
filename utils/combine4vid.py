
import cv2
import numpy as np

def combine_videos(vid_dirs, output_dir):
    vid_top_left_dir, vid_top_right_dir , vid_bot_left_dir,  vid_bot_right_dir =  vid_dirs
    cap1 = cv2.VideoCapture(vid_top_left_dir)
    cap2 = cv2.VideoCapture(vid_top_right_dir)
    cap3 = cv2.VideoCapture(vid_bot_left_dir)
    cap4 = cv2.VideoCapture(vid_bot_right_dir)

    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    frame_size = (frame_width, frame_height)
    target_frame_size = (frame_width*2, frame_height*2)
    fps = int(cap1.get(5))
    frame_count = cap1.get(7)
    print("frame_size: ", frame_size)
    print("frame_count: ", frame_count)
    print("fps: ", fps)
     
    output_writer = cv2.VideoWriter(output_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, target_frame_size)
    print(cap1.isOpened())
    print(cap2.isOpened())
    print(cap3.isOpened())
    print(cap4.isOpened())
    while cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
        
        if ret1 and ret2 and ret3 and ret4: 
            top = cv2.hconcat([frame1, frame2])
            bot = cv2.hconcat([frame3, frame4])
            combined = cv2.vconcat( [top, bot] )
            output_writer.write(combined)
            print("writing")
        else: 
            print('Stream disconnted')
            break

    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    output_writer.release()


if __name__ == "__main__":
    for i in [5]:
        main_dir = "/home/henrychang/Desktop/"
        sub_dir  = "Restoredx2/"
        output_dir =        main_dir + sub_dir + "vid%dRestoredx2.mp4"% i

        vid_top_left_dir  = main_dir + sub_dir + "vid%d_bicubic.mp4"  % i 
        vid_top_right_dir = main_dir + sub_dir + "vid%d_mcdn.mp4"     % i 
        vid_bot_left_dir  = main_dir + sub_dir + "vid%d_bilinear.mp4" % i 
        vid_bot_right_dir = main_dir + sub_dir + "vid%d_fsrcnn.mp4"   % i 

        vid_dirs = [vid_top_left_dir, vid_top_right_dir, vid_bot_left_dir, vid_bot_right_dir]
        combine_videos(vid_dirs, output_dir)