import cv2


def crop_video(vid_in_path, vid_out_path):
    cap = cv2.VideoCapture(vid_in_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    fps = int(cap.get(5))
    frame_count = cap.get(7)
    print("frame_count: ", frame_count)
    output_writer = cv2.VideoWriter(vid_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,(target_width, target_height))

    top  = int( (1080 - target_height)/2) 
    left = int( (1920 - target_height)/2) 
    while cap.isOpened():
        ret, frame = cap.read()
        if ret: 
            cropped = frame[ top: top + target_height   , left : left + target_width] 
            # cv2.imshow("cropped", cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllwindows()
            output_writer.write(cropped)
            print("writing")
        else: 
            print('Stream disconnted')
            print( (target_width, target_height) )
            print( cropped.shape)

            break

    cap.release()
    output_writer.release()


crop_ratio = 4
if __name__ == "__main__":
    for i in [1,2,3,4]:
        video_dir = "/home/henrychang/Desktop/S4video/FHD/vid%d.mp4" % i 
        target_width  = 1920 // crop_ratio 
        target_height = 1080 // crop_ratio
        output_dir = "/home/henrychang/Desktop/vid%d_cropped_1_%dth.mp4" % (i, crop_ratio) 
        crop_video(video_dir, output_dir)