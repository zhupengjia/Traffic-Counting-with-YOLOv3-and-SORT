# import the necessary packages
import numpy as np
import argparse
import math
import imutils
import time
import cv2
import datetime
import os
from yolo import YOLO 
from PIL import Image, ImageDraw, ImageFont
from sort import *
import tensorflow as tf
from timeit import default_timer as timer


def main(yolo):
    tracker = Sort()
    memory = {}
    lines = [[(203, 128), (558, 253)],
             [(396, 72), (516, 65)],
             [(557, 18), (646, 109)]]
    angle_split = [0, None, None]

    car = [[0] * len(lines), [0] * len(lines)] #up, down
    motor = [[0] * len(lines), [0] * len(lines)]
    bus = [[0] * len(lines), [0] * len(lines)]
    truck = [[0] * len(lines), [0] * len(lines)]

    descrip_pos = (569, 332)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    input_file = "data/0800-0815.mp4"

    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.4,
        help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    # Return true if line segments AB and CD intersect
    def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def vector_angle(midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))

    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([args["yolo"], "classes.names"])
    #LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    #Put your video path here
    vs = cv2.VideoCapture(input_file)
    writer = None
    (W, H) = (None, None)
    
    font = ImageFont.truetype(font="font/FiraMono-Medium.otf", size=16)
    
    frameIndex = 0

    # loop over frames from the video file stream
    prev_time = timer()
    accum_time = 0
    curr_fps = 0
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        image = Image.fromarray(frame[...,::-1])
        boxes, out_class, confidences, midPoint = yolo.detect_image(image)
        image = np.asarray(image)

        # and class IDs, respectively
        classIDs = []
        #classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.2)

        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)

        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}
        
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
                
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))

                    origin_p0 = (p0[0], frame.shape[0] - p0[1])  # get midpoint respective to botton-left
                    origin_p1 = (p1[0], frame.shape[0] - p1[1])
                    cv2.line(frame, p0, p1, color, 3)

                    for line_i, line in enumerate(lines):
                        if intersect(p0, p1, line[0], line[1]):
                            angle = vector_angle(origin_p0, origin_p1)
                            detected_class = yolo.counter(p0,out_class, midPoint)

                            if angle_split[line_i] is None or  angle >= angle_split[line_i]:
                                # up
                                if detected_class == 'car':
                                    car[0][line_i] += 1
                                elif detected_class == 'motorbike':
                                    motor[0][line_i] += 1
                                elif detected_class == 'bus':
                                    bus[0][line_i] += 1
                                else:
                                    truck[0][line_i] += 1
                            else:
                                # down
                                if detected_class == 'car':
                                    car[1][line_i] += 1
                                elif detected_class == 'motorbike':
                                    motor[1][line_i] += 1
                                elif detected_class == 'bus':
                                    bus[1][line_i] += 1
                                else:
                                    truck[1][line_i] += 1

                    '''        
                    if p0[0] > 921 and p0[0] < 1440 and p0[1] < 606 and p0[1] > 437:
                        p0Memory = p0
                        state = True
                        if 
                    '''        
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                text = "{}".format(indexIDs[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1
                
        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        draw.text(descrip_pos, '     car motor bus truck', fill=(255,255,255), font=font)
        descrip_pos_v = descrip_pos[1] + 20
        for line_i, line in enumerate(lines):
            if angle_split[line_i] is not None:
                text_pos = (descrip_pos[0], descrip_pos_v)

                draw.text(text_pos, 'up   {}  {}  {}  {}'.format(car[0][line_i],
                                                                   motor[0][line_i],
                                                                   bus[0][line_i],
                                                                   truck[0][line_i]),
                        fill=colors[line_i], font=font)
                descrip_pos_v += 20
                text_pos = (descrip_pos[0], descrip_pos_v)

                draw.text(text_pos, 'down {}  {}  {}  {}'.format(car[1][line_i],
                                                        motor[1][line_i],
                                                        bus[1][line_i],
                                                        truck[1][line_i]),
                            fill=colors[line_i], font=font)
            else:
                text_pos = (descrip_pos[0], descrip_pos_v)

                draw.text(text_pos, '     {}  {}  {}  {}'.format(car[0][line_i],
                                                                 motor[0][line_i],
                                                                 bus[0][line_i],
                                                                 truck[0][line_i]),
                          fill=colors[line_i], font=font)
            descrip_pos_v += 20

        frame = np.asarray(frame)
        # draw line
        for line_i, line in enumerate(lines):
            cv2.line(frame, line[0], line[1], colors[line_i], 1)

        # draw counter
        # counter += 1
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        #cv2.putText(frame, text=fps, org=(3, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #            fontScale=0.8, color=(255, 0, 0), thickness=1)
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_fps = vs.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, 'fps: %d' %(video_fps), (9,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 5)
            writer = cv2.VideoWriter('output.mp4', fourcc, video_fps, (frame.shape[1], frame.shape[0]), True)

        # write the output frame to disk
        writer.write(frame)
        
        #cv2.namedWindow("hasil", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("hasil", 904, 544)
        #cv2.imshow('hasil',frame)
        #cv2.waitKey(1)

        # increase frame index
        frameIndex += 1

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
    main(YOLO())
