from behave_analysis.process.session import NEW_Session
from behave_analysis.process.camera_trigger import find_drop_frames, load_camera_trigger
import cv2
import os
import numpy as np

def check_drop_frames(session: NEW_Session):
        video_object = cv2.VideoCapture(session.video.video_file)
    frame_trigger_onsets_idx = load_camera_trigger(session).frame_trigger_onsets_idx
    num_frames_dropped, index_dropped_frame = find_drop_frames(session, frame_trigger_onsets_idx, for_video_reader=True)
        if not video_object.isOpened(): print("Error opening video file")

        for idx, num in enumerate(num_frames_dropped):
            index = index_dropped_frame[idx]
            while num >= 0:
                index_dropped_frame = np.append(index_dropped_frame, index+1)
                index+=1
                num-=1
        
        index_dropped_frame.sort()
        # test_list = [1,20,40]
        print(session.video.fps)
        
        for frame_seq in index_dropped_frame:
            video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_seq)
            ret, frame = video_object.read()
            cv2.imshow(f'Frame {frame_seq}', frame)
            key = cv2.waitKey(0)
            if key == ord('n'): continue

        video_object.release()
        cv2.destroyAllWindows()