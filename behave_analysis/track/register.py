from behave_analysis.process.session import NEW_Session
from settings.settings_process import settings_process
import sys
import cv2
import numpy as np


def correct_and_register_frame(frame: object, video: object, fisheye_correction_map: tuple, skip_fisheye_correction: bool=False, skip_registration: bool=False) -> object:
    if fisheye_correction_map and not skip_fisheye_correction:
        frame = cv2.copyMakeBorder(frame, video.y_offset, int((fisheye_correction_map[0].shape[0] - frame.shape[0]) - video.y_offset), video.x_offset, int((fisheye_correction_map[0].shape[1] - frame.shape[1]) - video.x_offset), cv2.BORDER_CONSTANT, value=0)
        frame = cv2.remap(frame, fisheye_correction_map[0], fisheye_correction_map[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        frame = frame[video.y_offset:video.height + video.y_offset, video.x_offset:video.width + video.x_offset]
    if isinstance(video.registration_transform, np.ndarray) and not skip_registration:
        if 'affine' in video.registration_type:
            frame = cv2.warpAffine(frame, video.registration_transform, frame.shape[0:2])
        if 'homography' in video.registration_type:
            frame = cv2.warpPerspective(frame, video.registration_transform, frame.shape[0:2])
    return frame.astype(np.uint8)

def load_fisheye_correction_map(video: object):
    if video.fisheye_correction_file:
        fisheye_correction = np.load(video.fisheye_correction_file);
        fisheye_correction_map = (fisheye_correction[:, :, 0:2], fisheye_correction[:, :, 2] * 0)
    else:
        fisheye_correction_map = None
    return fisheye_correction_map
    
def generate_rendered_arena(session: NEW_Session, size: int) -> object:
    rendered_arena = 255 * np.ones(size).astype(np.uint8)
    #! This section must be modified with a new section for each type of arena (default: 92-cm circle with a square shelter and a 50cmx10cm removable rectangle in the middle)
    if "place preference" in session.experiment:
        pass #TODO: make place preference arena
    else:
        cv2.rectangle(rendered_arena, (int(size[0]/2 - 250), int(size[1]/2 - 50)), (int(size[0]/2 + 250), int(size[1]/2 + 50)), 190, thickness=1) # rectangle in center
        cv2.rectangle(rendered_arena, (int(size[0]/2 - 50), int(size[1]/2 + 458)), (int(size[0]/2 + 50), int(size[1]/2 + 360)), 210, thickness=-1) # the shelter
        cv2.circle(rendered_arena, (int(size[0]/2), int(size[1]/2)), 460, 0, 1, lineType = 16) # arena outline
        click_targets = np.array(([size[0]/2 - 250, size[1]/2 - 50], [size[0]/2 - 250, size[1]/2 + 50], [size[0]/2 + 250, size[1]/2 + 50], [size[0]/2 + 250, size[1]/2 - 50])).astype(int)
    return rendered_arena, click_targets

class Register():
    def __init__(self, session: NEW_Session, video: object, video_object: object) -> object:
        self.generate_rendered_arena(session)
        self.add_click_targets()
        self.get_image_of_actual_arena(video_object, (video.num_frames * 2 / 3))
        self.perform_fisheye_correction(video)
        self.initialize_transform()
        self.refine_transform()
        self.get_shelter_position(session,video_object,video.fps)
        self.get_barrier_position(session,video_object,video.fps)

# ----MAIN FUNCTIONS--------------------------------------------------------------------
    def generate_rendered_arena(self, session: NEW_Session):
        self.rendered_arena, self.click_targets = generate_rendered_arena(session, settings_process.size)

    def add_click_targets(self):
        for i, click_target in enumerate(self.click_targets):
            self.rendered_arena_with_click_targets = cv2.circle(self.rendered_arena, (click_target[0], click_target[1]), 3, 255, -1)
            self.rendered_arena_with_click_targets = cv2.circle(self.rendered_arena, (click_target[0], click_target[1]), 4, 0, 1)
            self.rendered_arena_with_click_targets = cv2.putText(self.rendered_arena, str(i+1), tuple(click_target), 0, 1.0, 100, thickness=2)

    def get_image_of_actual_arena(self, video_object: object, frame):
        video_object.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, self.actual_arena = video_object.read()

    def perform_fisheye_correction(self, video: object):
        self.fisheye_correction_map = load_fisheye_correction_map(video)
        self.actual_arena = correct_and_register_frame(self.actual_arena[:, :, 0], video, self.fisheye_correction_map, skip_registration=True)

    def initialize_transform(self):
        print("\n{}REGISTRATION ({})\n\nStep 1: Click the points in the actual arena corresponding to the numbered dots on the rendered arena -- in order!".format(' '*20, settings_process.registration))
        cv2.namedWindow('rendered arena')
        cv2.imshow('rendered arena', self.rendered_arena)
        cv2.startWindowThread()
        cv2.namedWindow('actual arena')
        self.actual_clicked_points = []
        cv2.setMouseCallback('actual arena', self.alignment_click_targets)
        while True:
            cv2.imshow('actual arena', self.actual_arena)
            if len(self.actual_clicked_points) == len(self.click_targets): break # once all points are clicked
            key = cv2.waitKey(10)
            if key == ord('q'): print('quit.'); sys.exit()
        cv2.destroyAllWindows()

    def refine_transform(self):
        print('Step 2: In the overlay, left click the rendered arena and then right click the corresponding location on the actual arena. \nStep 3: Press space bar when ur satisfied or press q to quit.\n') 
        cv2.namedWindow('overlay')
        cv2.setMouseCallback('overlay', self.click_additional_click_targets, self)
        while True:
            if len(self.actual_clicked_points) == len(self.click_targets) and self.time_to_update:
                if settings_process.registration=='partial affine':
                    self.transform = cv2.estimateAffinePartial2D(np.array(self.actual_clicked_points), self.click_targets, method=cv2.RANSAC, maxIters=6000, confidence=0.995, refineIters=20)[0]
                    actual_arena_registered = cv2.warpAffine(self.actual_arena, self.transform, self.actual_arena.shape[::-1])

                if settings_process.registration=='affine':
                    self.transform = cv2.estimateAffine2D(np.array(self.actual_clicked_points), self.click_targets, method=cv2.RANSAC, maxIters=6000, confidence=0.995, refineIters=20)[0]
                    actual_arena_registered = cv2.warpAffine(self.actual_arena, self.transform, self.actual_arena.shape[::-1])

                if settings_process.registration=='homography':
                    self.transform = cv2.findHomography(np.array(self.actual_clicked_points), self.click_targets, method=cv2.LMEDS, maxIters=12000, confidence=0.995)[0]
                    actual_arena_registered = cv2.warpPerspective(self.actual_arena, self.transform, self.actual_arena.shape[::-1])

                self.overlay_of_arenas = cv2.addWeighted(actual_arena_registered, 0.7, self.rendered_arena, 0.3, 0)
                self.time_to_update = False
            cv2.imshow('overlay', self.overlay_of_arenas)
            key = cv2.waitKey(10)
            if key==ord('q'): print('quit.'); sys.exit()
            if key==ord(' '): break
        cv2.destroyAllWindows()

    def get_shelter_position(self,session,video_object,fps):
        """ Ask user where shelter and barrier were positioned"""

        self.actual_arena = []
        self.get_image_of_actual_arena(video_object, (session.shelter_only_time[0]+10)*60*fps)
        self.clicked_points = []
        # ask user where the shelter is
        print("Where is the shelter? Click first the top left, then the bottom right corner of the shelter")
        cv2.namedWindow('where is shelter')
        cv2.setMouseCallback('where is shelter', self.position_click_targets)
        while True:
            cv2.imshow('where is shelter', self.actual_arena)
            if len(self.clicked_points) == 2: break # once both points are clicked
            key = cv2.waitKey(10)
            if key == ord('q'): print('quit.'); sys.exit()
        cv2.destroyAllWindows()
        
        session.shelter_location = self.clicked_points

    def get_barrier_position(self,session,video_object,fps):
        """ Ask user where shelter and barrier were positioned"""

        # ask user where the barrier is
        if len(session.barrier_time) > 0:
            self.clicked_points = []
            self.actual_arena = []
            self.get_image_of_actual_arena(video_object, (session.barrier_time[0]+10)*60*fps)
            print("Where is the barrier? Click the first edge")
            cv2.namedWindow('where is barrier')
            cv2.setMouseCallback('where is barrier', self.position_click_targets)
            while True:
                cv2.imshow('where is barrier', self.actual_arena)
                if len(self.clicked_points) == 1: break # once both points are clicked
                key = cv2.waitKey(10)
                if key == ord('q'): print('quit.'); sys.exit()
            cv2.destroyAllWindows()
            
            # get a different frame if barrier flip
            if session.barrier_flip_time:
                self.get_image_of_actual_arena(video_object, (session.barrier_flip_time+10)*60*fps)
            print("Where is the barrier? Click the second edge")
            cv2.namedWindow('where is barrier')
            cv2.setMouseCallback('where is barrier', self.position_click_targets)
            while True:
                cv2.imshow('where is barrier', self.actual_arena)
                if len(self.clicked_points) == 2: break # once both points are clicked
                key = cv2.waitKey(10)
                if key == ord('q'): print('quit.'); sys.exit()
            cv2.destroyAllWindows()

            # calculate center of barrier
            barrier_center = [np.mean([self.clicked_points[0][0],self.clicked_points[1][0]]).astype(int),
                                       np.mean([self.clicked_points[0][1],self.clicked_points[1][1]]).astype(int)]
            self.clicked_points.append(barrier_center)
            session.barrier_location = self.clicked_points

# ----CLICK CALLBACK FUNCTIONS-------------------------------------------------------------

    def alignment_click_targets(self, event,x,y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.actual_arena = cv2.circle(self.actual_arena, (x, y), 3, 255, -1)
            self.actual_arena = cv2.circle(self.actual_arena, (x, y), 4, 0, 1)
            self.actual_clicked_points.append([x,y])
            self.time_to_update = True
    
    def position_click_targets(self, event,x,y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.actual_arena = cv2.circle(self.actual_arena, (x, y), 3, 255, -1)
            self.clicked_points.append([x,y])

    def click_additional_click_targets(self, event,x,y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN: # click on the rendered arena within the overlay
            cv2.circle(self.overlay_of_arenas, (x, y), 3, 0, -1)
            cv2.circle(self.overlay_of_arenas, (x, y), 4, 255, 1)
            self.click_targets = np.concatenate((self.click_targets, np.reshape(np.array([x, y]),(1,2))))
        elif event == cv2.EVENT_RBUTTONDOWN: # click on the actual arena within the overlay
            cv2.circle(self.overlay_of_arenas, (x, y), 3, 255, -1)
            cv2.circle(self.overlay_of_arenas, (x, y), 4, 0, 1)
            clicked_point = np.array([np.array([[x, y]], dtype='float32')])
            if 'affine' in settings_process.registration:
                inverse_transform = cv2.invertAffineTransform(self.transform)    
                click_in_actual_arena_coordinates = list(cv2.transform(clicked_point, inverse_transform)[0][0].astype(int))
            if 'homography' in settings_process.registration:
                inverse_transform = cv2.invert(self.transform)[1]
                click_in_actual_arena_coordinates = list(cv2.perspectiveTransform(clicked_point, inverse_transform)[0][0].astype(int))
            self.actual_clicked_points.append(click_in_actual_arena_coordinates)
        self.time_to_update = True