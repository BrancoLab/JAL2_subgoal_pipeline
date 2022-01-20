from behave_analysis.utils.settings_objects import Settings_process

settings_process = Settings_process(

    create_new_registration = False,
    skip_registration = False,

    registration = 'homography', # 'affine' 'partial affine' or 'homography' (use the least complex needed)
    fisheye_correction_file = ".\\sample_data\\fisheye_maps.npy", # remove setting if n/a
    size = (1024,1024), # (width, height) how big to make the renderings, in pixels
    pixels_per_cm = 10, # for the arena drawn in register.generate_rendered_arena, report here the ratio between size of arena in pixels and actual size in cm

    by_experiment=False,
    experiments = ['block pre edge vectors'],

    by_session=True,
    sessions=[1,4,5],

    all_sessions=False

)
