# from behave_analysis.utils.directory import Directory
# import os

# def test_directory():
#     base_folder = ".\\sample_data\\trial clips"
#     experiment  = "block edge vectors"
#     dir = Directory(base_folder, experiment=experiment, stim_type='laser', tracking_video=True)
#     file_name = dir.file_name(mouse='mouse', trial_num=1)
#     assert file_name==".\\sample_data\\trial clips\\__tracking__\\laser videos\\mouse-1.mp4"
#     assert dir.path==".\\sample_data\\trial clips\\__tracking__\\laser videos"
#     assert os.path.isdir(dir.path)
#     os.rmdir(dir.path)
#     os.rmdir(".\\sample_data\\trial clips\\__tracking__")

#     dir = Directory(base_folder, experiment=experiment, stim_type='audio', tracking_video=False)
#     file_name = dir.file_name(mouse='mouse', trial_num=1)
#     assert file_name==".\\sample_data\\trial clips\\block edge vectors\\escape videos\\mouse-1.mp4"
#     assert dir.path==".\\sample_data\\trial clips\\block edge vectors\\escape videos"
#     assert os.path.isdir(dir.path)
#     os.rmdir(dir.path)
#     os.rmdir(".\\sample_data\\trial clips\\block edge vectors")
#     os.rmdir(".\\sample_data\\trial clips")

#     base_folder = ".\\sample_data\\statistics"
#     dir = Directory(base_folder, analysis_type='escape targets', plot=True)
#     file_name = dir.file_name(title='test plot', color_by='speed')
#     assert file_name==".\\sample_data\\statistics\\escape targets\\test plot_color=speed.png"
#     assert dir.path==".\\sample_data\\statistics\\escape targets"
#     assert os.path.isdir(dir.path)
#     os.rmdir(dir.path)
#     os.rmdir(".\\sample_data\\statistics")
    


