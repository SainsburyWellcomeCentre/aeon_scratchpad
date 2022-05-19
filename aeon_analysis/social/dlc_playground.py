import deeplabcut as dlc

config_file_path = dlc.create_new_project(
    project='',
    experimenter='',
    videos=[],
    working_directory=None,
    copy_videos=False,
    videotype="",
    multianimal=False,
)

dlc.add_new_videos(
    config='',
    videos='',
    copy_videos=False,
    coords=None,
    extract_frames=False)

dlc.extract_frames(
    config='',
    mode="automatic",
    algo="kmeans",
    crop=False,
    userfeedback=False,
    cluster_step=5,
    cluster_resizewidth=30,
    cluster_color=False,
    opencv=True,
    slider_width=25,
    config3d=None,
    extracted_cam=0,
    videos_list=None,
)

dlc.label_frames(
    config='',
    multiple_individuals_GUI=False,
    imtypes=["*.png"],
    config3d=None,
    sourceCam=None,
    jump_unlabeled=False
)

dlc.check_labels(
    config='',
    scale=1,
    dpi=100,
    draw_skeleton=True,
    visualizeindividuals=True)