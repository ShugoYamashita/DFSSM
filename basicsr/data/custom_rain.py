import cv2
cv2.setNumThreads(1)
from os import path as osp

from basicsr.utils import img2tensor, scandir

def paired_paths_from_folder_rainds(folders, keys, filename_tmpl, dataset_type='all'):
    """Generate paired paths from folders.
    For RainDS
    ref: https://github.com/Ephemeral182/UDR-S2Former_deraining/blob/main/dataloader_udr.py

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    gt_list = []
    rain_list = []

    gt_path = osp.join(gt_folder, 'gt')
    raindrop_path = osp.join(input_folder, 'raindrop')
    rainstreak_path = osp.join(input_folder, 'rainstreak')
    streak_drop_path = osp.join(input_folder, 'rainstreak_raindrop')

    raindrop_names = list(scandir(raindrop_path))
    rainstreak_names = list(scandir(rainstreak_path))
    streak_drop_names = list(scandir(streak_drop_path))

    rd_input = []
    rd_gt = []

    rs_input = []
    rs_gt = []

    rd_rs_input=[]
    rd_rs_gt = []

    for name in raindrop_names:
        rd_input.append(osp.join(raindrop_path,name))
        gt_name = name.replace('rd','norain')
        rd_gt.append(osp.join(gt_path,gt_name))

    for name in rainstreak_names:
        rs_input.append(osp.join(rainstreak_path,name))
        gt_name = name.replace('rain','norain')
        rs_gt.append(osp.join(gt_path,gt_name))

    for name in streak_drop_names:
        rd_rs_input.append(osp.join(streak_drop_path,name))
        gt_name = name.replace('rd-rain','norain')
        rd_rs_gt.append(osp.join(gt_path,gt_name))

    if dataset_type=='all':
        gt_list += rd_gt
        rain_list += rd_input
        gt_list += rs_gt
        rain_list += rs_input
        gt_list += rd_rs_gt
        rain_list += rd_rs_input
    elif dataset_type=='rs':
        gt_list += rs_gt
        rain_list += rs_input
    elif dataset_type=='rd':
        gt_list += rd_gt
        rain_list += rd_input
    elif dataset_type=='rsrd':
        gt_list += rd_rs_gt
        rain_list += rd_rs_input

    assert len(rain_list) == len(gt_list), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(rain_list)}, {len(gt_list)}.')

    paths = []
    for idx in range(len(gt_list)):
        paths.append(
            dict([(f'{input_key}_path', rain_list[idx]),
                  (f'{gt_key}_path', gt_list[idx])]))
    return paths

def paired_paths_from_folder_DDN_Data(folders, keys):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].

    Returns:
        list[str]: Returned path list.

    ex) input: 901_2.jpg, gt: 901.jpg
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = []
    for input_path in input_paths:
        filename, ext = osp.splitext(input_path)
        filename = filename.split('_')[0]
        gt_paths.append(filename + ext)

    paths = []
    for idx in range(len(gt_paths)):
        paths.append(
            dict([(f'{input_key}_path', osp.join(input_folder, input_paths[idx])),
                  (f'{gt_key}_path', osp.join(gt_folder, gt_paths[idx]))]))
    return paths

def paired_paths_GT_Rain(folders, keys):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].

    Returns:
        list[str]: Returned path list.

    ex) input: 901_2.jpg, gt: 901.jpg
    """

    input_folder, gt_folder = folders
    input_key, gt_key = keys

    train_list = osp.join(input_folder, 'filename_GT_Rain_train.txt')
    with open(train_list) as f:
        contents = f.readlines()
        input_paths = [i.strip() for i in contents]
        gt_paths = [osp.join(osp.dirname(i), 'gt.png') for i in input_paths]

    paths = []
    for idx in range(len(gt_paths)):
        paths.append(
            dict([(f'{input_key}_path', osp.join(input_folder, input_paths[idx])),
                  (f'{gt_key}_path', osp.join(input_folder, gt_paths[idx]))]))
    return paths

def paired_paths_GT_Rain_val(folders, keys, filename='rain_gtrain_2100.txt'):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].

    Returns:
        list[str]: Returned path list.

    ex) input: 901_2.jpg, gt: 901.jpg
    """

    input_folder, gt_folder = folders
    input_key, gt_key = keys

    train_list = osp.join(input_folder, filename)
    with open(train_list) as f:
        contents = f.readlines()
        input_paths = [i.strip() for i in contents]
        gt_paths = [osp.join(osp.dirname(i), 'gt.png') for i in input_paths]

    paths = []
    for idx in range(len(gt_paths)):
        paths.append(
            dict([(f'{input_key}_path', osp.join(input_folder, input_paths[idx])),
                  (f'{gt_key}_path', osp.join(input_folder, gt_paths[idx]))]))
    return paths
