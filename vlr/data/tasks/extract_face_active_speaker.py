import os
import sys
sys.path.append("/home/duytran/Desktop/vlr")

import glob
import torch
import subprocess
import argparse
import warnings
import torch.multiprocessing as mp
import tqdm
from dataclasses import dataclass
from logging import getLogger

from vlr.data.processors.active_speaker_extracting import ActiveSpeakerExtracting

logger = getLogger(__name__)
warnings.filterwarnings("ignore")


def args_parser():
    """
    Extract face and active speaker from raw video arguments.
    """

    parser = argparse.ArgumentParser(
        description="Extract face and active speaker from raw video arguments.")
    
    parser.add_argument('--video_folder_path',           type=str,
                        default=None,  help='Path for inputs')
    
    parser.add_argument('--video_extension',           type=str,
                        default="mp4",  help='extension of video files [mp4, avi]')
    
    parser.add_argument('--output_folder',           type=str,
                        default=None,  help='Path for temps, outputs')
    
    parser.add_argument('--pretrain_model',         type=str,
                        default="/home/duytran/Desktop/vlr/vlr/data/utils/TalkNet/pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

    parser.add_argument('--n_datalader_thread',     type=int,
                        default=10,   help='Number of workers')
    
    parser.add_argument('--num_proc',           type=int,
                        default=None,  help='Number of processes')
    
    parser.add_argument('--facedet_scale',          type=float, default=0.25,
                        help='Scale factor for face detection, the frames will be scale to 0.25 orig')
    
    parser.add_argument('--min_track',              type=int,
                        default=10,   help='Number of min frames for each shot')
    
    parser.add_argument('--num_failed_det',          type=int,   default=10,
                        help='Number of missed detections allowed before tracking is stopped')
    
    parser.add_argument('--min_face_size',           type=int,
                        default=1,    help='Minimum face size in pixels')
    
    parser.add_argument('--crop_scale',             type=float,
                        default=0.40, help='Scale bounding box')

    parser.add_argument('--start',                 type=int,
                        default=0,   help='The start time of the video')
    
    parser.add_argument('--extract',                 type=str,
                        default='sample',   help='the way to extract active speaker [sample, origin]')
    
    parser.add_argument('--duration',              type=int, default=0,
                        help='The duration of the video, when set as 0, will extract the whole video')

    args = parser.parse_args()

    if os.path.isfile(args.pretrain_model) == False: # Download the pretrained model
        Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
        cmd = "gdown --id %s -O %s"%(Link, args.pretrain_model)
        subprocess.call(cmd, shell=True, stdout=None)
    
    if args.num_proc is None:
        args.num_proc = os.cpu_count()

    args.activespeaker = ActiveSpeakerExtracting(
        minTrack=args.min_track,
        numFailedDet=args.num_failed_det,
        minFaceSize=args.min_face_size,
        cropScale=args.crop_scale,
        start=args.start,
        duration=args.duration,
        pretrainModel=args.pretrain_model,
        facedetScale=args.facedet_scale,
        nDataLoaderThread=args.n_datalader_thread,
        videoFolderPath=args.video_folder_path,
        outputPath=args.output_folder,
        extract=args.extract,
    )

    return args


def get_dataset(data_dir: str, data_out : str, video_extension: str):

    source = glob.glob(data_dir + "/*" + video_extension)
    source = [os.path.basename(x).split(".")[0] for x in source]
    out = glob.glob(data_out + "/*")
    out = [os.path.basename(x) for x in out]
    unfinished = source.copy()
    for file in source:
        if file + "_finished" in out:
            unfinished.remove(file)
    unfinished = [os.path.join(data_dir, x + "." + video_extension) for x in unfinished]
    return list(unfinished)


def main(args):
    """
    Main function.
    :param args:    arguments.
    """
    # Prepare dataset.
    logger.info("Preparing dataset...")
    un_file_list = get_dataset(args.video_folder_path, args.output_folder, args.video_extension)
    unf_files_num = len(un_file_list)

    if unf_files_num == 0:
        print('Nothing to do.')
        exit(0)
    print('Found {} unprocessed files.'.format(unf_files_num))
    print('Processing with {} processes.'.format(args.num_proc))

    # Count possible number of processes.
    mp.set_start_method('spawn')
    with tqdm.tqdm(total=unf_files_num) as pbar:
        def update(*a):
            pbar.update()

        with torch.multiprocessing.Pool(processes=args.num_proc) as pool:
            for _ in pool.imap_unordered(args.activespeaker.process, un_file_list):
                update()


if __name__ == "__main__":
    args = args_parser()
    main(args)
    
    
    

    