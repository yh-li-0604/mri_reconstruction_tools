import os
import numpy as np
from pathlib import Path
# import matlab.engine

# from twix_metadata_def import *
from src.io_utils import *
import einops as eo

def readMeasData_VE11_CCIR_CAPTURE(datFileLocation,
                                #    registered_vars
                                   ):
    # check_mk_dirs(cache_folder)

    registered_vars, current_pos = read_protocol(
        datFileLocation=datFileLocation, which_scan=-1)

    shape_dict = dict(
        line_num=registered_vars['iNoOfFourierLines'], partition_num=registered_vars['lPartitions'], echo_num=registered_vars['NEcoMeas'])

    # first read all mdh header
    scan_meta_date_list, shape_dict = read_scan_meta_data(
        datFileLocation, current_pos, shape_dict)

    nav, kSpaceData = read_navigator_kspace_data(
        datFileLocation, scan_meta_date_list, shape_dict)

    return nav, kSpaceData

# def readMeasData_VE11_CCIR_CAPTURE(datFileLocation):


if __name__ == "__main__":
    data_dir = Path(
        '/data/anlab/PET_MOCO/PETMRdata/CAPTURE_DCE/NO-DCE-002/meas_MID00042_FID44015_CAPTURE_FA15_Dyn.dat')
    readMeasData_VE11_CCIR_CAPTURE(datFileLocation=data_dir )#, cache_folder=Path(
        # '/data/anlab/Chunxu/DL_MOTIF/1_2_CAPTURE/cache'))
