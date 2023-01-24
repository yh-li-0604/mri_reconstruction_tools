import os
import pathlib
import numpy as np
from pathlib import Path
import matlab.engine
from joblib import Parallel, delayed
import multiprocessing

firstSlice = 1
lastSlice = 150
nPhases = 5  # 5 respiratory phases
# sec. This was communicated to the MRI technologists. Should be correct except the very first few cases (which were probably on VB20)
injectionTime = 30
# 340 Total time to reconstruct. Includes 1 contrast ending at the injection time.
durationToReconstruct = 340
timePerContrast = 10  # 10sec
lambdaFactorInterPhase = 0  # Smoothness across respiratory phases
lambdaFactorInterContrast = 0.05  # Smoothness across contrasts
lambdaFactorTGV_1stDerivative = 0.00375  # Spatial smoothness on derivatives
bComputeSensitivities = False
percentW = 12.5  # Percent for Hamming apodization

data_dir = Path('/data/anlab/PET_MOCO/PETMRdata/CAPTURE_DCE')
folder_list = list(data_dir.glob('NO-DCE-*'))

dat_file_paths = []
object_ids = []
engines = []
for folder in folder_list:
    p = list(folder.glob('*_CAPTURE_FA15_Dyn.dat'))[0]
    object_id = p.parts[-2]
    object_ids.append(object_id)
    dat_file_paths.append(p)
    # engines.append(matlab.engine.start_matlab())
futures = []

print('Dat files to be reconstructed:', object_ids)
num_cores = multiprocessing.cpu_count()
usage = int(num_cores*0.8)
def MCNUFFT_reconstruction(path,eng,object_id):
    print("start computing", object_id)
    # if object_id == "NO-DCE-002" or object_id == "NO-DCE-003":
    # future = eng.NUFFT_processCAPTURE_VarW_NQM_DCE_PostInj_Vida_Chunxu(str(path), firstSlice, lastSlice, nPhases, injectionTime, durationToReconstruct, timePerContrast,
    #                                                                         lambdaFactorInterPhase, lambdaFactorInterContrast, lambdaFactorTGV_1stDerivative, bComputeSensitivities, percentW, nargout=7, background=True)
    # else:
    future = eng.NUFFT_processCAPTURE_VarW_NQM_DCE_PostInj_Prisma_Chunxu(str(path), firstSlice, lastSlice, nPhases, injectionTime, durationToReconstruct, timePerContrast,
                                                                        lambdaFactorInterPhase, lambdaFactorInterContrast, lambdaFactorTGV_1stDerivative, bComputeSensitivities, percentW, nargout=7, background=True)
    output = future.result()
    recon_MCNUFFT = output[0]
    output_Filename = object_id+output[1]+'.h5'
    print("start saving",object_id)
    eng.h5create(Path('results') / output_Filename, '/subjects', eng.size(recon_MCNUFFT),nargout=0)
    eng.h5write(Path('results') / output_Filename, '/subjects', recon_MCNUFFT,nargout=0)


print('Cpu Cores:', num_cores, 'Using:', usage)
# Parallel(n_jobs=usage)(delayed(MCNUFFT_reconstruction)(path,eng) for path, eng in zip(dat_file_paths, engines))
for path, eng in zip(dat_file_paths, engines):
    future = eng.NUFFT_processCAPTURE_VarW_NQM_DCE_PostInj_Prisma_Chunxu(str(path), firstSlice, lastSlice, nPhases, injectionTime, durationToReconstruct, timePerContrast,
     lambdaFactorInterPhase, lambdaFactorInterContrast, lambdaFactorTGV_1stDerivative, bComputeSensitivities, percentW, nargout=7, background=True)
    futures.append(future)

# for future in futures:                                                                           
#     output = future.result()
#     recon_MCNUFFT = output[0]
#     output_Filename = object_id+output[1]+'.h5'
#     print("start saving",object_id)
#     eng.h5create(str(Path('results') / output_Filename), '/subjects', eng.size(recon_MCNUFFT),nargout=0, background=True)
#     eng.h5write(str(Path('results') / output_Filename), '/subjects', recon_MCNUFFT,nargout=0, background=True)
if __name__ == "__main__":
    MCNUFFT_reconstruction(dat_file_paths[0],matlab.engine.start_matlab(),object_ids[0])