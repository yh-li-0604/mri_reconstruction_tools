import numpy as np

prisma_header_type = \
np.dtype([
('firstInt'           , '<u4') ,
('numberOfScansInFile', '<u4') ,
('measID'             , '<u4') ,
('fileID'             , '<u4') ,
])

meas_type = \
np.dtype([
('measOffset','<u8'),
('measLength','<u8'),
])

scan_header_type = \
np.dtype([
('DMAlen1'                   , '<u2') ,
('DMAlen2'                   , '<u1') ,
('None'                      , '<u1') ,
('lMeasUID'                  , '<i4') ,
('ulScanCounter'             , '<u4') ,
('ulTimeStamp'               , '<u4') ,
('ulPMUTimeStamp'            , '<u4') ,
('ushSystemType'             , '<u2') ,
('ulPTABPosDelay'            , '<u2') ,
('lPTABPosX'                 , '<i4') ,
('lPTABPosY'                 , '<i4') ,
('lPTABPosZ'                 , '<i4') ,
('ulReserved1'               , '<u4') ,
('aulEvalInfoMask'           , 'V8') ,
#    previous this is read by two 4 bytes unsigned int, 
#   and turn them into raw binary afterward, now we read them all.
# ('aulEvalInfoMaskMostSig'    , '<u4') ,
# ('aulEvalInfoMaskLeastSig'   , '<u4') ,
('ushSamplesInScan'          , '<u2') ,
('ushUsedChannels'           , '<u2') ,
('ushLine'                   , '<u2') ,
('ushAcquisition'            , '<u2') ,
('ushSlice'                  , '<u2') ,
('ushPartition'              , '<u2') ,
('ushEcho'                   , '<u2') ,
('ushPhase'                  , '<u2') ,
('ushRepetition'             , '<u2') ,
('ushSet'                    , '<u2') ,
('ushSeg'                    , '<u2') ,
('ushIda'                    , '<u2') ,
('ushIdb'                    , '<u2') ,
('ushIdc'                    , '<u2') ,
('ushIdd'                    , '<u2') ,
('ushIde'                    , '<u2') ,
('ushPre'                    , '<u2') ,
('ushPost'                   , '<u2') ,
('ushKSpaceCentreColumn'     , '<u2') ,
('ushCoilSelect'             , '<u2') ,
('fReadOutOffcentre'         , '<f4') ,
('ulTimeSinceLastRF'         , '<u4') ,
('ushKSpaceCentreLineNo'     , '<u2') ,
('ushKSpaceCentrePartitionNo', '<u2') ,
('fSag'                      , '<f4') ,
('fCor'                      , '<f4') ,
('fTra'                      , '<f4') ,
('aflQuaternion'             , '<f4', 4 ) ,
('aushIceProgramPara'        , '<u2', 24) ,
('aushReservedPara'          , '<u2', 4 ) ,
('ushApplicationCounter'     , '<u2') ,
('ushApplicationMask'        , '<u2') ,
('ulCRC'                     , '<u4') ,
])

ch_header_type = \
np.dtype([
('ulTypeAndChannelLength', '<u4' ) ,        #  0: (4) bit  0.. 7: type (0x02 => ChannelHeader)                                                                                    , bit  8..31: channel length (header+data) in byte, type   := ulTypeAndChannelLength & 0x000000FF, length := ulTypeAndChannelLength >> 8
( 'lMeasUID'             , '<u4' ) ,      #   4: (4) measurement user ID                                                                                                ,                                                 ,                                              ,
( 'ulScanCounter'        , '<u4' ) ,    #  8: (4) scan counter [1...]                                                                                                     ,                                                 ,                                              ,
( 'ulReserved1'          , '<u4' ) ,    #  12:(4) reserved                                                                                                               ,                                                 ,                                              ,
( 'ulSequenceTime'       , '<u4' ) ,    #  16:(4) Sequence readout starting time bit 31..9 time in [10us]                                       bit  8..0 time in [25ns],                                                 ,                                              ,
( 'ulUnused2'            , '<u4' ) ,     #  20:(4) unused                                                                                                            ,                                                 ,                                              ,
( 'ulChannelId'          , '<u2' ) ,    #   24:(2) unused                                                                                                              ,                                                 ,                                              ,
( 'ulUnused3'            , '<u2' ) ,     #   26:(2) unused                                                                                                           ,                                                 ,                                              ,
( 'ulCRC'                , '<u4' ) ,     #  28:(4) CRC32 checksum of channel header                                                                                 ,                                                 ,                                              ,
])

mdhBitFields = dict(\
        MDH_ACQEND                   = 0,                                                                                                            
        MDH_RTFEEDBACK               = 1,                                                                                                            
        MDH_HPFEEDBACK               = 2,                                                                                                            
        MDH_ONLINE                   = 3,                                                                                                            
        MDH_OFFLINE                  = 4,                                                                                                            
        MDH_SYNCDATA                 = 5,                                                                                                            
        MDH_LASTSCANINCONCAT         = 8,                                                                                                            
        MDH_RAWDATACORRECTION        = 10,                                                                                                            
        MDH_LASTSCANINMEAS           = 11,                                                                                                            
        MDH_SCANSCALEFACTOR          = 12,                                                                                                            
        MDH_2NDHADAMARPULSE          = 13,                                                                                                            
        MDH_REFPHASESTABSCAN         = 14,                                                                                                            
        MDH_PHASESTABSCAN            = 15,                                                                                                            
        MDH_D3FFT                    = 16,                                                                                                            
        MDH_SIGNREV                  = 17,                                                                                                            
        MDH_PHASEFFT                 = 18,                                                                                                            
        MDH_SWAPPED                  = 19,                                                                                                            
        MDH_POSTSHAREDLINE           = 20,                                                                                                            
        MDH_PHASCOR                  = 21,                                                                                                            
        MDH_PATREFSCAN               = 22,                                                                                                            
        MDH_PATREFANDIMASCAN         = 23,                                                                                                            
        MDH_REFLECT                  = 24,                                                                                                            
        MDH_NOISEADJSCAN             = 25,                                                                                                            
        MDH_SHARENOW                 = 26,      # all lines are acquired from the actual and previous e.g. phases                                                      =             ==
        MDH_LASTMEASUREDLINE         = 27,      # indicates that the current line is the last measured line of all succeeding e.g. phases                        =             = = 
        MDH_FIRSTSCANINSLICE         = 28,      # indicates first scan in slice                                                                                  = any(setFlags= = needed for time stamps)
        MDH_LASTSCANINSLICE          = 29,      # indicates  last scan in slice                                                                                  = any(setFlags= = needed for time stamps)
        MDH_TREFFECTIVEBEGIN         = 30,      # indicates the begin time stamp for TReff                                                                       = any(setFlags= = triggered measurement)
        MDH_TREFFECTIVEEND           = 31,      # indicates the   end time stamp for TReff (triggered measurement)                                                 =             = = 
        
        MDH_MDS_REF_POSITION         = 32,      # indicates the reference position for move during scan images (must be set once per slice/partition in MDS mode)=             = = 
        MDH_SLC_AVERAGED             = 33,      # indicates avveraged slice for slice partial averaging scheme                                                   =             = = 
        MDH_TAGFLAG1                 = 34,      # adjust scan                                                                                                    =             = = 
        MDH_CT_NORMALIZE             = 35,      # Marks scans used to calculate correction maps for TimCT-Prescan normalize                                          =             = = 
        MDH_SCAN_FIRST               = 36,      # Marks the first scan of a particular map                                                                           =             = = 
        MDH_SCAN_LAST                = 37,      # Marks the last scan of a particular map                                                                            =             = = 
        MDH_FIRST_SCAN_IN_BLADE      = 40,      # Marks the first line of a blade                                                                                    =             = = 
        MDH_LAST_SCAN_IN_BLADE       = 41,      # Marks the last line of a blade                                                                                     =             = = 
        MDH_LAST_BLADE_IN_TR         = 42,      # Set for all lines of the last BLADE in each TR interval                                                            =             = = 
        MDH_PACE                     = 44,      # Distinguishes PACE scans from non PACE scans.                                                                      =             = = 
        MDH_RETRO_LASTPHASE          = 45,      # Marks the last phase in a heartbeat                                                                                =             = = 
        MDH_RETRO_ENDOFMEAS          = 46,      # Marks an ADC at the end of the measurement                                                                         =             = = 
        MDH_RETRO_REPEATTHISHEARTBEAT= 47,      # Repeat the current heartbeat when this bit is found                                                                =             = = 
        MDH_RETRO_REPEATPREVHEARTBEAT= 48,      # Repeat the previous heartbeat when this bit is found                                                               =             = = 
        MDH_RETRO_ABORTSCANNOW       = 49,      # Just abort everything                                                                                              =             = = 
        MDH_RETRO_LASTHEARTBEAT      = 50,      # This adc is from the last heartbeat (a dummy)                                                                      =             = = 
        MDH_RETRO_DUMMYSCAN          = 51,      # This adc is just a dummy scan, throw it away                                                                       =             = = 
        MDH_RETRO_ARRDETDISABLED     = 52,      # Disable all arrhythmia detection when this bit is found                                                            =             = = 
        MDH_B1_CONTROLLOOP           = 53,      # Marks the readout as to be used for B1 Control Loop                                                                =             = = 
        MDH_SKIP_ONLINE_PHASCOR      = 54,      # Marks scans not to be online phase corrected, even if online phase correction is switched on                       =             = = 
        MDH_SKIP_REGRIDDING          = 55,      # Marks scans not to be regridded, even if regridding is switched on                                                 =             = = 
)