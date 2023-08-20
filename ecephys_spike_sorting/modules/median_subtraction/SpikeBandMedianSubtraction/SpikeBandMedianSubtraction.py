import sys
import os
import json
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Not enough input arguments.")
        print("Usage: " + sys.argv[0] + " [json file] [data file] [max channel]")
        exit()
    json_file = sys.argv[1]
    data_file = sys.argv[2]
    max_channel = int(sys.argv[3])

    # global configs
    residuals_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "residuals.dat")
    total_channels = 384
    buffer_samples = 1
    sample_rate = 30000

    # Read json file config
    with open(json_file, "r") as f:
        config_json = json.load(f)
    # print(config_json)
    scaling = np.array(config_json['scaling'])
    config_mask = np.array(config_json['mask'])
    channel = np.array(config_json['channel'])
    offset = np.array(config_json['offset'])
    assert len(scaling) == total_channels and len(config_mask) == total_channels \
            and len(channel) == total_channels and len(offset) == total_channels

    print("Creating file in put and output stream")
    # We use np.memmap to directly create an modifible memory-mapped array on the data file
    # TODO: change the filename here
    inputFileMmap = np.memmap("continuous_raw.dat", dtype = "int16", mode = "r")
    inputFileMmap = inputFileMmap.reshape((-1, total_channels))
    outputFileMmap = np.memmap("continuous_ans.dat", dtype = "int16", mode = "w+", shape = inputFileMmap.shape)
    print(inputFileMmap[0, :10])
    print(inputFileMmap.shape)
    file_size = inputFileMmap.nbytes
    sample_size = file_size // total_channels // 2
    print("File size in bytes:", file_size)
    print("File size in samples:", sample_size)
    print("File size in seconds:", sample_size / sample_rate)

    # prepare another mmap array for storing the residule. This can be used to recover the raw data
    median_group_size = 24
    residualsFileMmap = np.memmap(residuals_file, dtype = "int16", mode = "w+", shape = (sample_size, median_group_size))
    scaling = scaling.reshape(1, -1, median_group_size)
    config_mask = config_mask.reshape(1, -1, median_group_size)
    channel = channel.reshape(1, -1, median_group_size)
    offset = offset.reshape(1, -1, median_group_size)
    max_channel_mask = np.arange(total_channels) < max_channel

    # start calculation
    start_time = time.time()

    # sample_chunksize = 100000 # set upper limit for memory consumption
    sample_chunksize = 10000 # set upper limit for memory consumption

    for chunk_id in range((sample_size - 1)// sample_chunksize + 1):
        chunk_sample_start = chunk_id * sample_chunksize
        chunk_sample_stop = (chunk_id + 1) * sample_chunksize
        if chunk_sample_stop > sample_size:
            chunk_sample_stop = sample_size
        chunk_sample_size = chunk_sample_stop - chunk_sample_start

        print("Index: ", chunk_sample_start, chunk_sample_stop, chunk_sample_size)

        input_subtr_offset = inputFileMmap[chunk_sample_start:chunk_sample_stop].reshape(chunk_sample_size, total_channels // median_group_size, median_group_size)
        input_subtr_offset = input_subtr_offset - offset

        # chunk_residuals = np.nanmedian(input_subtr_offset[:, config_mask], axis = -1) # (chunk_sample_size, median_group_size)

        # TODO: deal with probe config_mask
        chunk_residuals = np.nanmedian(input_subtr_offset, axis = 1) # (chunk_sample_size, median_group_size)
        residualsFileMmap[chunk_sample_start: chunk_sample_stop] = chunk_residuals

        input_subtr_offset = input_subtr_offset - (chunk_residuals[:, np.newaxis, :].astype("float") * scaling).astype("int16")
        input_subtr_offset = input_subtr_offset.reshape(-1, total_channels)
        outputFileMmap[chunk_sample_start:chunk_sample_stop, max_channel_mask] = input_subtr_offset[:, max_channel_mask]

    print(residualsFileMmap.shape)
    residualsFileMmap.flush()
    outputFileMmap.flush()
    end_time = time.time()
    print("Total processing time:", end_time - start_time, "seconds")
  