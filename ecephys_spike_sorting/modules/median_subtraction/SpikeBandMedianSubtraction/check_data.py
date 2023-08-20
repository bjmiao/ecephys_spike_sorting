import numpy as np
import matplotlib.pyplot as plt

inputFileMmap_raw = np.memmap("continuous_raw.dat", dtype = "int16", mode = "r")
inputFileMmap_raw = inputFileMmap_raw.reshape(-1, 384)
print(inputFileMmap_raw.shape)


inputFileMmap_ref = np.memmap("continuous_ref.dat", dtype = "int16", mode = "r")
inputFileMmap_ref = inputFileMmap_ref.reshape(-1, 384)
print(inputFileMmap_ref.shape)

inputFileMmap_myresult = np.memmap("continuous_ans.dat", dtype="int16", mode = "r")
inputFileMmap_myresult = inputFileMmap_myresult.reshape(-1, 384)
print(inputFileMmap_myresult.shape)

# inputFileMmap_raw = np.memmap("continuous_raw.dat", dtype = "int16", mode = "r")
# inputFileMmap_raw = inputFileMmap_raw.reshape(-1, 384)
# print(inputFileMmap_raw[0])

residual_myanswer = np.memmap("residuals.dat", dtype = "int16", mode = "r")
residual_myanswer = residual_myanswer.reshape(-1, 24)
print(residual_myanswer.shape)


residual_raw = np.memmap("residuals_ref.dat", dtype = "int16", mode = "r")
residual_raw = residual_raw.reshape(-1, 24)
print(residual_raw.shape)

plt.subplot(131)
plt.plot(inputFileMmap_ref[:1000, 0], label = "C++ version")
plt.plot(inputFileMmap_myresult[:1000, 0], label = "My version")
plt.plot(inputFileMmap_raw[:1000, 0], label = "Raw")
plt.legend()
plt.title("Signal")

plt.subplot(132)
plt.plot(residual_raw[10:1000, 0], label = "C++ version")
plt.plot(residual_myanswer[10:1000, 0], label = "My version")
plt.legend()
plt.title("Residual")

# Read json file config
import json
json_file = "example_probe_info.json"
total_channels = 384
with open(json_file, "r") as f:
    config_json = json.load(f)
# print(config_json)
scaling = np.array(config_json['scaling'])
config_mask = np.array(config_json['mask'])
channel = np.array(config_json['channel'])
offset = np.array(config_json['offset'])
assert len(scaling) == total_channels and len(config_mask) == total_channels \
        and len(channel) == total_channels and len(offset) == total_channels

first_1000_samples = inputFileMmap_raw[:1000].copy()
first_1000_samples = first_1000_samples - offset.reshape(1, -1)
 
#  input_subtr_offset - (chunk_residuals[:, :, np.newaxis].astype("float") * scaling.reshape(1, median_group_size, -1)).astype("int16")

# version 1
v1 = np.median(first_1000_samples.reshape(1000, 24, -1), axis = 1)
# version 2
v2 = np.median(first_1000_samples.reshape(1000, -1, 24), axis = 2)

plt.subplot(133)
plt.plot(v1[:, 0], label = "median axis=1")
plt.plot(v2[:, 0], label = "median axis=2")
plt.legend()
plt.title("Two version of median")

plt.show()