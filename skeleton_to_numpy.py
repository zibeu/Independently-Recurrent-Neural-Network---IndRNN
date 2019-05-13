import glob
import os
import numpy as np
import csv


SKELETON_DIR = 'nturgb+d_skeletons'
NPY_DIR = 'nturgb_npy'
TRAIN_DS = '_train.csv'
TEST_DS = '_test.csv'

skeleton_files_mask = os.path.join(SKELETON_DIR, '*.skeleton')
skeleton_files = glob.glob(skeleton_files_mask)


max_frame_count = 300
max_joints = 50

full_ds = []

#for idx, file_name in enumerate(skeleton_files[:568]):
for idx, file_name in enumerate(skeleton_files):
    if idx%100 == 0:
        print(idx)
    basename = os.path.basename(file_name)
    name = os.path.splitext(basename)[0]
    label = name.split('A')[1]
    with open(file_name) as f:
        framecount = int(f.readline())

        sequence_frames = []

        for frame in range(framecount):
            body_count = int(f.readline())
            if body_count <= 0 or body_count>2:
                print('continue, no body')
                break
            joints_xyz = []
            for body in range(body_count):
                skeleton_info = f.readline()
                joint_counts = int(f.readline()) #25
                for joint in range(joint_counts):
                    joint_info = f.readline()
                    joint_info_array = joint_info.split()
                    x, y, z = joint_info_array[:3]
                    joint_info_xyz = np.array([float(x), float(y), float(z)])
                    joints_xyz.append(joint_info_xyz)
            pad_joints = max_joints - len(joints_xyz)
            joints_xyz = np.array(joints_xyz)
            joints_xyz = np.pad(joints_xyz, ((0, pad_joints), (0, 0)), mode='constant')
            frame_xyz = np.stack(joints_xyz)
            sequence_frames.append(frame_xyz)
        if len(sequence_frames) > 0:
            file_name = os.path.join(NPY_DIR, name+ '.npy')
            sample = [name+'.npy', int(label)-1]
            full_ds.append(sample)
            np.save(file_name, np.array(sequence_frames))

#train_ds = full_ds[:380]
#test_ds = full_ds[380:]

train_ds = full_ds[:40320]
test_ds = full_ds[40320:]

with open(os.path.join(NPY_DIR, TRAIN_DS), 'w') as train_ds_file:
    writer = csv.writer(train_ds_file, lineterminator='\n')
    writer.writerows(train_ds)

with open(os.path.join(NPY_DIR, TEST_DS), 'w') as test_ds_file:
    writer = csv.writer(test_ds_file, lineterminator='\n')
    writer.writerows(test_ds)
    
