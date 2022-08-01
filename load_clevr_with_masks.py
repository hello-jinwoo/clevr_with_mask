# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CLEVR (with masks) dataset reader."""
import sys
import cv2
import tqdm
import os
import json
import argparse
from pathlib import Path

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [240, 320]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--tf_records_path', default='./clevr_with_masks_train.tfrecords')
	parser.add_argument('--date', default='07/29/2022')
	parser.add_argument('--target_path', default='./')
	parser.add_argument('--batch_size', default=100)
	parser.add_argument('--train_size', default=70000)
	parser.add_argument('--val_size', default=15000)

	args = parser.parse_args()

	tf_records_path = args.tf_records_path
	target_path = args.target_path
	batch_size = args.batch_size
	date = args.date
	num_train_batch = args.train_size // args.batch_size
	num_val_batch = args.val_size // args.batch_size

	# make folders to save results
	for folder in ["iamges", "masks", "scenes"]:
		if folder == "scenes":
			Path(os.path.join(target_path, folder)).mkdir(parents=True, exist_ok=True)
		else:
			for split in ["train", "val"]:
				Path(os.path.join(target_path, folder, split)).mkdir(parents=True, exist_ok=True)

	dataset = dataset(tf_records_path)
	batched_dataset = dataset.batch(batch_size)  # optional batching
	iterator = batched_dataset.make_one_shot_iterator()
	# batched_dataset = dataset.batch(batch_size)  # optional batching
	# iterator = batched_dataset.make_initializable_iterator()

	data = iterator.get_next()
	tf.disable_v2_behavior()
	with tf.train.SingularMonitoredSession() as sess:
		# val set
		print("generating val..")
		split = 'val'
		offset = 0
		metadata = {"info": dict(), "scenes": list()}
		metadata['info']['split'] = split
		metadata['info']['date'] = date
		metadata['info']['description'] = "We downloaded clevr_with_mask dataset from https://github.com/deepmind/multi_object_datasets and genenrate this dataset in official CLEVR_v1.0 format. There are some differences in scenes/objects attributes."
		metadata['info']['object_bbox'] = "Only bbox information is made by us. We extract bbox from mask. bbox is generated in [x_start, x_end, y_start, y_end] fashion and each element represents pixel index"
		for i in tqdm.tqdm(range(num_val_batch)):
			d = sess.run(data)

			for img_i in range(batch_size):
				idx = offset + img_i

				# save image
				image = d['image'][img_i]
				idx_str = str(idx)
				idx_str = "0" * (6 - len(idx_str)) + idx_str
				image_name = f"CLEVR_{split}_{idx_str}.png"
				cv2.imwrite(os.path.join(target_path, "images", split, image_name), image)

				# save metadata
				each_data = dict()
				each_data['split'] = split
				each_data['image_filename'] = image_name
				image_index = idx
				each_data['objects'] = []
				for obj_i in range(MAX_NUM_ENTITIES):
					if np.sum(d['mask'][img_i][obj_i]) == 0:
						break
					else:
						each_data['objects'].append(dict())
						mask = d['mask'][img_i][obj_i]
						cv2.imwrite(os.path.join(target_path, "masks", split, f"{image_name.split('.')[0]}_{obj_i}.png"), 
									mask)
						# background
						if obj_i == 0:
							each_data['objects'][obj_i]['bbox'] = [0, 0, 0, 0]
						# objects
						else:
							# make bbox
							x_start = int(np.where(mask==255)[0].min())
							x_end = int(np.where(mask==255)[0].max())
							y_start = int(np.where(mask==255)[1].min())
							y_end = int(np.where(mask==255)[1].max())
							each_data['objects'][obj_i]['bbox'] = [y_start, y_end, x_start, x_end]
					for k, v in d.items():
						if k in ['image', 'mask']:
							continue
						else:
							# type conversion for json
							if 'int' in str(type(v[img_i][obj_i])):
								each_data['objects'][obj_i][k] = int(v[img_i][obj_i])
							if 'float' in str(type(v[img_i][obj_i])):
								each_data['objects'][obj_i][k] = float(v[img_i][obj_i])
							if 'array' in str(type(v[img_i][obj_i])):
								each_data['objects'][obj_i][k] = v[img_i][obj_i].tolist()
				
				if len(list(each_data.keys())) > 0:
					metadata['scenes'].append(each_data)
		
			offset += batch_size

		with open(os.path.join(target_path, "scenes", f"CLEVR_{split}_scenes.json"), "w") as f:
			json.dump(metadata, f)

		# train set
		print("generating train..")
		split = 'train'
		offset = 0
		metadata = {"info": dict(), "scenes": list()}
		metadata['info']['split'] = split
		metadata['info']['date'] = date
		metadata['info']['description'] = "We downloaded clevr_with_mask dataset from https://github.com/deepmind/multi_object_datasets and genenrate this dataset in official CLEVR_v1.0 format. There are some differences in scenes/objects attributes."
		metadata['info']['object_bbox'] = "Only bbox information is made by us. We extract bbox from mask. bbox is generated in [x_start, x_end, y_start, y_end] fashion and each element represents pixel index"
		for i in tqdm.tqdm(range(num_train_batch)):
			d = sess.run(data)

			for img_i in range(batch_size):
				idx = offset + img_i

				# save image
				image = d['image'][img_i]
				idx_str = str(idx)
				idx_str = "0" * (6 - len(idx_str)) + idx_str
				image_name = f"CLEVR_{split}_{idx_str}.png"
				cv2.imwrite(os.path.join(target_path, "images", split, image_name), image)

				# save metadata
				each_data = dict()
				each_data['split'] = split
				each_data['image_filename'] = image_name
				image_index = idx
				each_data['objects'] = []
				for obj_i in range(MAX_NUM_ENTITIES):
					if np.sum(d['mask'][img_i][obj_i]) == 0:
						break
					else:
						each_data['objects'].append(dict())
						mask = d['mask'][img_i][obj_i]
						cv2.imwrite(os.path.join(target_path, "masks", split, f"{image_name.split('.')[0]}_{obj_i}.png"), 
									mask)
						# background
						if obj_i == 0:
							each_data['objects'][obj_i]['bbox'] = [0, 0, 0, 0]
						# objects
						else:
							# make bbox
							x_start = int(np.where(mask==255)[0].min())
							x_end = int(np.where(mask==255)[0].max())
							y_start = int(np.where(mask==255)[1].min())
							y_end = int(np.where(mask==255)[1].max())
							each_data['objects'][obj_i]['bbox'] = [y_start, y_end, x_start, x_end]
					for k, v in d.items():
						if k in ['image', 'mask']:
							continue
						else:
							# type conversion for json
							if 'int' in str(type(v[img_i][obj_i])):
								each_data['objects'][obj_i][k] = int(v[img_i][obj_i])
							if 'float' in str(type(v[img_i][obj_i])):
								each_data['objects'][obj_i][k] = float(v[img_i][obj_i])
							if 'array' in str(type(v[img_i][obj_i])):
								each_data['objects'][obj_i][k] = v[img_i][obj_i].tolist()

				if len(list(each_data.keys())) > 0:
					metadata['scenes'].append(each_data)
		
			offset += batch_size

		with open(os.path.join(target_path, "scenes", f"CLEVR_{split}_scenes.json"), "w") as f:
			json.dump(metadata, f)