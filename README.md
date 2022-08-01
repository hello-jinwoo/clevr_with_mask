## clevr_with_masks
This repository contains the code for generating dataset in [official CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/) format from tfrecords file which is provided by [deepmind's multi_object_datasets repository](https://github.com/deepmind/multi_object_datasets).

## Preparation
- Environment
    - tensorflow (I used 2.4.1 but other versions will be okay)
    - numpy I used 1.19.5 but other versions will be okay)
- download tfrecords [[here]](https://github.com/deepmind/multi_object_datasets)

## Run
```
python load_clevr_with_masks.py --tf_records_path {path_for_tf_records} --target_path {path_for_results}
```

## Output sample
You can see more output samples in sample folder. We generate and save object masks in png form.
<table>
  <tr>
    <th scope="row">Image</th> 
    <td ><center><img src="sample/CLEVR_image_0.png" height="180"> </center></td>
  </tr>
  <tr>
    <th scope="row">Masks</th>
    <td ><center><img src="sample/CLEVR_mask_0_0.png" height="180"> </center></td>
    <td ><center><img src="sample/CLEVR_mask_0_1.png" height="180"> </center></td>
    <td ><center><img src="sample/CLEVR_mask_0_2.png" height="180"> </center></td>
    <td ><center><img src="sample/CLEVR_mask_0_3.png" height="180"> </center></td>
  </tr>
</table>


### Citation
```
@misc{multiobjectdatasets19,
  title={Multi-Object Datasets},
  author={Kabra, Rishabh and Burgess, Chris and Matthey, Loic and
          Kaufman, Raphael Lopez and Greff, Klaus and Reynolds, Malcolm and
          Lerchner, Alexander},
  howpublished={https://github.com/deepmind/multi-object-datasets/},
  year={2019}
}
```
