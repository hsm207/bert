from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
from pathlib import Path
import argparse

tf.enable_eager_execution()


def main(args):
    checkpoint1 = Path(args.checkpoint1)
    checkpoint2 = Path(args.checkpoint2)

    reader1 = pywrap_tensorflow.NewCheckpointReader(str(checkpoint1.resolve()))
    reader2 = pywrap_tensorflow.NewCheckpointReader(str(checkpoint2.resolve()))

    var_to_shape_map1 = reader1.get_variable_to_shape_map()

    diff_tensors = [key for key in var_to_shape_map1
                    if not tf.reduce_all(tf.equal(reader1.get_tensor(key), reader2.get_tensor(key))).numpy()]

    print(f'These are the tensors that differ between {checkpoint1.name} and {checkpoint2.name}:\n')
    for tensor in diff_tensors:
        print(tensor, var_to_shape_map1[tensor])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the weights that differ given two checkpoint files.')
    parser.add_argument('--checkpoint1', required=True, type=str, help='Path to a checkpoint file')
    parser.add_argument('--checkpoint2',
                        required=True,
                        type=str,
                        help='Path to another checkpoint file to compare with checkpoint1')

    args = parser.parse_args()

    main(args)
