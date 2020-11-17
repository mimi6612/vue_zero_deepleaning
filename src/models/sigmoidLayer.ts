import * as tf from "@tensorflow/tfjs";
import {
  getKernelsForBackend,
  NumericDataType,
  Tensor
} from "@tensorflow/tfjs";

export default class SigmoidLayer {
  forward(x: Tensor): Tensor {
    return tf.sigmoid(x);
  }
}
