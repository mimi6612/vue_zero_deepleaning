import * as tf from "@tensorflow/tfjs";
export function crossEntropyError(
  logits: tf.Tensor,
  labels: tf.Tensor
): tf.Tensor {
  const logitsReshaped = logits.reshape([logits.size]);
  const labelsReshaped = labels.reshape([labels.size]);
  return tf
    .dot(labelsReshaped, logitsReshaped.log())
    .sum()
    .div(-logits.shape[0]);
}
