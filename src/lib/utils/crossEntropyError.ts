import * as tf from "@tensorflow/tfjs";
export function crossEntropyError(
  logits: tf.Tensor,
  labels: tf.Tensor
): number {
  const logitsReshaped = logits.reshape([logits.size]);
  const labelsReshaped = labels.reshape([labels.size]);
  -tf.dot(labelsReshaped, logitsReshaped.log()).print();
  return (
    -tf
      .dot(labelsReshaped, logitsReshaped.log())
      .sum()
      .arraySync() / logits.shape[0]
  );
}
