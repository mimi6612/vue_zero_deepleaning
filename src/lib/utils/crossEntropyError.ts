import * as tf from "@tensorflow/tfjs";
export function crossEntropyError(
  labels: tf.Tensor1D,
  logits: tf.Tensor1D
): number {
  return (
    tf
      .dot(labels, logits.log())
      .asScalar()
      .arraySync() * -1.0
  );
}
