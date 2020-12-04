import * as tf from "@tensorflow/tfjs";
export function meanSquaredError(a: tf.Tensor1D, b: tf.Tensor1D): number {
  return (
    tf
      .squaredDifference(a, b)
      .sum()
      .asScalar()
      .arraySync() * 0.5
  );
}
