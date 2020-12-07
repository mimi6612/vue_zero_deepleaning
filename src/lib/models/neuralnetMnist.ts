import * as tf from "@tensorflow/tfjs";
import { crossEntropyError } from "@/lib/utils/crossEntropyError";
interface NeuralnetMnistInteterface {
  W1: number[][];
  W2: number[][];
  W3: number[][];
  b1: number[];
  b2: number[];
  b3: number[];
}
export default class NeuralnetMnist {
  W1: tf.Tensor;
  W2: tf.Tensor;
  W3: tf.Tensor;
  b1: tf.Tensor<tf.Rank.R1>;
  b2: tf.Tensor<tf.Rank.R1>;
  b3: tf.Tensor<tf.Rank.R1>;

  constructor({ W1, W2, W3, b1, b2, b3 }: NeuralnetMnistInteterface) {
    this.W1 = tf.tensor(W1);
    this.W2 = tf.tensor(W2);
    this.W3 = tf.tensor(W3);
    this.b1 = tf.tensor(b1);
    this.b2 = tf.tensor(b2);
    this.b3 = tf.tensor(b3);
  }
  predict(x: tf.Tensor): tf.Tensor<tf.Rank.R1> {
    const a1 = x.dot(this.W1).add(this.b1);
    const z1 = a1.sigmoid();
    const a2 = z1.dot(this.W2).add(this.b2);
    const z2 = a2.sigmoid();
    const a3: tf.Tensor<tf.Rank.R1> = z2.dot(this.W3).add(this.b3);
    const z3 = a3.softmax();
    return z3;
  }

  loss(t: tf.Tensor, x: tf.Tensor): number {
    const y = this.predict(x);
    return crossEntropyError(y, t);
  }
}
