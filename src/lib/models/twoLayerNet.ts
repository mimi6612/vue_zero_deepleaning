import * as tf from "@tensorflow/tfjs";
import { crossEntropyError } from "@/lib/utils/crossEntropyError";
import { Tensor } from "@tensorflow/tfjs";

interface TwoLayerNetParams {
  W1: tf.Tensor;
  W2: tf.Tensor;
  b1: tf.Tensor;
  b2: tf.Tensor;
}

export class TwoLayerNet {
  params: TwoLayerNetParams;

  constructor(
    inputSize: number,
    hiddenSize: number,
    outputSize: number,
    weightInitStd = 0.01
  ) {
    this.params = {
      W1: tf.randomNormal([inputSize, hiddenSize]).mul(weightInitStd),
      b1: tf.zeros([hiddenSize]),
      W2: tf.randomNormal([hiddenSize, outputSize]).mul(weightInitStd),
      b2: tf.zeros([outputSize])
    };
  }

  predict(x: tf.Tensor) {
    const { W1, W2, b1, b2 } = this.params;
    const a1 = x.dot(W1).add(b1);
    const z1 = a1.sigmoid();
    const a2 = z1.dot(W2).add(b2);
    const z2 = a2.softmax();
    return z2;
  }

  loss(x: tf.Tensor, t: tf.Tensor): tf.Tensor {
    const y = this.predict(x);
    return crossEntropyError(y, t);
  }

  accuracy(x: tf.Tensor, t: tf.Tensor): tf.Tensor {
    const y = this.predict(x);
    const accuracy = tf
      .equal(y.argMax(), t.argMax())
      .sum()
      .div(x.shape[0]);
    accuracy.print();
    return accuracy;
  }
  grads(x: tf.Tensor, t: tf.Tensor): any {
    const f = (
      W1: tf.Tensor,
      W2: tf.Tensor,
      b1: tf.Tensor,
      b2: tf.Tensor
    ): tf.Tensor => {
      return this.loss(x, t);
    };
    const g = tf.grads(f);
    return g([this.params.W1, this.params.W2, this.params.b1, this.params.b2]);
  }
}
