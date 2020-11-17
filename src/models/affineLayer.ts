import * as tf from "@tensorflow/tfjs";
import { NumericDataType, Tensor } from "@tensorflow/tfjs";
export default class AffineLayer {
  // プロパティは名前: 型
  W: Tensor;
  b: Tensor;

  constructor(W: Tensor, b: Tensor) {
    this.W = W;
    this.b = b;
  }

  forward(x: Tensor): Tensor {
    return x.dot(this.W).add(this.b);
  }
}
