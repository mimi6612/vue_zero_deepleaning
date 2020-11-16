import * as tf from "@tensorflow/tfjs";
import { NumericDataType, Tensor } from "@tensorflow/tfjs";
import stepFunction from "@/lib/stepFunction";
export default class AffineLayer {
  // プロパティは名前: 型
  W: Tensor;
  b: Tensor;

  constructor(W: Tensor, b: Tensor) {
    this.W = W;
    this.b = b;
  }

  forward(x: Tensor): Tensor {
    return this.W.dot(x).add(this.b);
  }
}
