import * as tf from "@tensorflow/tfjs";
import { NumericDataType } from "@tensorflow/tfjs";
import stepFunction from "@/lib/utils/stepFunction";
export default class Perceptron {
  // プロパティは名前: 型
  w1: number;
  w2: number;
  b: number;
  activateFunction: Function;

  constructor(w1: number, w2: number, b: number) {
    this.w1 = w1;
    this.w2 = w2;
    this.b = b;
    this.activateFunction = stepFunction;
  }

  activate(x1: number, x2: number): number {
    const A = tf.tensor([[this.w1, this.w2, this.b]]);
    const x = tf.tensor([[x1], [x2], [1]]);
    const a = tf.matMul(A, x);
    return this.activateFunction(a);
  }
}
