const mnist = require("mnist");
import * as tf from "@tensorflow/tfjs";
import { TwoLayerNet } from "@/lib/models/twoLayerNet";
import { ref } from "vue";

interface MnistTwoLayerNet {}
export function useMnistTwoLayerNet() {
  const net = new TwoLayerNet(784, 10, 10);
  const set = mnist.set(10, 100);
  const xBatch = tf.tensor(
    set.training.map((t: { input: number[] }) => t.input)
  );
  const tBatch = tf.tensor(
    set.training.map((t: { output: number[] }) => t.output)
  );

  const predict = ref(net.predict(xBatch));
  const loss = ref(net.loss(xBatch, tBatch));
  console.error("mi");
  // const grad = ref(net.grad(xBatch, tBatch));
  net.grads(xBatch, tBatch);
  net.accuracy(xBatch, tBatch);
  console.error("te");
  return {
    predict,
    loss
    // grad
  };
}
