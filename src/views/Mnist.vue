<template>
  <div class="mnist">
    lllllhhhhhhkkkkkkkkksssssははtttttqqqqqqqwwwwwww
    {{ mnistTwoLayerNet.predict.value }}
    {{ mnistTwoLayerNet.loss.value }}
    <!-- {{ mnistTwoLayerNet.grad.value }} -->
    {{ neuralnetState.batchPrediction }}
    {{ neuralnetState.loss }}
    {{ crossError }}
    <label>
      Number:
      <input
        ref="input"
        v-model.number="neuralnetState.num"
        type="text"
        @input="reloadImage"
      />
    </label>

    <button type="button" @click="reloadImage">reload</button>
    <canvas ref="canvasRef" width="28" height="28" class="canvas" />
    <p>
      これは{{
        Math.round(
          neuralnetState.probability[neuralnetState.prediction] * 10000
        ) / 100
      }}%の確率で {{ neuralnetState.prediction }}です
    </p>
    <div class="table">
      <table class="u-full-width">
        <thead>
          <tr>
            <th>Number</th>
            <th>Probability</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(probability, index) in neuralnetState.probability"
            :key="index"
          >
            <td>{{ index }}</td>
            <td>{{ Math.round(probability * 10000) / 100 }}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, reactive, onMounted } from "vue";
import * as tf from "@tensorflow/tfjs";

import sampleWeight from "@/assets/sample_weight.json";

const mnist = require("mnist");
import AffineLayer from "@/lib/models/affineLayer";
import SigmoidLayer from "@/lib/models/sigmoidLayer";
import NeuralnetMnist from "@/lib/models/neuralnetMnist";

import mnistBatch from "@/lib/utils/mnistBatch";

import { meanSquaredError } from "@/lib/utils/meanSquaredError";

import { crossEntropyError } from "@/lib/utils/crossEntropyError";

import { useMnistTwoLayerNet } from "@/composables/mnistTwoLayerNet";

export default defineComponent({
  name: "Mnist",
  components: {},
  setup() {
    const affineLayer = new AffineLayer(
      tf.tensor([
        [1, 2],
        [2, 5],
        [3, 2]
      ]),
      tf.tensor([1, 2])
    );
    const a = affineLayer.forward(tf.tensor([3, 4, 5]));
    const sigmoidLayer = new SigmoidLayer();
    const b = sigmoidLayer.forward(a);

    const neuralnetState = reactive<{
      num: number;
      x: number[];
      probability: number[];
      prediction: number;
      batchPrediction: any;
      loss: any;
    }>({
      num: 1,
      x: [],
      probability: [],
      prediction: -1,
      batchPrediction: null,
      loss: null
    });
    const canvasRef = ref<HTMLCanvasElement>();
    const drawImage = (data: number[]) => {
      const context = canvasRef.value?.getContext("2d");
      mnist.draw(data, context);
    };

    const neuralnetMnist = new NeuralnetMnist(sampleWeight);
    const predictMnist = (data: number[]): tf.Tensor<tf.Rank.R1> => {
      const x = tf.tensor(data);
      return neuralnetMnist.predict(x);
    };

    const predicrtMnistBatch = () => {
      const set = mnist.set(30, 10);
      console.error(set);
      const xBatch = tf.tensor(
        set.training.map((t: { input: number[] }) => t.input)
      );
      const tBatch = tf.tensor(
        set.training.map((t: { output: number[] }) => t.output)
      );
      neuralnetState.batchPrediction = neuralnetMnist
        .predict(xBatch)
        .arraySync();
      neuralnetState.loss = neuralnetMnist.loss(xBatch, tBatch);
    };

    const reloadImage = async () => {
      const x = mnist[neuralnetState.num].get();
      drawImage(x);
      const probabilityTensor = predictMnist(x);
      const probability = probabilityTensor.arraySync();
      const prediction = probabilityTensor.argMax().arraySync();
      Object.assign(neuralnetState, { x, probability, prediction });
    };

    onMounted(() => {
      predicrtMnistBatch();
      reloadImage();
    });

    const batches = mnistBatch.getBatch(10);

    const error = meanSquaredError(a.as1D(), b.as1D());

    const crossError = crossEntropyError(
      tf.tensor([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.5, 0.5]
      ]),
      tf.tensor([
        [1, 0],
        [0, 1],
        [1, 0]
      ])
    );

    const mnistTwoLayerNet = useMnistTwoLayerNet();

    return {
      sampleWeight,
      affineLayer,
      a,
      b,
      canvasRef,
      neuralnetState,
      reloadImage,
      batches,
      error,
      crossError,
      mnistTwoLayerNet
    };
  }
});
</script>

<style scoped>
.table {
  margin: 30px;
}
.canvas {
  width: 200px;
}
</style>
