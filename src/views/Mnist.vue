<template>
  <div class="perceptron">
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
import AffineLayer from "@/models/affineLayer";
import SigmoidLayer from "@/models/sigmoidLayer";
import NeuralnetMnist from "@/models/neuralnetMnist";

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
    }>({
      num: 1,
      x: [],
      probability: [],
      prediction: -1
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

    const reloadImage = async () => {
      const x = mnist[neuralnetState.num].get();
      drawImage(x);
      const probabilityTensor = predictMnist(x);
      const probability = probabilityTensor.arraySync();
      const prediction = probabilityTensor.argMax().arraySync();
      Object.assign(neuralnetState, { x, probability, prediction });
    };

    onMounted(() => {
      reloadImage();
    });

    return {
      sampleWeight,
      affineLayer,
      a,
      b,
      canvasRef,
      neuralnetState,
      reloadImage
    };
  }
});
</script>

<style scoped>
.table {
  margin: 30px;
}
.canvas {
  width: 100px;
}
</style>
