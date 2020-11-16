<template>
  <div class="perceptron">
    <label>
      Number:
      <input
        ref="input"
        v-model.number="num"
        type="text"
        @input="reloadImage"
      />
    </label>

    <button type="button" @click="reloadImage">reload</button>
    <canvas ref="canvas" width="30" height="30" />
    {{ mnistData }}
    {{ sampleWeight }}
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";

import sampleWeight from "@/assets/sample_weight.json";

const mnist = require("mnist");

export default defineComponent({
  name: "Mnist",
  components: {},
  data: function(): any {
    return {
      num: 1,
      mnistData: [],
      sampleWeight
    };
  },
  mounted() {
    this.mnistData = mnist[this.num].get();
    this.drawData(this.mnistData);
  },
  methods: {
    reloadImage() {
      this.mnistData = mnist[this.num].get();
      this.drawData(this.mnistData);
    },
    drawData(data: [Number]) {
      const context = this.$refs.canvas.getContext("2d");
      mnist.draw(data, context);
    }
  }
});
</script>
