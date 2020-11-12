<template>
  <div class="perceptron">
    <label>
      数字:
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
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";

const mnist = require("mnist");

export default defineComponent({
  name: "Mnist",
  components: {},
  data: function(): any {
    return {
      num: 1,
      mnistData: []
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
      console.error(data);
      console.error(this.$refs);
      const context = this.$refs.canvas.getContext("2d");
      mnist.draw(data, context);
    }
  }
});
</script>
