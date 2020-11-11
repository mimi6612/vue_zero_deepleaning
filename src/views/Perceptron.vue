<template>
  <div class="perceptron">
    perceptron yeah!
    {{ perceptron }}
    <input v-model.number="x1" type="text" @input="activate" />
    <input v-model.number="x2" type="text" @input="activate" />
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import Perceptron from "@/models/perceptron";

export default defineComponent({
  name: "Perceptron",
  components: {},
  data: function(): any {
    return {
      perceptron: new Perceptron(1, 2, 3),
      x1: null,
      x2: null
    };
  },
  methods: {
    activate() {
      console.error(this.perceptron.activate(this.x1, this.x2));
    }
  }
});
</script>
