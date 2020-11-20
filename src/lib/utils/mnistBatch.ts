const mnist = require("mnist");
export default {
  getBatch(samples: number): [{ input: [number]; output: [number] }] {
    const set = mnist.set(samples, 0);
    return set.training;
  }
};
