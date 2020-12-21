import { createApp } from "vue";
import App from "./App.vue";
import store from "./store";
import router from "./router";

import "milligram/dist/milligram.min.css";

createApp(App)
  .use(router)
  .use(store)
  .mount("#app");
