import { createApp } from "vue";
import App from "./App.vue";
import store from "./store";
import router from "./router";

import "@/assets/Skeleton/css/normalize.css";
import "@/assets/Skeleton/css/skeleton.css";

createApp(App)
  .use(router)
  .use(store)
  .mount("#app");
