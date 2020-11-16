module.exports = {
  configureWebpack: {
    module: {
      rules: [
        {
          test: /\.json$/,
          loader: "json-loader",
          type: "javascript/auto"
        }
      ]
    }
  }
};
