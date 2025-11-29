export default {
  expo: {
    name: "Aevorium",
    slug: "aevorium",
    version: "1.0.0",
    orientation: "portrait",
    userInterfaceStyle: "dark",
    newArchEnabled: false,
    splash: {
      backgroundColor: "#0e1117"
    },
    assetBundlePatterns: ["**/*"],
    ios: {
      supportsTablet: true
    },
    android: {
      adaptiveIcon: {
        backgroundColor: "#0e1117"
      }
    },
    web: {
      bundler: "metro"
    },
    extra: {
      apiUrl: "http://10.175.165.92:8000"
    }
  }
};
