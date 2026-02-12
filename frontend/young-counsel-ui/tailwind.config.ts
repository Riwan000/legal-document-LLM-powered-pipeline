import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#05070b",
        "background-elevated": "#12141c",
        "background-elevated-soft": "#181b24",
        border: "#2a2f3c",
        accent: {
          DEFAULT: "#d2a559",
          soft: "#f3d8a0"
        },
        text: {
          DEFAULT: "#f5f5f5",
          muted: "#a0a5b8",
          subtle: "#6b7184"
        },
        risk: {
          high: "#f97373",
          medium: "#facc60",
          low: "#4ade80"
        },
        "failed-muted": "rgba(185, 28, 28, 0.6)"
      },
      boxShadow: {
        card: "0 18px 45px rgba(0,0,0,0.45)"
      },
      borderRadius: {
        xl: "0.9rem"
      }
    }
  },
  plugins: []
};

export default config;

