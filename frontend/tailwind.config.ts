import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        border: "var(--border)",
        "border-bright": "var(--border-bright)",
        accent: "var(--accent)",
        "accent-dim": "var(--accent-dim)",
        tx: "var(--text)",
        muted: "var(--text-muted)",
        dim: "var(--text-dim)",
        danger: "var(--danger)",
        warn: "var(--warn)",
        ok: "var(--success)",
      },
      fontFamily: {
        syne: ["var(--font-syne)", "Syne", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "GeistMono", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
