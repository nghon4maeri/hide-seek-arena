/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        arena: {
          bg: "#0b1020",
          panel: "#111827",
          panel2: "#172033",
          line: "#263347",
          text: "#e5e7eb",
          muted: "#94a3b8"
        }
      },
      boxShadow: {
        glow: "0 0 32px rgba(34, 211, 238, 0.18)"
      }
    }
  },
  plugins: []
};

