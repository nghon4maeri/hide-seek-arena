/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Courier New"', '"JetBrains Mono"', "monospace"],
        mono: ['"Courier New"', '"JetBrains Mono"', "monospace"],
      },
    },
  },
  plugins: [],
};
