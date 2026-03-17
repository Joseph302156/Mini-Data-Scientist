/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          900: "#050509",
          800: "#0B0B12",
          700: "#14141F",
        },
        accent: {
          500: "#4F46E5",
          400: "#6366F1",
        },
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "BlinkMacSystemFont", "Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};

