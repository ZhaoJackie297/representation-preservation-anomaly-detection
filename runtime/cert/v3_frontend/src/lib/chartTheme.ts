export interface ChartTheme {
  green: string;
  red: string;
  amber: string;
  blue: string;
  purple: string;
  grid: string;
  text: string;
  surface: string;
}

export function chartTheme(theme: "light" | "dark"): ChartTheme {
  return theme === "dark"
    ? {
        green: "#71c893",
        red: "#ef8a84",
        amber: "#e3b55e",
        blue: "#80bde2",
        purple: "#b7a4e8",
        grid: "#29372e",
        text: "#9eaaa2",
        surface: "#17201a"
      }
    : {
        green: "#286f4b",
        red: "#a7443f",
        amber: "#a76f1e",
        blue: "#356f93",
        purple: "#6b57a0",
        grid: "#dfe6df",
        text: "#69756d",
        surface: "#ffffff"
      };
}
