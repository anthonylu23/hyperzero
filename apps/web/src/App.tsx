import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import AboutPage from "./pages/AboutPage";
import GamePage from "./pages/GamePage";

// Lazy: the report pulls in recharts (~190 KB), only needed on this route.
const TechnicalReportPage = React.lazy(() => import("./pages/TechnicalReportPage"));

type Theme = "dark" | "light";

/** Router root. Owns theme so it persists across every route. */
export default function App() {
  const [theme, setTheme] = React.useState<Theme>(() => {
    const stored = window.localStorage.getItem("theme");
    return stored === "light" || stored === "dark" ? stored : "dark";
  });

  React.useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = React.useCallback(() => {
    const update = () =>
      setTheme((current) => (current === "dark" ? "light" : "dark"));

    if (document.startViewTransition) {
      document.startViewTransition(update);
    } else {
      update();
    }
  }, []);

  return (
    <Routes>
      <Route
        path="/"
        element={<GamePage theme={theme} onToggleTheme={toggleTheme} />}
      />
      <Route
        path="/about"
        element={<AboutPage theme={theme} onToggleTheme={toggleTheme} />}
      />
      <Route
        path="/technical-report"
        element={
          <React.Suspense fallback={null}>
            <TechnicalReportPage theme={theme} onToggleTheme={toggleTheme} />
          </React.Suspense>
        }
      />
      <Route path="*" element={<Navigate replace to="/" />} />
    </Routes>
  );
}
