import { Moon, Sun } from "lucide-react";
import { Link } from "react-router-dom";

import GlassSurface from "../components/GlassSurface";
import NavMenu from "../components/NavMenu";

type Theme = "dark" | "light";

const VARIANTS = [
  { tag: "2D", title: "6 × 7 · Connect 4", blurb: "Classic Connect Four — the validation benchmark." },
  { tag: "3D", title: "4 × 4 × 4 · Connect 4", blurb: "The main higher-dimensional target. Near-perfect play." },
  { tag: "4D", title: "4 × 4 × 4 × 4 · Connect 4", blurb: "The stretch variant — 64 columns, dense threats." },
];

const HOW_TO_PLAY = [
  "Pick a dimensionality (2D / 3D / 4D), then set each board axis length and K, the number in a row needed to win.",
  "You play X and move first; the universal agent answers as O. Configuration locks once the first piece drops — Restart to change it.",
  "Drop a piece into any legal column; gravity pulls it to the lowest open cell along the gravity axis.",
];

export default function AboutPage({
  theme,
  onToggleTheme,
}: {
  theme: Theme;
  onToggleTheme: () => void;
}) {
  return (
    <>
      <header className="glass-nav-wrap">
        <GlassSurface
          width="100%"
          height="auto"
          borderRadius={60}
          backgroundOpacity={0.15}
          opacity={0.55}
          blur={10}
          className="glass-nav"
        >
          <nav className="glass-nav-inner" aria-label="Primary">
            <Link className="nav-brand nav-underline" to="/">
              HyperZero
            </Link>
            <div className="nav-controls">
              <NavMenu />
              <button
                aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
                aria-pressed={theme === "dark"}
                className="theme-toggle"
                onClick={onToggleTheme}
                type="button"
              >
                {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
              </button>
            </div>
          </nav>
        </GlassSurface>
      </header>

      <main className="page-shell content-page">
        <div className="content-shell animate-fade-up">
          <section className="about-hero">
            <p className="about-eyebrow">About the project</p>
            <h1 className="about-title">HyperZero</h1>
            <p className="about-lead">
              AlphaZero-style self-play for <strong>N-dimensional Connect-K</strong>.
              One configurable engine generates a whole family of gravity-based
              Connect games, and a single neural agent — guided by Monte Carlo
              Tree Search — learns to play them across 2D, 3D, and 4D.
            </p>
          </section>

          <GlassSurface
            width="100%"
            height="auto"
            borderRadius={22}
            backgroundOpacity={0.12}
            opacity={0.5}
            blur={10}
            className="about-card animate-fade-up animate-delay-2"
          >
            <div className="about-card-body">
              <h2 className="about-h2">How to play</h2>
              <ol className="about-steps">
                {HOW_TO_PLAY.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            </div>
          </GlassSurface>

          <GlassSurface
            width="100%"
            height="auto"
            borderRadius={22}
            backgroundOpacity={0.12}
            opacity={0.5}
            blur={10}
            className="about-card animate-fade-up animate-delay-3"
          >
            <div className="about-card-body">
              <h2 className="about-h2">The variants</h2>
              <p className="about-note">
                The board scales by dimensionality, size, and connect length. The
                defaults below match the agent&apos;s training mix.
              </p>
              <div className="about-grid">
                {VARIANTS.map((variant) => (
                  <div className="about-variant" key={variant.title}>
                    <span className="about-variant-tag">{variant.tag}</span>
                    <span className="about-variant-title">{variant.title}</span>
                    <span className="about-variant-blurb">{variant.blurb}</span>
                  </div>
                ))}
              </div>
            </div>
          </GlassSurface>

          <GlassSurface
            width="100%"
            height="auto"
            borderRadius={22}
            backgroundOpacity={0.12}
            opacity={0.5}
            blur={10}
            className="about-card animate-fade-up animate-delay-4"
          >
            <div className="about-card-body">
              <h2 className="about-h2">The agent</h2>
              <p className="about-note">
                The demo is backed by a single universal policy–value transformer
                that loads once and legally plays every variant. On a held-out
                robust evaluation it scores <strong>0.8221</strong>, and in 3D it
                reaches <strong>94.4%</strong> against the heuristic baseline.
              </p>
              <div className="about-links">
                <Link className="about-link primary" to="/technical-report">
                  Read the technical report
                </Link>
                <a
                  className="about-link"
                  href="https://github.com/anthonylu23/hyperzero"
                  rel="noopener noreferrer"
                  target="_blank"
                >
                  View on GitHub
                </a>
              </div>
            </div>
          </GlassSurface>
        </div>
      </main>
    </>
  );
}
