import React from "react";
import { Link } from "react-router-dom";
import { BookOpen, FileText, Github, Menu } from "lucide-react";

const GITHUB_URL = "https://github.com/anthonylu23/hyperzero";

/**
 * Hamburger dropdown shared by the game and About headers: About + GitHub +
 * Technical Report. Closes on outside-click, Escape, or item select.
 */
export default function NavMenu() {
  const [open, setOpen] = React.useState(false);
  const wrapRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (!open) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const close = () => setOpen(false);

  return (
    <div className="nav-menu" ref={wrapRef}>
      <button
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="Open menu"
        className="nav-menu-trigger"
        onClick={() => setOpen((value) => !value)}
        type="button"
      >
        <Menu size={16} strokeWidth={2.2} />
      </button>
      {open ? (
        <div className="nav-menu-panel" role="menu">
          <Link
            className="nav-menu-item"
            onClick={close}
            role="menuitem"
            to="/about"
          >
            <BookOpen size={15} strokeWidth={2} />
            <span>About</span>
          </Link>
          <a
            className="nav-menu-item"
            href={GITHUB_URL}
            onClick={close}
            rel="noopener noreferrer"
            role="menuitem"
            target="_blank"
          >
            <Github size={15} strokeWidth={2} />
            <span>GitHub</span>
          </a>
          <Link
            className="nav-menu-item"
            onClick={close}
            role="menuitem"
            to="/technical-report"
          >
            <FileText size={15} strokeWidth={2} />
            <span>Technical Report</span>
          </Link>
        </div>
      ) : null}
    </div>
  );
}
