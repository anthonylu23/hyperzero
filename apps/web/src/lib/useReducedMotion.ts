import { useEffect, useState } from "react";

/**
 * Track the user's `prefers-reduced-motion` setting so animation-heavy
 * components (3D drops, pulses, confetti) can disable motion accordingly.
 */
export function useReducedMotion(): boolean {
  const [prefers, setPrefers] = useState(false);

  useEffect(() => {
    if (
      typeof window === "undefined" ||
      typeof window.matchMedia === "undefined"
    ) {
      return;
    }
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setPrefers(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  return prefers;
}
