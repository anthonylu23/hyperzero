import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import type { CSSProperties, ReactNode } from "react";
import "./GlassSurface.css";

type Channel = "R" | "G" | "B" | "A";

type GlassSurfaceProps = {
  children: ReactNode;
  width?: number | string;
  height?: number | string;
  borderRadius?: number;
  borderWidth?: number;
  brightness?: number;
  opacity?: number;
  blur?: number;
  displace?: number;
  backgroundOpacity?: number;
  saturation?: number;
  distortionScale?: number;
  redOffset?: number;
  greenOffset?: number;
  blueOffset?: number;
  xChannel?: Channel;
  yChannel?: Channel;
  mixBlendMode?: CSSProperties["mixBlendMode"];
  quality?: "auto" | "high" | "low";
  className?: string;
  style?: CSSProperties;
};

const displacementCache = new Map<string, string>();
let svgFilterSupport: boolean | null = null;

const detectSVGFilterSupport = () => {
  if (svgFilterSupport !== null) {
    return svgFilterSupport;
  }

  if (typeof window === "undefined" || typeof document === "undefined") {
    return false;
  }

  const ua = navigator.userAgent;
  const isWebkit = /Safari/.test(ua) && !/Chrome/.test(ua);
  const isFirefox = /Firefox/.test(ua);
  if (isWebkit || isFirefox) {
    svgFilterSupport = false;
    return svgFilterSupport;
  }

  const div = document.createElement("div");
  div.style.backdropFilter = "url(#test)";
  svgFilterSupport = div.style.backdropFilter !== "";
  return svgFilterSupport;
};

function usePrefersReducedMotion() {
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

const GlassSurface = ({
  children,
  width = "100%",
  height = "auto",
  borderRadius = 32,
  borderWidth = 0.06,
  brightness = 42,
  opacity = 0.65,
  blur = 14,
  displace = 0.4,
  backgroundOpacity = 0.2,
  saturation = 1.2,
  distortionScale = -120,
  redOffset = 0,
  greenOffset = 4,
  blueOffset = 8,
  xChannel = "R",
  yChannel = "G",
  mixBlendMode = "screen",
  quality = "auto",
  className = "",
  style = {},
}: GlassSurfaceProps) => {
  const uniqueId = useId().replace(/:/g, "-");
  const filterId = `glass-filter-${uniqueId}`;
  const redGradId = `red-grad-${uniqueId}`;
  const blueGradId = `blue-grad-${uniqueId}`;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const feImageRef = useRef<SVGFEImageElement | null>(null);
  const redChannelRef = useRef<SVGFEDisplacementMapElement | null>(null);
  const greenChannelRef = useRef<SVGFEDisplacementMapElement | null>(null);
  const blueChannelRef = useRef<SVGFEDisplacementMapElement | null>(null);
  const gaussianBlurRef = useRef<SVGFEGaussianBlurElement | null>(null);

  const generateDisplacementMap = useCallback(() => {
    const rect = containerRef.current?.getBoundingClientRect();
    const actualWidth = rect?.width || 400;
    const actualHeight = rect?.height || 200;
    const quantWidth = Math.max(64, Math.round(actualWidth / 32) * 32);
    const quantHeight = Math.max(64, Math.round(actualHeight / 32) * 32);
    const edgeSize = Math.min(actualWidth, actualHeight) * (borderWidth * 0.5);

    const cacheKey = [
      quantWidth,
      quantHeight,
      borderRadius,
      borderWidth,
      brightness,
      opacity,
      blur,
      mixBlendMode,
    ].join("-");

    const cached = displacementCache.get(cacheKey);
    if (cached) return cached;

    const svgContent = `
      <svg viewBox="0 0 ${quantWidth} ${quantHeight}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="${redGradId}" x1="100%" y1="0%" x2="0%" y2="0%">
            <stop offset="0%" stop-color="#0000"/>
            <stop offset="100%" stop-color="red"/>
          </linearGradient>
          <linearGradient id="${blueGradId}" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#0000"/>
            <stop offset="100%" stop-color="blue"/>
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="${quantWidth}" height="${quantHeight}" fill="black"></rect>
        <rect x="0" y="0" width="${quantWidth}" height="${quantHeight}" rx="${borderRadius}" fill="url(#${redGradId})" />
        <rect x="0" y="0" width="${quantWidth}" height="${quantHeight}" rx="${borderRadius}" fill="url(#${blueGradId})" style="mix-blend-mode: ${mixBlendMode}" />
        <rect x="${edgeSize}" y="${edgeSize}" width="${quantWidth - edgeSize * 2}" height="${quantHeight - edgeSize * 2}" rx="${borderRadius}" fill="hsl(0 0% ${brightness}% / ${opacity})" style="filter:blur(${blur}px)" />
      </svg>
    `;

    const uri = `data:image/svg+xml,${encodeURIComponent(svgContent)}`;
    displacementCache.set(cacheKey, uri);
    return uri;
  }, [
    borderRadius,
    borderWidth,
    blur,
    brightness,
    mixBlendMode,
    opacity,
    redGradId,
    blueGradId,
  ]);

  const updateDisplacementMap = useCallback(() => {
    if (!feImageRef.current) return;
    feImageRef.current.setAttribute("href", generateDisplacementMap());
  }, [generateDisplacementMap]);

  const reducedMotion = usePrefersReducedMotion();
  const [autoLow, setAutoLow] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const evaluate = () => {
      const cores = navigator.hardwareConcurrency || 4;
      setAutoLow(cores <= 2);
    };
    evaluate();
    window.addEventListener("resize", evaluate);
    return () => window.removeEventListener("resize", evaluate);
  }, []);

  type GlassStyle = CSSProperties & {
    "--glass-frost"?: number;
    "--glass-saturation"?: number;
    "--filter-id"?: string;
  };

  const finalQuality =
    quality === "auto" ? (autoLow ? "low" : "high") : quality;

  const svgFiltersSupported = useMemo(() => detectSVGFilterSupport(), []);
  const useSVGFilter =
    svgFiltersSupported && !reducedMotion && finalQuality === "high";

  useEffect(() => {
    if (!useSVGFilter) return;
    updateDisplacementMap();
  }, [updateDisplacementMap, width, height, useSVGFilter]);

  useEffect(() => {
    if (!useSVGFilter) return;

    const observers = [redChannelRef, greenChannelRef, blueChannelRef];
    const offsets = [redOffset, greenOffset, blueOffset];

    observers.forEach((ref, index) => {
      if (!ref.current) return;
      ref.current.setAttribute("scale", `${distortionScale + offsets[index]}`);
      ref.current.setAttribute("xChannelSelector", xChannel);
      ref.current.setAttribute("yChannelSelector", yChannel);
    });

    if (gaussianBlurRef.current) {
      gaussianBlurRef.current.setAttribute("stdDeviation", `${displace}`);
    }
  }, [
    blueOffset,
    distortionScale,
    displace,
    greenOffset,
    redOffset,
    useSVGFilter,
    xChannel,
    yChannel,
  ]);

  useEffect(() => {
    if (!useSVGFilter) return;
    if (typeof ResizeObserver === "undefined" || !containerRef.current) return;

    let pendingFrame: number | null = null;
    const observer = new ResizeObserver(() => {
      if (pendingFrame !== null) return;
      pendingFrame = requestAnimationFrame(() => {
        updateDisplacementMap();
        pendingFrame = null;
      });
    });

    observer.observe(containerRef.current);
    return () => {
      observer.disconnect();
      if (pendingFrame !== null) {
        cancelAnimationFrame(pendingFrame);
      }
    };
  }, [updateDisplacementMap, useSVGFilter]);

  const inlineStyle: GlassStyle = {
    ...style,
    width: typeof width === "number" ? `${width}px` : width,
    height: typeof height === "number" ? `${height}px` : height,
    borderRadius: `${borderRadius}px`,
    "--glass-frost": backgroundOpacity,
    "--glass-saturation": saturation,
    "--filter-id": useSVGFilter ? `url(#${filterId})` : undefined,
  };

  const variantClass = useSVGFilter
    ? "glass-surface--svg"
    : "glass-surface--fallback glass-surface--static";

  return (
    <div
      ref={containerRef}
      className={`glass-surface ${variantClass} ${className}`.trim()}
      style={inlineStyle}
    >
      {useSVGFilter && (
        <svg
          className="glass-surface__filter"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <filter
              id={filterId}
              colorInterpolationFilters="sRGB"
              x="0%"
              y="0%"
              width="100%"
              height="100%"
            >
              <feImage
                ref={feImageRef}
                x="0"
                y="0"
                width="100%"
                height="100%"
                preserveAspectRatio="none"
                result="map"
              />

              <feDisplacementMap
                ref={redChannelRef}
                in="SourceGraphic"
                in2="map"
                result="dispRed"
              />
              <feColorMatrix
                in="dispRed"
                type="matrix"
                values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"
                result="red"
              />

              <feDisplacementMap
                ref={greenChannelRef}
                in="SourceGraphic"
                in2="map"
                result="dispGreen"
              />
              <feColorMatrix
                in="dispGreen"
                type="matrix"
                values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0"
                result="green"
              />

              <feDisplacementMap
                ref={blueChannelRef}
                in="SourceGraphic"
                in2="map"
                result="dispBlue"
              />
              <feColorMatrix
                in="dispBlue"
                type="matrix"
                values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0"
                result="blue"
              />

              <feBlend in="red" in2="green" mode="screen" result="rg" />
              <feBlend in="rg" in2="blue" mode="screen" result="output" />
              <feGaussianBlur
                ref={gaussianBlurRef}
                in="output"
                stdDeviation="0.7"
              />
            </filter>
          </defs>
        </svg>
      )}

      <div className="glass-surface__content">{children}</div>
    </div>
  );
};

export default GlassSurface;
