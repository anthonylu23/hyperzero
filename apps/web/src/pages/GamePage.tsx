import React from "react";
import { Info, Minus, Moon, Plus, RotateCcw, Sun } from "lucide-react";

import { BoardScene } from "../components/BoardScene";
import GlassSurface from "../components/GlassSurface";
import NavMenu from "../components/NavMenu";
import ResultOverlay from "../components/ResultOverlay";
import {
  createGame,
  postAgentMove,
  postHumanMove,
  streamAgentMove,
} from "../lib/api";
import type { AgentMovePayload, GameSnapshot } from "../lib/types";

const DIFFICULTY = "quick";
const MIN_SIZE = 2;
const MAX_SIZE = 8;
const MIN_K = 2;
const MAX_K = 8;
const AXIS_LABELS = ["y", "x", "z", "w"];
const DIMENSIONS = [2, 3, 4] as const;
const CONFIG_DEBOUNCE_MS = 280;

type Dimensions = (typeof DIMENSIONS)[number];
type Theme = "dark" | "light";

interface BoardConfig {
  sizes: number[];
  k: number;
}

interface SearchProgress {
  completed: number;
  simulations: number;
  visits: number[];
}

const DIM_DEFAULTS: Record<Dimensions, BoardConfig> = {
  2: { sizes: [6, 7], k: 4 },
  3: { sizes: [4, 4, 4], k: 4 },
  4: { sizes: [4, 4, 4, 4], k: 4 },
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const maxKFor = (sizes: number[]) => Math.min(MAX_K, Math.max(...sizes));

export default function GamePage({
  theme,
  onToggleTheme,
}: {
  theme: Theme;
  onToggleTheme: () => void;
}) {
  const [dims, setDims] = React.useState<Dimensions>(2);
  const [sizes, setSizes] = React.useState<number[]>(DIM_DEFAULTS[2].sizes);
  const [k, setK] = React.useState<number>(DIM_DEFAULTS[2].k);
  const [game, setGame] = React.useState<GameSnapshot | null>(null);
  const [agent, setAgent] = React.useState<AgentMovePayload | null>(null);
  const [searchProgress, setSearchProgress] =
    React.useState<SearchProgress | null>(null);
  const [hoveredAction, setHoveredAction] = React.useState<number | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [resultDismissed, setResultDismissed] = React.useState(false);
  const hasAutoStartedRef = React.useRef(false);
  const gameStartRequestRef = React.useRef(0);
  const configTimerRef = React.useRef<number | null>(null);
  const [moveInFlight, setMoveInFlight] = React.useState(false);

  React.useEffect(() => {
    if (!error) {
      return;
    }
    const timer = window.setTimeout(() => setError(null), 6000);
    return () => window.clearTimeout(timer);
  }, [error]);

  const requestAgentMove = React.useCallback(
    async (gameId: string, shouldApply: () => boolean = () => true) => {
      setBusy(true);
      setSearchProgress(null);
      try {
        let response;
        try {
          response = await streamAgentMove(gameId, (event) => {
            if (!shouldApply()) {
              return;
            }
            if (event.event === "model_loading") {
              setSearchProgress({
                completed: 0,
                simulations: event.data.simulations,
                visits: [],
              });
            } else if (
              event.event === "search_started" ||
              event.event === "simulation_progress"
            ) {
              setSearchProgress({
                completed: event.data.simulations_completed,
                simulations: event.data.simulations,
                visits: event.data.visits,
              });
            }
          });
        } catch (streamError) {
          if (!shouldApply()) {
            return null;
          }
          console.warn("Agent streaming failed; falling back to REST.", streamError);
          response = await postAgentMove(gameId);
        }
        if (!shouldApply()) {
          return null;
        }
        setGame(response.game);
        setAgent(response.agent);
        setSearchProgress(null);
        setError(null);
        return response.game;
      } catch (err) {
        if (shouldApply()) {
          setError(err instanceof Error ? err.message : "Agent move failed");
          setSearchProgress(null);
        }
        return null;
      } finally {
        if (shouldApply()) {
          setBusy(false);
        }
      }
    },
    [],
  );

  const startGame = React.useCallback(
    async (config: BoardConfig = { sizes, k }) => {
      const requestId = gameStartRequestRef.current + 1;
      gameStartRequestRef.current = requestId;
      setBusy(true);
      setError(null);
      setAgent(null);
      setSearchProgress(null);
      setHoveredAction(null);
      setResultDismissed(false);
      try {
        const response = await createGame({
          shape: config.sizes,
          connect_k: config.k,
          gravity_axis: 0,
          human_mark: "X",
          difficulty: DIFFICULTY,
        });
        if (requestId !== gameStartRequestRef.current) {
          return;
        }
        setGame(response.game);
        if (response.game.is_agent_turn) {
          await requestAgentMove(
            response.game.game_id,
            () => requestId === gameStartRequestRef.current,
          );
        }
      } catch (err) {
        if (requestId === gameStartRequestRef.current) {
          setError(err instanceof Error ? err.message : "Could not start game");
        }
      } finally {
        if (requestId === gameStartRequestRef.current) {
          setBusy(false);
        }
      }
    },
    [sizes, k, requestAgentMove],
  );

  const cancelScheduledStart = React.useCallback(() => {
    if (configTimerRef.current !== null) {
      window.clearTimeout(configTimerRef.current);
      configTimerRef.current = null;
    }
  }, []);

  // Coalesce rapid stepper clicks into a single game creation.
  const scheduleStartGame = React.useCallback(
    (config: BoardConfig) => {
      cancelScheduledStart();
      configTimerRef.current = window.setTimeout(() => {
        configTimerRef.current = null;
        void startGame(config);
      }, CONFIG_DEBOUNCE_MS);
    },
    [cancelScheduledStart, startGame],
  );

  const startGameNow = React.useCallback(
    (config?: BoardConfig) => {
      cancelScheduledStart();
      void startGame(config);
    },
    [cancelScheduledStart, startGame],
  );

  React.useEffect(() => cancelScheduledStart, [cancelScheduledStart]);

  React.useEffect(() => {
    if (hasAutoStartedRef.current) {
      return;
    }
    hasAutoStartedRef.current = true;
    void startGame(DIM_DEFAULTS[2]);
  }, [startGame]);

  const playAction = React.useCallback(
    async (action: number) => {
      if (!game || busy || !game.is_human_turn) {
        return;
      }
      setMoveInFlight(true);
      setBusy(true);
      setError(null);
      setAgent(null);
      setSearchProgress(null);
      try {
        const humanResponse = await postHumanMove(game.game_id, action);
        setHoveredAction(null);
        setGame(humanResponse.game);
        if (humanResponse.game.is_agent_turn) {
          await requestAgentMove(humanResponse.game.game_id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Move failed");
      } finally {
        setBusy(false);
        setMoveInFlight(false);
      }
    },
    [busy, game, requestAgentMove],
  );

  const canEditConfig = Boolean(game && game.ply === 0 && !moveInFlight);
  const activeMode = game?.mode;
  const currentVisits = agent?.visits ?? searchProgress?.visits;
  const totalVisits = currentVisits?.reduce((sum, value) => sum + value, 0);
  const simsValue = agent
    ? String(agent.simulations)
    : searchProgress
      ? `${searchProgress.completed}/${searchProgress.simulations}`
      : "--";
  const turnTone = game?.terminal
    ? "idle"
    : game?.is_human_turn
      ? "human"
      : game?.is_agent_turn
        ? "agent"
        : "idle";
  const turnLabel = game?.terminal
    ? game.winner_mark === "Draw"
      ? "Draw"
      : game.winner === game.human_player
        ? "You win"
        : "Agent wins"
    : game?.is_human_turn
      ? "Your move"
      : game?.is_agent_turn
        ? "Agent Thinking"
        : "Ready";
  const searchStats = [
    ["mode", activeMode?.short_label ?? "--"],
    ["policy", String(game?.ply ?? 0)],
    ["value", agent ? agent.value.toFixed(3) : "--"],
    ["sims", simsValue],
    ["visits", totalVisits ? String(totalVisits) : "--"],
    ["best", agent ? String(agent.action) : "--"],
  ];

  const handleDimsChange = React.useCallback(
    (nextDims: Dimensions) => {
      if (!canEditConfig || nextDims === dims) {
        return;
      }
      const next = DIM_DEFAULTS[nextDims];
      setDims(nextDims);
      setSizes(next.sizes);
      setK(next.k);
      startGameNow(next);
    },
    [canEditConfig, dims, startGameNow],
  );

  const handleSizeChange = React.useCallback(
    (axis: number, delta: number) => {
      if (!canEditConfig) {
        return;
      }
      const nextSizes = sizes.map((size, index) =>
        index === axis ? clamp(size + delta, MIN_SIZE, MAX_SIZE) : size,
      );
      const nextK = clamp(k, MIN_K, maxKFor(nextSizes));
      setSizes(nextSizes);
      setK(nextK);
      scheduleStartGame({ sizes: nextSizes, k: nextK });
    },
    [canEditConfig, sizes, k, scheduleStartGame],
  );

  const handleKChange = React.useCallback(
    (delta: number) => {
      if (!canEditConfig) {
        return;
      }
      const nextK = clamp(k + delta, MIN_K, maxKFor(sizes));
      setK(nextK);
      scheduleStartGame({ sizes, k: nextK });
    },
    [canEditConfig, sizes, k, scheduleStartGame],
  );

  return (
    <main className="page-shell gameplay-page">
      <GlassNav
        busy={busy}
        canEditConfig={canEditConfig}
        dims={dims}
        sizes={sizes}
        k={k}
        onDimsChange={handleDimsChange}
        onSizeChange={handleSizeChange}
        onKChange={handleKChange}
        onRestart={() => startGameNow()}
        onToggleTheme={onToggleTheme}
        theme={theme}
      />

      <section className="game-layout" aria-label="HyperZero game">
        <div className="board-stage animate-fade-up animate-delay-2">
          {game ? (
            <BoardScene
              agent={agent}
              game={game}
              busy={busy}
              hoveredAction={hoveredAction}
              theme={theme}
              onAction={playAction}
              onHoverAction={setHoveredAction}
            />
          ) : (
            <div className="canvas-shell loading">
              <Loader label="Starting game" />
            </div>
          )}
        </div>

        <GlassSurface
          width="100%"
          height="auto"
          borderRadius={22}
          backgroundOpacity={0.12}
          opacity={0.5}
          blur={10}
          className="info-strip player-strip animate-fade-up animate-delay-3"
        >
          <div className="player-row" aria-label="Game status">
            <div className="player-group">
              <span>
                <i className="legend-dot human" />
                USER
              </span>
              <span>
                <i className="legend-dot agent" />
                AGENT
              </span>
            </div>
            <div className="stat-strip" aria-label="Search statistics">
              {searchStats.map(([label, value]) => (
                <span className="stat-pill" key={label}>
                  <b>{label}</b>
                  <span className="stat-value" key={value}>
                    {value}
                  </span>
                </span>
              ))}
            </div>
            <span
              className="turn-readout"
              data-testid="turn-status"
              data-thinking={game?.is_agent_turn ? "true" : undefined}
              data-human={game?.is_human_turn ? "true" : undefined}
            >
              <i className={`turn-dot ${turnTone}`} aria-hidden="true" />
              <span className="turn-label">{turnLabel}</span>
            </span>
          </div>
        </GlassSurface>

        {error ? (
          <div className="error-box game-error" data-testid="error-box" role="alert">
            <span className="error-dot" aria-hidden="true" />
            {error}
          </div>
        ) : null}
      </section>

      {resultDismissed ? null : (
        <ResultOverlay
          agent={agent}
          game={game}
          onDismiss={() => setResultDismissed(true)}
          onPlayAgain={() => startGameNow()}
        />
      )}
    </main>
  );
}

/** A small pulsing dot lattice used while a game is loading. */
function Loader({ label }: { label: string }) {
  return (
    <div className="board-loader" role="status" aria-label={label}>
      {Array.from({ length: 9 }, (_, index) => (
        <span
          className="board-loader-dot"
          key={index}
          style={{
            animationDelay: `${(index % 3) * 0.12 + Math.floor(index / 3) * 0.12}s`,
          }}
        />
      ))}
    </div>
  );
}

function Stepper({
  label,
  value,
  min,
  max,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  disabled: boolean;
  onChange: (delta: number) => void;
}) {
  return (
    <div className="nav-stepper" data-testid={`stepper-${label}`}>
      <span className="nav-stepper-label">{label}</span>
      <button
        aria-label={`Decrease ${label}`}
        className="nav-stepper-btn"
        disabled={disabled || value <= min}
        onClick={() => onChange(-1)}
        type="button"
      >
        <Minus size={12} />
      </button>
      <span className="nav-stepper-value" data-testid={`stepper-value-${label}`}>
        {value}
      </span>
      <button
        aria-label={`Increase ${label}`}
        className="nav-stepper-btn"
        disabled={disabled || value >= max}
        onClick={() => onChange(1)}
        type="button"
      >
        <Plus size={12} />
      </button>
    </div>
  );
}

function GlassNav({
  busy,
  canEditConfig,
  dims,
  sizes,
  k,
  theme,
  onDimsChange,
  onSizeChange,
  onKChange,
  onRestart,
  onToggleTheme,
}: {
  busy: boolean;
  canEditConfig: boolean;
  dims: Dimensions;
  sizes: number[];
  k: number;
  theme: Theme;
  onDimsChange: (dims: Dimensions) => void;
  onSizeChange: (axis: number, delta: number) => void;
  onKChange: (delta: number) => void;
  onRestart: () => void;
  onToggleTheme: () => void;
}) {
  const activeIndex = DIMENSIONS.indexOf(dims);
  const maxK = maxKFor(sizes);
  const [restartSpin, setRestartSpin] = React.useState(0);
  const lockedHint = canEditConfig
    ? undefined
    : "Locked after the first move — Restart to change.";
  return (
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
        <nav className="glass-nav-inner animate-fade-up" aria-label="Primary">
          <span className="nav-brand nav-underline">HyperZero</span>
          <div className="nav-controls">
            <div
              className={canEditConfig ? "nav-steppers" : "nav-steppers is-disabled"}
              data-testid="size-steppers"
              aria-label="Board size"
              aria-disabled={!canEditConfig}
              title={lockedHint}
            >
              {sizes.map((size, axis) => (
                <Stepper
                  disabled={!canEditConfig}
                  key={AXIS_LABELS[axis] ?? axis}
                  label={AXIS_LABELS[axis] ?? String(axis)}
                  max={MAX_SIZE}
                  min={MIN_SIZE}
                  onChange={(delta) => onSizeChange(axis, delta)}
                  value={size}
                />
              ))}
            </div>

            <div
              className={canEditConfig ? "nav-steppers" : "nav-steppers is-disabled"}
              data-testid="k-stepper"
              aria-label="Win length"
              aria-disabled={!canEditConfig}
              title={lockedHint}
            >
              <Stepper
                disabled={!canEditConfig}
                label="k"
                max={maxK}
                min={MIN_K}
                onChange={onKChange}
                value={k}
              />
            </div>

            <div
              className={
                canEditConfig
                  ? "nav-mode-segments"
                  : "nav-mode-segments is-disabled"
              }
              data-testid="dimension-selector"
              role="group"
              aria-label="Board dimensions"
              aria-disabled={!canEditConfig}
              title={lockedHint}
            >
              <span
                aria-hidden="true"
                className="nav-mode-thumb"
                style={
                  {
                    "--active-i": activeIndex >= 0 ? activeIndex : 0,
                  } as React.CSSProperties
                }
              />
              {DIMENSIONS.map((dimension) => (
                <button
                  aria-checked={dimension === dims}
                  className={dimension === dims ? "active" : ""}
                  disabled={!canEditConfig}
                  key={dimension}
                  onClick={() => onDimsChange(dimension)}
                  role="radio"
                  type="button"
                >
                  {dimension}D
                </button>
              ))}
            </div>

            <span className="nav-info-wrap">
              <button
                aria-describedby="config-info"
                aria-label="Configurator instructions"
                className="nav-info"
                type="button"
              >
                <Info size={18} strokeWidth={2.2} />
              </button>
              <span className="nav-info-tip" id="config-info" role="tooltip">
                Choose dimensions, set each board axis length, then set K, the
                number in a row needed to win. Configuration is editable only
                before the first move.
              </span>
            </span>

            <button
              aria-label="Restart"
              className="nav-action"
              disabled={busy}
              onClick={() => {
                setRestartSpin((value) => value + 1);
                onRestart();
              }}
              title="Restart"
              type="button"
            >
              <RotateCcw className="restart-icon" key={restartSpin} size={16} />
            </button>

            <NavMenu />

            <button
              className="theme-toggle"
              onClick={onToggleTheme}
              type="button"
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              aria-pressed={theme === "dark"}
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </nav>
      </GlassSurface>
    </header>
  );
}
