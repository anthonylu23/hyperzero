import * as React from "react";

import GlassSurface from "./GlassSurface";
import type { AgentMovePayload, GameSnapshot } from "../lib/types";

type Outcome = "win" | "loss" | "draw";

interface ResultOverlayProps {
  game: GameSnapshot | null;
  agent: AgentMovePayload | null;
  onPlayAgain: () => void;
  onDismiss: () => void;
}

const TITLES: Record<Outcome, string> = {
  win: "You Win",
  loss: "Agent Wins",
  draw: "Draw",
};

function resolveOutcome(game: GameSnapshot): Outcome {
  if (game.winner_mark === "Draw") {
    return "draw";
  }
  return game.winner === game.human_player ? "win" : "loss";
}

export function ResultOverlay({
  game,
  agent,
  onPlayAgain,
  onDismiss,
}: ResultOverlayProps) {
  const playAgainRef = React.useRef<HTMLButtonElement>(null);
  const terminal = Boolean(game?.terminal);

  React.useEffect(() => {
    if (!terminal) {
      return;
    }
    playAgainRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onDismiss();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [terminal, onDismiss]);

  if (!game || !terminal) {
    return null;
  }

  const outcome = resolveOutcome(game);
  const title = TITLES[outcome];
  const stats: Array<[string, string]> = [["plies", String(game.ply)]];
  if (agent) {
    stats.push(["value", agent.value.toFixed(3)]);
    stats.push(["sims", String(agent.simulations)]);
  }

  return (
    <div
      className="result-overlay"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onDismiss();
        }
      }}
    >
      <GlassSurface
        width="100%"
        height="auto"
        borderRadius={28}
        backgroundOpacity={0.18}
        opacity={0.6}
        blur={12}
        className="result-card"
      >
        <div
          aria-label={title}
          aria-modal="true"
          className="result-card-inner"
          data-outcome={outcome}
          role="dialog"
        >
          <span className="result-flash" aria-hidden="true" />
          <span className="result-title">{title}</span>
          <span className="result-subtitle">{game.mode.short_label}</span>
          <div className="result-stats" aria-label="Game summary">
            {stats.map(([label, value]) => (
              <span className="stat-pill" key={label}>
                <b>{label}</b>
                <span className="stat-value">{value}</span>
              </span>
            ))}
          </div>
          <div className="result-actions">
            <button
              className="result-primary"
              onClick={onPlayAgain}
              ref={playAgainRef}
              type="button"
            >
              Play again
            </button>
            <button className="result-ghost" onClick={onDismiss} type="button">
              View board
            </button>
          </div>
        </div>
      </GlassSurface>
    </div>
  );
}

export default ResultOverlay;
