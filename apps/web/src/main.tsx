import React from "react";
import ReactDOM from "react-dom/client";
import { Play, RotateCcw, Sparkles, Zap } from "lucide-react";

import { BoardScene } from "./components/BoardScene";
import { createGame, fetchModes, postAgentMove, postHumanMove } from "./lib/api";
import type { AgentMovePayload, GameSnapshot, ModeInfo } from "./lib/types";
import "./styles.css";

const MODE_ORDER = ["2d_6x7_k4", "3d_4x4x4_k4", "4d_4x4x4x4_k4"];

function App() {
  const [modes, setModes] = React.useState<ModeInfo[]>([]);
  const [modeId, setModeId] = React.useState("2d_6x7_k4");
  const [difficulty, setDifficulty] = React.useState("quick");
  const [humanMark, setHumanMark] = React.useState<"X" | "O">("X");
  const [game, setGame] = React.useState<GameSnapshot | null>(null);
  const [agent, setAgent] = React.useState<AgentMovePayload | null>(null);
  const [hoveredAction, setHoveredAction] = React.useState<number | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    fetchModes()
      .then((payload) => {
        const ordered = [...payload.modes].sort(
          (a, b) => MODE_ORDER.indexOf(a.id) - MODE_ORDER.indexOf(b.id),
        );
        setModes(ordered);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const requestAgentMove = React.useCallback(async (gameId: string) => {
    setBusy(true);
    try {
      const response = await postAgentMove(gameId);
      setGame(response.game);
      setAgent(response.agent);
      setError(null);
      return response.game;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Agent move failed");
      return null;
    } finally {
      setBusy(false);
    }
  }, []);

  const startGame = React.useCallback(async () => {
    setBusy(true);
    setError(null);
    setAgent(null);
    setHoveredAction(null);
    try {
      const response = await createGame({
        mode_id: modeId,
        human_mark: humanMark,
        difficulty,
      });
      setGame(response.game);
      if (response.game.is_agent_turn) {
        await requestAgentMove(response.game.game_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start game");
    } finally {
      setBusy(false);
    }
  }, [difficulty, humanMark, modeId, requestAgentMove]);

  React.useEffect(() => {
    if (!game && modes.length > 0) {
      void startGame();
    }
  }, [game, modes.length, startGame]);

  const playAction = React.useCallback(
    async (action: number) => {
      if (!game || busy || !game.is_human_turn) {
        return;
      }
      setBusy(true);
      setError(null);
      setAgent(null);
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
      }
    },
    [busy, game, requestAgentMove],
  );

  const selectedMode = modes.find((mode) => mode.id === modeId);
  const legalActions = game?.actions.filter((action) => action.legal) ?? [];
  const is4D = game?.mode.dimensions === 4;
  const showMoveDock = !is4D;
  const turnLabel = game?.terminal
    ? game.winner_mark === "Draw"
      ? "Draw"
      : `${game.winner_mark} wins`
    : `${game?.turn_mark ?? "X"} to move`;

  return (
    <main className={is4D ? "app-shell mode-4d" : "app-shell"}>
      <section className="control-panel" aria-label="Game controls">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <Sparkles size={18} />
          </div>
          <div>
            <h1>HyperZero</h1>
            <p>Universal agent match lab</p>
          </div>
        </div>

        <div className="control-group">
          <span className="control-label">Mode</span>
          <div className="segmented" data-testid="mode-selector">
            {modes.map((mode) => (
              <button
                className={mode.id === modeId ? "selected" : ""}
                key={mode.id}
                onClick={() => {
                  setHoveredAction(null);
                  setModeId(mode.id);
                }}
                type="button"
              >
                {mode.short_label}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <span className="control-label">Agent</span>
          <div className="segmented">
            {["quick", "normal", "strong"].map((item) => (
              <button
                className={item === difficulty ? "selected" : ""}
                key={item}
                onClick={() => setDifficulty(item)}
                type="button"
              >
                {item}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <span className="control-label">Side</span>
          <div className="segmented compact">
            {(["X", "O"] as const).map((mark) => (
              <button
                className={mark === humanMark ? "selected" : ""}
                key={mark}
                onClick={() => setHumanMark(mark)}
                type="button"
              >
                {mark}
              </button>
            ))}
          </div>
        </div>

        <button
          className="primary-action"
          data-testid="new-game"
          disabled={busy}
          onClick={startGame}
          type="button"
        >
          <RotateCcw size={18} />
          New game
        </button>
      </section>

      <section className="board-workspace">
        <div className="status-strip">
          <div>
            <span className="eyebrow">{selectedMode?.label ?? "Loading"}</span>
            <strong data-testid="turn-status">{turnLabel}</strong>
          </div>
          <div className="status-metrics">
            <span>{game?.ply ?? 0} ply</span>
            <span>{agent ? `${agent.duration_ms.toFixed(0)} ms` : "ready"}</span>
            <span>{agent ? `${agent.simulations} sims` : difficulty}</span>
          </div>
        </div>

        <BoardScene
          agent={agent}
          game={game}
          busy={busy}
          hoveredAction={hoveredAction}
          onAction={playAction}
          onHoverAction={setHoveredAction}
        />

        {showMoveDock ? (
          <div className="move-dock" aria-label="Legal moves">
            {legalActions.map((action) => (
              <button
                className={hoveredAction === action.action ? "hovered" : ""}
                data-testid={`move-${action.action}`}
                disabled={busy || !game?.is_human_turn}
                key={action.action}
                onBlur={() => setHoveredAction(null)}
                onClick={() => playAction(action.action)}
                onFocus={() => setHoveredAction(action.action)}
                onMouseEnter={() => setHoveredAction(action.action)}
                onMouseLeave={() => setHoveredAction(null)}
                type="button"
              >
                <Play size={14} />
                {action.coord.length === 0 ? action.action : action.coord.join(",")}
              </button>
            ))}
          </div>
        ) : null}
      </section>

      <aside className="readout-panel" aria-label="Match readout">
        <div className="readout-block">
          <span className="control-label">Match</span>
          <div className="score-row">
            <span className="piece x">X</span>
            <span>{game?.human_mark === "X" ? "Human" : "Agent"}</span>
          </div>
          <div className="score-row">
            <span className="piece o">O</span>
            <span>{game?.human_mark === "O" ? "Human" : "Agent"}</span>
          </div>
        </div>

        <div className="readout-block">
          <span className="control-label">Last search</span>
          {agent ? (
            <dl>
              <div>
                <dt>Action</dt>
                <dd data-testid="agent-action">{agent.action}</dd>
              </div>
              <div>
                <dt>Value</dt>
                <dd>{agent.value.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Visits</dt>
                <dd>{agent.visits.reduce((sum, value) => sum + value, 0)}</dd>
              </div>
            </dl>
          ) : (
            <p className="quiet">No agent move yet.</p>
          )}
        </div>

        <div className="readout-block">
          <span className="control-label">State</span>
          <div className="state-light">
            <Zap size={16} />
            <span data-testid="state-summary">
              {busy ? "Thinking" : game?.terminal ? "Finished" : "Live"}
            </span>
          </div>
        </div>

        {error ? (
          <div className="error-box" data-testid="error-box">
            {error}
          </div>
        ) : null}
      </aside>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
