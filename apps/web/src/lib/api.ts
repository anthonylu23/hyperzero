import type { AgentMovePayload, GameSnapshot, ModeInfo } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

interface ApiErrorPayload {
  detail?: {
    error?: string;
    message?: string;
  };
}

export interface ModesPayload {
  modes: ModeInfo[];
  difficulties: Record<string, { simulations: number }>;
}

export interface CreateGamePayload {
  mode_id?: string;
  shape?: number[];
  connect_k?: number;
  gravity_axis?: number;
  human_mark: "X" | "O";
  difficulty: string;
}

export interface GameResponse {
  game: GameSnapshot;
}

export interface AgentMoveResponse {
  move: unknown;
  agent: AgentMovePayload;
  game: GameSnapshot;
}

export function fetchModes() {
  return request<ModesPayload>("/modes");
}

export function createGame(payload: CreateGamePayload) {
  return request<GameResponse>("/games", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function postHumanMove(gameId: string, action: number) {
  return request<GameResponse>(`/games/${gameId}/moves`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });
}

export function postAgentMove(gameId: string) {
  return request<AgentMoveResponse>(`/games/${gameId}/agent-move`, {
    method: "POST",
  });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: {
      "content-type": "application/json",
      ...init?.headers,
    },
    ...init,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as ApiErrorPayload;
      message = payload.detail?.message ?? payload.detail?.error ?? message;
    } catch {
      // Keep the status text when the server returns a non-JSON response.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}
