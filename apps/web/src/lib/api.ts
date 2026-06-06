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

export type AgentMoveStreamEvent =
  | {
      event: "model_loading";
      data: {
        difficulty: string;
        simulations: number;
        loaded: boolean;
      };
    }
  | {
      event: "search_started" | "simulation_progress";
      data: {
        simulations_completed: number;
        simulations: number;
        duration_ms: number;
        visits: number[];
        policy: number[];
      };
    }
  | {
      event: "move_final";
      data: AgentMoveResponse;
    };

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

export async function streamAgentMove(
  gameId: string,
  onEvent: (event: AgentMoveStreamEvent) => void,
): Promise<AgentMoveResponse> {
  const response = await fetch(`${API_URL}/games/${gameId}/agent-move-stream`, {
    headers: {
      accept: "text/event-stream",
    },
    method: "POST",
  });
  if (!response.ok) {
    throw await responseError(response);
  }
  if (!response.body) {
    throw new Error("Streaming is not supported by this browser");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: AgentMoveResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const event = parseStreamEvent(block);
      if (!event) {
        continue;
      }
      onEvent(event);
      if (event.event === "move_final") {
        finalResponse = event.data;
      }
    }
  }

  if (buffer.trim()) {
    const event = parseStreamEvent(buffer);
    if (event) {
      onEvent(event);
      if (event.event === "move_final") {
        finalResponse = event.data;
      }
    }
  }
  if (!finalResponse) {
    throw new Error("Agent stream ended before a final move");
  }
  return finalResponse;
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
    throw await responseError(response);
  }
  return (await response.json()) as T;
}

async function responseError(response: Response) {
  let message = `${response.status} ${response.statusText}`;
  try {
    const payload = (await response.json()) as ApiErrorPayload;
    message = payload.detail?.message ?? payload.detail?.error ?? message;
  } catch {
    // Keep the status text when the server returns a non-JSON response.
  }
  return new Error(message);
}

function parseStreamEvent(block: string): AgentMoveStreamEvent | null {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  if (!dataLines.length) {
    return null;
  }
  return {
    event: eventName,
    data: JSON.parse(dataLines.join("\n")),
  } as AgentMoveStreamEvent;
}
