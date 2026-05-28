export type PlayerValue = -1 | 0 | 1;

export interface ModeInfo {
  id: string;
  label: string;
  short_label: string;
  description: string;
  dimensions: number;
  shape: number[];
  connect_k: number;
  gravity_axis: number;
  action_shape: number[];
  num_actions: number;
}

export interface CellInfo {
  index: number;
  coord: number[];
  value: PlayerValue;
}

export interface ActionInfo {
  action: number;
  coord: number[];
  legal: boolean;
  height: number;
  next_cell: number[] | null;
}

export interface LastMove {
  action: number;
  column_coord: number[];
  cell_index: number;
  cell_coord: number[];
  player: 1 | -1;
  ply: number;
}

export interface GameSnapshot {
  game_id: string;
  mode: ModeInfo;
  human_player: 1 | -1;
  human_mark: "X" | "O";
  agent_player: 1 | -1;
  agent_mark: "X" | "O";
  difficulty: string;
  turn: 1 | -1;
  turn_mark: "X" | "O";
  winner: PlayerValue | null;
  winner_mark: "X" | "O" | "Draw" | null;
  terminal: boolean;
  is_human_turn: boolean;
  is_agent_turn: boolean;
  ply: number;
  state: {
    board: unknown;
    player_to_move: 1 | -1;
    ply: number;
    terminal: boolean;
    winner: PlayerValue | null;
    last_move: LastMove | null;
    legal_mask: boolean[];
  };
  cells: CellInfo[];
  actions: ActionInfo[];
}

export interface AgentMovePayload {
  action: number;
  duration_ms: number;
  simulations: number;
  value: number;
  visits: number[];
  policy: number[];
}
