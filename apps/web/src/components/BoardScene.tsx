import * as React from "react";
import { OrbitControls, Text } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import type { Mesh, OrthographicCamera } from "three";

import type {
  ActionInfo,
  AgentMovePayload,
  CellInfo,
  GameSnapshot,
  LastMove,
} from "../lib/types";

interface BoardSceneProps {
  game: GameSnapshot | null;
  agent: AgentMovePayload | null;
  busy: boolean;
  hoveredAction: number | null;
  onAction: (action: number) => void;
  onHoverAction: (action: number | null) => void;
}

interface ActionCoord4D {
  x: number;
  z: number;
  w: number;
}

const X_COLOR = "#f06449";
const O_COLOR = "#38bdf8";
const EMPTY_COLOR = "#3a4a58";
const EMPTY_EDGE_COLOR = "#a7bac8";
const LAST_COLOR = "#f7d25c";
const GHOST_COLOR = "#d8ff65";

const CUBE_SPACING = 3.2;
const CUBE_SIZE = 2.85;
const XZ_SLOT_SPACING = 0.58;
const Y_SLOT_SPACING = 0.82;

export function BoardScene({
  game,
  agent,
  busy,
  hoveredAction,
  onAction,
  onHoverAction,
}: BoardSceneProps) {
  if (!game) {
    return (
      <div className="canvas-shell loading">
        <span>Loading board</span>
      </div>
    );
  }

  if (game.mode.dimensions === 4) {
    return (
      <CubeRow4DBoard
        agent={agent}
        busy={busy}
        game={game}
        hoveredAction={hoveredAction}
        onAction={onAction}
        onHoverAction={onHoverAction}
      />
    );
  }

  return (
    <SpatialBoard
      busy={busy}
      game={game}
      hoveredAction={hoveredAction}
      onAction={onAction}
      onHoverAction={onHoverAction}
    />
  );
}

function SpatialBoard({
  game,
  busy,
  hoveredAction,
  onAction,
  onHoverAction,
}: Omit<BoardSceneProps, "agent" | "game"> & { game: GameSnapshot }) {
  const visibleActions = game.actions.filter(
    (action) => action.legal && action.next_cell,
  );
  const cameraPosition = cameraFor(game.mode.dimensions);
  const lastMoveIndex = game.state.last_move?.cell_index ?? -1;
  const hoveredActionInfo = actionById(game, hoveredAction);
  const hoveredLabel = hoveredActionInfo
    ? hoveredActionInfo.coord.length === 0
      ? `Column ${hoveredActionInfo.action}`
      : `Column ${hoveredActionInfo.coord.join(",")}`
    : "Hover a column";

  return (
    <div className="canvas-shell" data-testid="board-shell">
      <div className="hover-readout" data-testid="hover-readout">
        {hoveredLabel}
      </div>

      <Canvas camera={{ position: cameraPosition, fov: 42 }}>
        <color attach="background" args={["#10151b"]} />
        <ambientLight intensity={0.65} />
        <directionalLight intensity={1.8} position={[4, 7, 5]} />
        <directionalLight intensity={0.45} position={[-5, 4, -4]} />
        <BoardLattice
          cells={game.cells}
          game={game}
          hoveredAction={hoveredActionInfo}
        />
        {game.cells.map((cell) => (
          <Piece
            cell={cell}
            game={game}
            isLast={cell.index === lastMoveIndex}
            key={cell.index}
          />
        ))}
        {visibleActions.map((action) => (
          <ActionGhost
            action={action}
            disabled={busy || !game.is_human_turn}
            game={game}
            hovered={hoveredAction === action.action}
            key={action.action}
            onAction={onAction}
            onHoverAction={onHoverAction}
          />
        ))}
        <OrbitControls enablePan={false} maxDistance={16} minDistance={5} />
      </Canvas>
    </div>
  );
}

function CubeRow4DBoard({
  game,
  agent,
  busy,
  hoveredAction,
  onAction,
  onHoverAction,
}: BoardSceneProps & { game: GameSnapshot }) {
  const hovered = actionById(game, hoveredAction);
  const hoveredCoord = actionCoord4D(hovered);
  const focusAction = hovered ?? actionFromLastMove(game) ?? firstLegalAction(game);
  const focus = actionCoord4D(focusAction) ?? { x: 0, z: 0, w: 0 };
  const hoverText =
    hovered && hoveredCoord && hovered.next_cell
      ? `Hover: action (x=${hoveredCoord.x}, z=${hoveredCoord.z}, w=${hoveredCoord.w}), lands at y=${hovered.next_cell[0]}`
      : game.state.last_move
        ? moveReadout(game, game.state.last_move)
        : "Hover: choose a cube column";
  const lastMoveText = game.state.last_move
    ? moveReadout(game, game.state.last_move)
    : "No moves yet";

  return (
    <div className="cube4d-shell" data-testid="board-shell">
      <section className="cube4d-stage" aria-label="4D cube row">
        <div className="cube4d-overlay">
          <span className="control-label">w axis</span>
          <span data-testid="cube4d-hover-readout">{hoverText}</span>
        </div>
        <Canvas
          camera={{ position: [4.8, 2.2, 10], zoom: 44 }}
          orthographic
          onCreated={({ camera }) => camera.lookAt(0, 0, 0)}
        >
          <color attach="background" args={["#10151b"]} />
          <ambientLight intensity={0.7} />
          <directionalLight intensity={1.9} position={[3, 7, 6]} />
          <directionalLight intensity={0.5} position={[-6, 4, -4]} />
          <CubeCameraFit />
          <CubeRowScene
            disabled={busy || !game.is_human_turn}
            game={game}
            hovered={hovered}
            hoveredCoord={hoveredCoord}
            onAction={onAction}
            onHoverAction={onHoverAction}
          />
          <OrbitControls
            enablePan={false}
            makeDefault
            maxDistance={26}
            minDistance={8}
            target={[0, 0, 0]}
          />
        </Canvas>
      </section>

      <aside className="cube4d-inspector" aria-label="4D move details">
        <section className="cube4d-panel-block">
          <span className="control-label">Move details</span>
          <div className="cube4d-readout" data-testid="cube4d-readout">
            {hoverText}
          </div>
        </section>

        <section className="cube4d-panel-block">
          <span className="control-label">Active cube</span>
          <h2>w={focus.w}</h2>
          <p>
            x={focus.x}, z={focus.z}
          </p>
        </section>

        <section className="cube4d-panel-block">
          <span className="control-label">Search readout</span>
          <dl>
            <div>
              <dt>Action</dt>
              <dd>{agent?.action ?? "-"}</dd>
            </div>
            <div>
              <dt>Value</dt>
              <dd>{agent ? agent.value.toFixed(3) : "-"}</dd>
            </div>
            <div>
              <dt>Sims</dt>
              <dd>{agent?.simulations ?? "-"}</dd>
            </div>
          </dl>
        </section>

        <section className="cube4d-panel-block">
          <span className="control-label">Last move</span>
          <div className="cube4d-readout">{lastMoveText}</div>
        </section>

        <section className="cube4d-panel-block">
          <span className="control-label">Legend</span>
          <div className="tensor-legend">
            <span>
              <i className="legend-dot human" /> Human
            </span>
            <span>
              <i className="legend-dot agent" /> Agent
            </span>
            <span>
              <i className="legend-dot last" /> Last
            </span>
            <span>
              <i className="legend-dot preview" /> Hover
            </span>
          </div>
        </section>
      </aside>
    </div>
  );
}

function CubeCameraFit() {
  const { camera, size } = useThree();

  React.useEffect(() => {
    const orthographicCamera = camera as OrthographicCamera;
    orthographicCamera.zoom = Math.max(30, Math.min(50, size.width / 13.8));
    orthographicCamera.lookAt(0, 0, 0);
    orthographicCamera.updateProjectionMatrix();
  }, [camera, size.width]);

  return null;
}

function CubeRowScene({
  game,
  hovered,
  hoveredCoord,
  disabled,
  onAction,
  onHoverAction,
}: {
  game: GameSnapshot;
  hovered: ActionInfo | null;
  hoveredCoord: ActionCoord4D | null;
  disabled: boolean;
  onAction: (action: number) => void;
  onHoverAction: (action: number | null) => void;
}) {
  const lastMoveIndex = game.state.last_move?.cell_index ?? -1;
  const actions = game.actions.filter((action) => action.legal && action.next_cell);

  return (
    <group>
      {[0, 1, 2, 3].map((w) => (
        <group key={`cube-${w}`} position={[cubeOffset(w), 0, 0]}>
          <Text
            anchorX="center"
            anchorY="middle"
            color="#d8ff65"
            fontSize={0.26}
            position={[0, CUBE_SIZE / 2 + 0.45, 0]}
          >
            w{w}
          </Text>
          <CubeFrame active={hoveredCoord?.w === w} />
        </group>
      ))}

      {game.cells.map((cell) => (
        <CubeSlot
          cell={cell}
          game={game}
          highlighted={cellHighlighted(cell, hoveredCoord)}
          isLast={cell.index === lastMoveIndex}
          key={cell.index}
        />
      ))}

      {actions.map((action) => (
        <ColumnTarget
          action={action}
          disabled={disabled}
          hovered={hovered?.action === action.action}
          key={action.action}
          related={columnRelated(action, hoveredCoord)}
          onAction={onAction}
          onHoverAction={onHoverAction}
        />
      ))}
    </group>
  );
}

function CubeFrame({ active }: { active: boolean }) {
  return (
    <mesh>
      <boxGeometry args={[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]} />
      <meshBasicMaterial
        color={active ? GHOST_COLOR : "#536472"}
        opacity={active ? 0.48 : 0.34}
        transparent
        wireframe
      />
    </mesh>
  );
}

function CubeSlot({
  cell,
  game,
  highlighted,
  isLast,
}: {
  cell: CellInfo;
  game: GameSnapshot;
  highlighted: boolean;
  isLast: boolean;
}) {
  const occupied = cell.value !== 0;
  const color = isLast
    ? LAST_COLOR
    : cell.value === game.human_player
      ? X_COLOR
      : cell.value === game.agent_player
        ? O_COLOR
        : highlighted
          ? GHOST_COLOR
          : EMPTY_COLOR;
  const opacity = occupied || isLast ? 1 : highlighted ? 0.5 : 0.38;
  const edgeOpacity = occupied || isLast ? 0.94 : highlighted ? 0.92 : 0.78;
  const radius = occupied || isLast ? 0.15 : highlighted ? 0.17 : 0.15;

  return (
    <group position={cubeCoordPosition(cell.coord)}>
      <mesh>
        <sphereGeometry args={[radius, 18, 18]} />
        <meshStandardMaterial
          color={color}
          depthTest={occupied || isLast}
          depthWrite={occupied || isLast}
          emissive={highlighted ? GHOST_COLOR : "#1f303c"}
          emissiveIntensity={highlighted ? 0.16 : occupied ? 0 : 0.08}
          metalness={occupied ? 0.18 : 0}
          opacity={opacity}
          roughness={0.42}
          transparent
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[radius + 0.011, 12, 12]} />
        <meshBasicMaterial
          color={highlighted ? GHOST_COLOR : EMPTY_EDGE_COLOR}
          depthTest={false}
          depthWrite={false}
          opacity={edgeOpacity}
          transparent
          wireframe
        />
      </mesh>
    </group>
  );
}

function ColumnTarget({
  action,
  hovered,
  related,
  disabled,
  onAction,
  onHoverAction,
}: {
  action: ActionInfo;
  hovered: boolean;
  related: boolean;
  disabled: boolean;
  onAction: (action: number) => void;
  onHoverAction: (action: number | null) => void;
}) {
  const coord = actionCoord4D(action);
  if (!coord || !action.next_cell) {
    return null;
  }
  const position = cubeActionPosition(coord);
  const previewPosition = cubeCoordPosition(action.next_cell);
  const hoverAction = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    onHoverAction(action.action);
  };

  return (
    <group>
      <mesh
        onClick={(event) => {
          event.stopPropagation();
          if (!disabled) {
            onAction(action.action);
          }
        }}
        onPointerMove={hoverAction}
        onPointerOut={() => onHoverAction(null)}
        onPointerOver={hoverAction}
        position={position}
      >
        <boxGeometry args={[0.42, CUBE_SIZE, 0.42]} />
        <meshBasicMaterial
          color={GHOST_COLOR}
          opacity={hovered ? 0.16 : related ? 0.055 : 0.002}
          transparent
        />
      </mesh>
      {hovered ? (
        <mesh position={previewPosition}>
          <sphereGeometry args={[0.24, 24, 24]} />
          <meshStandardMaterial
            color={GHOST_COLOR}
            emissive={GHOST_COLOR}
            emissiveIntensity={0.34}
            opacity={0.62}
            transparent
          />
        </mesh>
      ) : null}
    </group>
  );
}

function BoardLattice({
  cells,
  game,
  hoveredAction,
}: {
  cells: CellInfo[];
  game: GameSnapshot;
  hoveredAction: ActionInfo | null;
}) {
  return (
    <group>
      {cells.map((cell) => (
        <Slot
          cell={cell}
          game={game}
          highlighted={cellInActionColumn(cell, hoveredAction)}
          key={`slot-${cell.index}`}
        />
      ))}
    </group>
  );
}

function Slot({
  cell,
  game,
  highlighted,
}: {
  cell: CellInfo;
  game: GameSnapshot;
  highlighted: boolean;
}) {
  const occupied = cell.value !== 0;
  const opacity = highlighted ? 0.34 : occupied ? 0.12 : 0.25;
  const edgeOpacity = highlighted ? 0.86 : 0.48;
  const radius = highlighted ? 0.36 : 0.31;

  return (
    <group position={coordToPosition(cell.coord, game)}>
      <mesh>
        <sphereGeometry args={[radius, 24, 24]} />
        <meshStandardMaterial
          color={highlighted ? GHOST_COLOR : EMPTY_COLOR}
          emissive={highlighted ? GHOST_COLOR : "#000000"}
          emissiveIntensity={highlighted ? 0.1 : 0}
          opacity={opacity}
          roughness={0.62}
          transparent
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[radius + 0.015, 18, 18]} />
        <meshBasicMaterial
          color={highlighted ? GHOST_COLOR : EMPTY_EDGE_COLOR}
          opacity={edgeOpacity}
          transparent
          wireframe
        />
      </mesh>
    </group>
  );
}

function Piece({
  cell,
  game,
  isLast,
}: {
  cell: CellInfo;
  game: GameSnapshot;
  isLast: boolean;
}) {
  if (cell.value === 0) {
    return null;
  }
  const color = cell.value === 1 ? X_COLOR : O_COLOR;
  return (
    <mesh position={coordToPosition(cell.coord, game)}>
      <sphereGeometry args={[isLast ? 0.38 : 0.34, 32, 32]} />
      <meshStandardMaterial
        color={isLast ? LAST_COLOR : color}
        metalness={0.2}
        roughness={0.34}
      />
    </mesh>
  );
}

function ActionGhost({
  action,
  disabled,
  game,
  hovered,
  onAction,
  onHoverAction,
}: {
  action: ActionInfo;
  disabled: boolean;
  game: GameSnapshot;
  hovered: boolean;
  onAction: (action: number) => void;
  onHoverAction: (action: number | null) => void;
}) {
  const ref = React.useRef<Mesh>(null);
  if (!action.next_cell) {
    return null;
  }
  return (
    <mesh
      ref={ref}
      onClick={(event) => {
        event.stopPropagation();
        if (!disabled) {
          onAction(action.action);
        }
      }}
      onPointerOut={() => onHoverAction(null)}
      onPointerOver={(event) => {
        event.stopPropagation();
        onHoverAction(action.action);
      }}
      position={coordToPosition(action.next_cell, game)}
    >
      <sphereGeometry args={[hovered ? 0.5 : 0.42, 28, 28]} />
      <meshStandardMaterial
        color={GHOST_COLOR}
        emissive={GHOST_COLOR}
        emissiveIntensity={disabled ? 0.04 : hovered ? 0.32 : 0.18}
        opacity={disabled ? 0.12 : hovered ? 0.54 : 0.34}
        transparent
      />
    </mesh>
  );
}

function cubeCoordPosition(coord: number[]): [number, number, number] {
  const [y, x, z, w] = coord;
  return [
    cubeOffset(w) + (x - 1.5) * XZ_SLOT_SPACING,
    (y - 1.5) * Y_SLOT_SPACING,
    (z - 1.5) * XZ_SLOT_SPACING,
  ];
}

function cubeActionPosition(coord: ActionCoord4D): [number, number, number] {
  return [
    cubeOffset(coord.w) + (coord.x - 1.5) * XZ_SLOT_SPACING,
    0,
    (coord.z - 1.5) * XZ_SLOT_SPACING,
  ];
}

function cubeOffset(w: number) {
  return (w - 1.5) * CUBE_SPACING;
}

function actionById(game: GameSnapshot, action: number | null) {
  if (action === null) {
    return null;
  }
  return game.actions.find((candidate) => candidate.action === action) ?? null;
}

function firstLegalAction(game: GameSnapshot) {
  return game.actions.find((action) => action.legal) ?? null;
}

function actionFromLastMove(game: GameSnapshot) {
  const lastMove = game.state.last_move;
  if (!lastMove) {
    return null;
  }
  return game.actions.find((action) => action.action === lastMove.action) ?? null;
}

function actionCoord4D(action: ActionInfo | null): ActionCoord4D | null {
  if (!action || action.coord.length < 3) {
    return null;
  }
  return {
    x: action.coord[0],
    z: action.coord[1],
    w: action.coord[2],
  };
}

function cellHighlighted(cell: CellInfo, coord: ActionCoord4D | null) {
  if (!coord || cell.coord.length < 4) {
    return false;
  }
  const [, x, z, w] = cell.coord;
  return x === coord.x && z === coord.z && w === coord.w;
}

function columnRelated(action: ActionInfo, coord: ActionCoord4D | null) {
  const actionCoord = actionCoord4D(action);
  return (
    !!actionCoord &&
    !!coord &&
    actionCoord.x === coord.x &&
    actionCoord.z === coord.z &&
    actionCoord.w !== coord.w
  );
}

function moveReadout(game: GameSnapshot, move: LastMove) {
  const [x, z, w] = move.column_coord;
  const y = move.cell_coord[0];
  const actor = move.player === game.agent_player ? "Agent" : "Human";
  return `${actor} played (x=${x}, z=${z}, w=${w}), landed at y=${y}`;
}

function cellInActionColumn(cell: CellInfo, action: ActionInfo | null) {
  if (!action) {
    return false;
  }
  const gravityAxis = 0;
  const actionAxes = cell.coord
    .map((_, axis) => axis)
    .filter((axis) => axis !== gravityAxis);
  return actionAxes.every((axis, index) => cell.coord[axis] === action.coord[index]);
}

function coordToPosition(coord: number[], game: GameSnapshot): [number, number, number] {
  const shape = game.mode.shape;
  const scale = game.mode.dimensions === 2 ? 0.92 : 1.05;
  const x = ((coord[1] ?? 0) - ((shape[1] ?? 1) - 1) / 2) * scale;
  const y = ((coord[0] ?? 0) - ((shape[0] ?? 1) - 1) / 2) * scale;
  const z = ((coord[2] ?? 0) - ((shape[2] ?? 1) - 1) / 2) * scale;
  return [x, y, z];
}

function cameraFor(dimensions: number): [number, number, number] {
  if (dimensions === 2) {
    return [0, 0.6, 8.4];
  }
  return [5.8, 4.8, 6.4];
}
