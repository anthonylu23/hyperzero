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
} from "../lib/types";

interface BoardSceneProps {
  game: GameSnapshot | null;
  agent: AgentMovePayload | null;
  busy: boolean;
  hoveredAction: number | null;
  theme: "dark" | "light";
  onAction: (action: number) => void;
  onHoverAction: (action: number | null) => void;
}

interface ActionCoord4D {
  x: number;
  z: number;
  w: number;
}

interface BoardPalette {
  agent: string;
  empty: string;
  emptyEdge: string;
  frame: string;
  ghost: string;
  ghostEmissive: string;
  human: string;
  last: string;
}

const BOARD_PALETTES: Record<"dark" | "light", BoardPalette> = {
  dark: {
    agent: "#e03535",
    empty: "#4b5355",
    emptyEdge: "#c8c1b7",
    frame: "#667071",
    ghost: "#00b8ad",
    ghostEmissive: "#123d3b",
    human: "#00b8ad",
    last: "#f0a322",
  },
  light: {
    agent: "#6f675f",
    empty: "#d0c6b8",
    emptyEdge: "#776f66",
    frame: "#9c9185",
    ghost: "#e99a19",
    ghostEmissive: "#5c3a0a",
    human: "#e99a19",
    last: "#00b8ad",
  },
};

const XZ_SLOT_SPACING = 0.58;
const Y_SLOT_SPACING = 0.82;
const CUBE_MARGIN = 1.1;
const CUBE_GAP = 0.35;

interface CubeLayout {
  // Per-axis cell counts in coord order [y (gravity), x, z, w].
  counts: [number, number, number, number];
  // Per-axis center offsets used to recentre coordinates around the origin.
  centers: [number, number, number, number];
  // Wireframe frame extents for one cube.
  frameW: number;
  frameH: number;
  frameD: number;
  // Distance between adjacent cube centers along the w (row) axis.
  cubeSpacing: number;
}

function makeCubeLayout(shape: number[]): CubeLayout {
  const y = shape[0] ?? 1;
  const x = shape[1] ?? 1;
  const z = shape[2] ?? 1;
  const w = shape[3] ?? 1;
  const frameW = XZ_SLOT_SPACING * (x - 1) + CUBE_MARGIN;
  const frameH = Y_SLOT_SPACING * (y - 1) + CUBE_MARGIN;
  const frameD = XZ_SLOT_SPACING * (z - 1) + CUBE_MARGIN;
  return {
    counts: [y, x, z, w],
    centers: [(y - 1) / 2, (x - 1) / 2, (z - 1) / 2, (w - 1) / 2],
    frameW,
    frameH,
    frameD,
    cubeSpacing: frameW + CUBE_GAP,
  };
}

export function BoardScene({
  game,
  agent,
  busy,
  hoveredAction,
  theme,
  onAction,
  onHoverAction,
}: BoardSceneProps) {
  const palette = BOARD_PALETTES[theme];

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
        palette={palette}
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
      palette={palette}
      onAction={onAction}
      onHoverAction={onHoverAction}
    />
  );
}

function SpatialBoard({
  game,
  busy,
  hoveredAction,
  palette,
  onAction,
  onHoverAction,
}: Omit<BoardSceneProps, "agent" | "game" | "theme"> & {
  game: GameSnapshot;
  palette: BoardPalette;
}) {
  const visibleActions = game.actions.filter(
    (action) => action.legal && action.next_cell,
  );
  const cameraPosition = cameraFor(game.mode.dimensions);
  const lastMoveIndex = game.state.last_move?.cell_index ?? -1;
  const hoveredActionInfo = actionById(game, hoveredAction);

  return (
    <div
      className="canvas-shell"
      data-board-shape={game.mode.shape.join("x")}
      data-connect-k={game.mode.connect_k}
      data-testid="board-shell"
    >
      <Canvas camera={{ position: cameraPosition, fov: 42 }} gl={{ alpha: true }}>
        <ambientLight intensity={0.65} />
        <directionalLight intensity={1.8} position={[4, 7, 5]} />
        <directionalLight intensity={0.45} position={[-5, 4, -4]} />
        <BoardLattice
          cells={game.cells}
          game={game}
          hoveredAction={hoveredActionInfo}
          palette={palette}
        />
        {game.cells.map((cell) => (
          <Piece
            cell={cell}
            game={game}
            isLast={cell.index === lastMoveIndex}
            key={cell.index}
            palette={palette}
          />
        ))}
        {visibleActions.map((action) => (
          <ActionGhost
            action={action}
            disabled={busy || !game.is_human_turn}
            game={game}
            hovered={hoveredAction === action.action}
            key={action.action}
            palette={palette}
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
  busy,
  hoveredAction,
  palette,
  onAction,
  onHoverAction,
}: Omit<BoardSceneProps, "theme"> & {
  game: GameSnapshot;
  palette: BoardPalette;
}) {
  const hovered = actionById(game, hoveredAction);
  const hoveredCoord = actionCoord4D(hovered);
  const layout = makeCubeLayout(game.mode.shape);

  return (
    <div
      className="cube4d-shell"
      data-board-shape={game.mode.shape.join("x")}
      data-connect-k={game.mode.connect_k}
      data-testid="board-shell"
    >
      <section className="cube4d-stage" aria-label="4D cube row">
        <Canvas
          camera={{ position: [4.8, 2.2, 10], zoom: 44 }}
          gl={{ alpha: true }}
          orthographic
          onCreated={({ camera }) => camera.lookAt(0, 0, 0)}
        >
          <ambientLight intensity={0.7} />
          <directionalLight intensity={1.9} position={[3, 7, 6]} />
          <directionalLight intensity={0.5} position={[-6, 4, -4]} />
          <CubeCameraFit layout={layout} />
          <CubeRowScene
            disabled={busy || !game.is_human_turn}
            game={game}
            hovered={hovered}
            hoveredCoord={hoveredCoord}
            layout={layout}
            palette={palette}
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
    </div>
  );
}

function CubeCameraFit({ layout }: { layout: CubeLayout }) {
  const { camera, size } = useThree();
  const rowWidth = layout.counts[3] * layout.cubeSpacing;

  React.useEffect(() => {
    const orthographicCamera = camera as OrthographicCamera;
    orthographicCamera.zoom = Math.max(18, Math.min(50, size.width / (rowWidth * 1.08)));
    orthographicCamera.lookAt(0, 0, 0);
    orthographicCamera.updateProjectionMatrix();
  }, [camera, size.width, rowWidth]);

  return null;
}

function CubeRowScene({
  game,
  hovered,
  hoveredCoord,
  layout,
  disabled,
  palette,
  onAction,
  onHoverAction,
}: {
  game: GameSnapshot;
  hovered: ActionInfo | null;
  hoveredCoord: ActionCoord4D | null;
  layout: CubeLayout;
  disabled: boolean;
  palette: BoardPalette;
  onAction: (action: number) => void;
  onHoverAction: (action: number | null) => void;
}) {
  const lastMoveIndex = game.state.last_move?.cell_index ?? -1;
  const actions = game.actions.filter((action) => action.legal && action.next_cell);
  const cubeIndices = Array.from({ length: layout.counts[3] }, (_, index) => index);

  return (
    <group>
      {cubeIndices.map((w) => (
        <group key={`cube-${w}`} position={[cubeOffset(w, layout), 0, 0]}>
          <Text
            anchorX="center"
            anchorY="middle"
            color={palette.ghost}
            fontSize={0.26}
            position={[0, layout.frameH / 2 + 0.45, 0]}
          >
            w{w}
          </Text>
          <CubeFrame active={hoveredCoord?.w === w} layout={layout} palette={palette} />
        </group>
      ))}

      {game.cells.map((cell) => (
        <CubeSlot
          cell={cell}
          game={game}
          highlighted={cellHighlighted(cell, hoveredCoord)}
          isLast={cell.index === lastMoveIndex}
          key={cell.index}
          layout={layout}
          palette={palette}
        />
      ))}

      {actions.map((action) => (
        <ColumnTarget
          action={action}
          disabled={disabled}
          hovered={hovered?.action === action.action}
          key={action.action}
          layout={layout}
          palette={palette}
          related={columnRelated(action, hoveredCoord)}
          onAction={onAction}
          onHoverAction={onHoverAction}
        />
      ))}
    </group>
  );
}

function CubeFrame({
  active,
  layout,
  palette,
}: {
  active: boolean;
  layout: CubeLayout;
  palette: BoardPalette;
}) {
  return (
    <mesh>
      <boxGeometry args={[layout.frameW, layout.frameH, layout.frameD]} />
      <meshBasicMaterial
        color={active ? palette.ghost : palette.frame}
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
  layout,
  palette,
}: {
  cell: CellInfo;
  game: GameSnapshot;
  highlighted: boolean;
  isLast: boolean;
  layout: CubeLayout;
  palette: BoardPalette;
}) {
  const occupied = cell.value !== 0;
  const color =
    cell.value === game.human_player
      ? palette.human
      : cell.value === game.agent_player
        ? palette.agent
        : highlighted
          ? palette.ghost
          : palette.empty;
  const opacity = occupied || isLast ? 1 : highlighted ? 0.32 : 0.22;
  const edgeOpacity = occupied || isLast ? 0.94 : highlighted ? 0.65 : 0.52;
  const radius = occupied || isLast ? 0.15 : highlighted ? 0.17 : 0.15;

  return (
    <group position={cubeCoordPosition(cell.coord, layout)}>
      <mesh>
        <sphereGeometry args={[radius, 18, 18]} />
        <meshStandardMaterial
          color={color}
          depthTest={occupied || isLast}
          depthWrite={occupied || isLast}
          emissive={highlighted ? palette.ghost : palette.ghostEmissive}
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
          color={isLast ? palette.last : highlighted ? palette.ghost : palette.emptyEdge}
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
  layout,
  palette,
  onAction,
  onHoverAction,
}: {
  action: ActionInfo;
  hovered: boolean;
  related: boolean;
  disabled: boolean;
  layout: CubeLayout;
  palette: BoardPalette;
  onAction: (action: number) => void;
  onHoverAction: (action: number | null) => void;
}) {
  const coord = actionCoord4D(action);
  if (!coord || !action.next_cell) {
    return null;
  }
  const position = cubeActionPosition(coord, layout);
  const previewPosition = cubeCoordPosition(action.next_cell, layout);
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
        <boxGeometry args={[0.42, layout.frameH, 0.42]} />
        <meshBasicMaterial
          color={palette.ghost}
          opacity={hovered ? 0.16 : related ? 0.055 : 0.002}
          transparent
        />
      </mesh>
      {hovered ? (
        <mesh position={previewPosition}>
          <sphereGeometry args={[0.24, 24, 24]} />
          <meshStandardMaterial
            color={palette.ghost}
            emissive={palette.ghost}
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
  palette,
}: {
  cells: CellInfo[];
  game: GameSnapshot;
  hoveredAction: ActionInfo | null;
  palette: BoardPalette;
}) {
  return (
    <group>
      {cells.map((cell) => (
        <Slot
          cell={cell}
          game={game}
          highlighted={cellInActionColumn(cell, hoveredAction)}
          key={`slot-${cell.index}`}
          palette={palette}
        />
      ))}
    </group>
  );
}

function Slot({
  cell,
  game,
  highlighted,
  palette,
}: {
  cell: CellInfo;
  game: GameSnapshot;
  highlighted: boolean;
  palette: BoardPalette;
}) {
  const occupied = cell.value !== 0;
  const opacity = highlighted ? 0.20 : occupied ? 0.12 : 0.14;
  const edgeOpacity = highlighted ? 0.55 : 0.30;
  const radius = highlighted ? 0.36 : 0.31;

  return (
    <group position={coordToPosition(cell.coord, game)}>
      <mesh>
        <sphereGeometry args={[radius, 24, 24]} />
        <meshStandardMaterial
          color={highlighted ? palette.ghost : palette.empty}
          emissive={highlighted ? palette.ghost : "#000000"}
          emissiveIntensity={highlighted ? 0.1 : 0}
          opacity={opacity}
          roughness={0.62}
          transparent
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[radius + 0.015, 18, 18]} />
        <meshBasicMaterial
          color={highlighted ? palette.ghost : palette.emptyEdge}
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
  palette,
}: {
  cell: CellInfo;
  game: GameSnapshot;
  isLast: boolean;
  palette: BoardPalette;
}) {
  if (cell.value === 0) {
    return null;
  }
  const color = cell.value === game.human_player ? palette.human : palette.agent;
  return (
    <group position={coordToPosition(cell.coord, game)}>
      <mesh>
        <sphereGeometry args={[isLast ? 0.38 : 0.34, 32, 32]} />
        <meshStandardMaterial color={color} metalness={0.2} roughness={0.34} />
      </mesh>
      {isLast ? (
        <mesh>
          <sphereGeometry args={[0.43, 18, 18]} />
          <meshBasicMaterial
            color={palette.last}
            opacity={0.82}
            transparent
            wireframe
          />
        </mesh>
      ) : null}
    </group>
  );
}

function ActionGhost({
  action,
  disabled,
  game,
  hovered,
  palette,
  onAction,
  onHoverAction,
}: {
  action: ActionInfo;
  disabled: boolean;
  game: GameSnapshot;
  hovered: boolean;
  palette: BoardPalette;
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
        color={palette.ghost}
        emissive={palette.ghost}
        emissiveIntensity={disabled ? 0.04 : hovered ? 0.18 : 0.10}
        opacity={disabled ? 0.12 : hovered ? 0.32 : 0.20}
        transparent
      />
    </mesh>
  );
}

function cubeCoordPosition(
  coord: number[],
  layout: CubeLayout,
): [number, number, number] {
  const [y, x, z, w] = coord;
  const [cy, cx, cz] = layout.centers;
  return [
    cubeOffset(w, layout) + (x - cx) * XZ_SLOT_SPACING,
    (y - cy) * Y_SLOT_SPACING,
    (z - cz) * XZ_SLOT_SPACING,
  ];
}

function cubeActionPosition(
  coord: ActionCoord4D,
  layout: CubeLayout,
): [number, number, number] {
  const [, cx, cz] = layout.centers;
  return [
    cubeOffset(coord.w, layout) + (coord.x - cx) * XZ_SLOT_SPACING,
    0,
    (coord.z - cz) * XZ_SLOT_SPACING,
  ];
}

function cubeOffset(w: number, layout: CubeLayout) {
  return (w - layout.centers[3]) * layout.cubeSpacing;
}

function actionById(game: GameSnapshot, action: number | null) {
  if (action === null) {
    return null;
  }
  return game.actions.find((candidate) => candidate.action === action) ?? null;
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
