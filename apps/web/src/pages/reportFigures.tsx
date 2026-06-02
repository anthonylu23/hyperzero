import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  ARCH,
  COMPLEXITY,
  EVAL,
  LOSS,
  MILESTONES,
  RANKING,
  SEEDS_4D,
  SERIES,
  VALIDATION,
} from "./reportData";

// All chrome (grid, axes, ticks, tooltip, legend) is themed through CSS custom
// properties so the charts follow the dark/light toggle; series colours come
// from reportData.SERIES. Recharts renders SVG, so `hsl(var(--…))` strings work
// directly as fills/strokes — the same approach the old hand-rolled SVG used.

const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

const GRID = "hsl(var(--border) / 0.5)";
const AXIS_LINE = "hsl(var(--border) / 0.6)";
const TICK = {
  fill: "hsl(var(--text-secondary))",
  fontSize: 11,
  fontFamily: '"IBM Plex Mono", ui-monospace, monospace',
};
const AXIS_LABEL = { fill: "hsl(var(--text-secondary))", fontSize: 12 };
const LABEL_FILL = "hsl(var(--text-secondary))";

interface Series {
  key: string;
  label: string;
  color: string;
}

// --- Shared chrome ---------------------------------------------------------

function ChartTooltip({ active, payload, label, unit = "" }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="fig-tooltip">
      {label !== undefined && label !== "" && <div className="fig-tooltip-label">{label}</div>}
      {payload.map((p: any) => (
        <div className="fig-tooltip-row" key={p.dataKey ?? p.name}>
          <span className="fig-tooltip-swatch" style={{ background: p.color ?? p.payload?.fill }} />
          <span className="fig-tooltip-name">{p.name}</span>
          <span className="fig-tooltip-val">
            {fmt(p.value)}
            {unit}
          </span>
        </div>
      ))}
    </div>
  );
}

function useSeriesToggle() {
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const toggle = (key: string) =>
    setHidden((h) => ({ ...h, [key]: !h[key] }));
  return { hidden, toggle };
}

function SeriesLegend({
  series,
  hidden,
  toggle,
}: {
  series: Series[];
  hidden: Record<string, boolean>;
  toggle: (key: string) => void;
}) {
  return (
    <ul className="fig-legend">
      {series.map((s) => (
        <li
          className={hidden[s.key] ? "fig-legend-item off" : "fig-legend-item"}
          key={s.key}
          onClick={() => toggle(s.key)}
        >
          <span className="fig-legend-swatch" style={{ background: s.color }} />
          {s.label}
        </li>
      ))}
    </ul>
  );
}

const labelFmt = (v: any) => fmt(Number(v));

// --- Generic grouped bar ---------------------------------------------------

function GroupedBar({
  series,
  groups,
  data,
  ymax,
  ylabel = "",
  unit = "",
  height = 330,
}: {
  series: Series[];
  groups: string[];
  data: Record<string, number[]>;
  ymax?: number;
  ylabel?: string;
  unit?: string;
  height?: number;
}) {
  const { hidden, toggle } = useSeriesToggle();
  const showLabels = series.length * groups.length <= 9;
  const rows = groups.map((g, gi) => {
    const r: Record<string, number | string> = { group: g };
    series.forEach((s) => {
      r[s.key] = data[s.key][gi];
    });
    return r;
  });

  return (
    <ResponsiveContainer height={height} width="100%">
      <BarChart data={rows} margin={{ top: 24, right: 16, bottom: 4, left: 8 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis axisLine={{ stroke: AXIS_LINE }} dataKey="group" tick={TICK} tickLine={false} />
        <YAxis
          axisLine={false}
          domain={[0, ymax ?? "auto"]}
          label={{ angle: -90, position: "insideLeft", style: AXIS_LABEL, value: ylabel }}
          tick={TICK}
          tickLine={false}
          width={52}
        />
        <Tooltip
          content={<ChartTooltip unit={unit} />}
          cursor={{ fill: "hsl(var(--text-primary) / 0.05)" }}
        />
        <Legend
          content={() => <SeriesLegend hidden={hidden} series={series} toggle={toggle} />}
        />
        {series.map((s) => (
          <Bar
            dataKey={s.key}
            fill={s.color}
            hide={hidden[s.key]}
            key={s.key}
            name={s.label}
            radius={[3, 3, 0, 0]}
          >
            {showLabels && (
              <LabelList
                dataKey={s.key}
                fill={LABEL_FILL}
                fontSize={9}
                formatter={labelFmt}
                position="top"
              />
            )}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

// --- Generic multi-line chart ----------------------------------------------

function MultiLine({
  series,
  x,
  data,
  ylabel = "",
  xlabel = "iteration",
  unit = "",
  ymin = 0,
  ymax,
  height = 330,
}: {
  series: Series[];
  x: number[];
  data: Record<string, number[]>;
  ylabel?: string;
  xlabel?: string;
  unit?: string;
  ymin?: number;
  ymax?: number;
  height?: number;
}) {
  const { hidden, toggle } = useSeriesToggle();
  const rows = x.map((xv, i) => {
    const r: Record<string, number> = { x: xv };
    series.forEach((s) => {
      r[s.key] = data[s.key][i];
    });
    return r;
  });
  const dots = x.length <= 32;

  return (
    <ResponsiveContainer height={height} width="100%">
      <LineChart data={rows} margin={{ top: 16, right: 18, bottom: 18, left: 8 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          axisLine={{ stroke: AXIS_LINE }}
          dataKey="x"
          label={{ position: "insideBottom", offset: -10, style: AXIS_LABEL, value: xlabel }}
          tick={TICK}
          tickLine={false}
          type="number"
        />
        <YAxis
          axisLine={false}
          domain={[ymin, ymax ?? "auto"]}
          label={{ angle: -90, position: "insideLeft", style: AXIS_LABEL, value: ylabel }}
          tick={TICK}
          tickLine={false}
          width={52}
        />
        <Tooltip
          content={<ChartTooltip unit={unit} />}
          cursor={{ stroke: AXIS_LINE, strokeDasharray: "3 3" }}
        />
        <Legend
          content={() => <SeriesLegend hidden={hidden} series={series} toggle={toggle} />}
        />
        {series.map((s) => (
          <Line
            activeDot={{ r: 4 }}
            dataKey={s.key}
            dot={dots ? { r: 2.4, fill: s.color } : false}
            hide={hidden[s.key]}
            isAnimationActive
            key={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={2.2}
            type="monotone"
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

// --- Bound figure components -----------------------------------------------

export const FigComplexity = () => (
  <GroupedBar
    data={{ actions: COMPLEXITY.actions, cells: COMPLEXITY.cells, lines: COMPLEXITY.lines }}
    groups={COMPLEXITY.groups}
    series={[
      { key: "actions", label: "legal actions", color: SERIES.actions },
      { key: "cells", label: "board cells", color: SERIES.cells },
      { key: "lines", label: "winning lines", color: SERIES.lines },
    ]}
    ylabel="count"
  />
);

export const FigLoss = () => (
  <MultiLine
    data={{ total: LOSS.total, policy: LOSS.policy, value: LOSS.value }}
    series={[
      { key: "total", label: "total loss", color: SERIES.total },
      { key: "policy", label: "policy loss", color: SERIES.policy },
      { key: "value", label: "value loss", color: SERIES.value },
    ]}
    x={LOSS.iters}
    ylabel="loss"
  />
);

export const FigEval = () => (
  <MultiLine
    data={{ random: EVAL.random, tactical: EVAL.tactical, heuristic: EVAL.heuristic }}
    series={[
      { key: "random", label: "random", color: SERIES.random },
      { key: "tactical", label: "tactical", color: SERIES.tactical },
      { key: "heuristic", label: "heuristic", color: SERIES.heuristic },
    ]}
    unit="%"
    x={EVAL.iters}
    ylabel="win rate (%)"
    ymax={100}
  />
);

export const FigArch = () => (
  <GroupedBar
    data={{ resnet: ARCH.resnet, linemlp: ARCH.linemlp, transformer: ARCH.transformer }}
    groups={ARCH.groups}
    series={[
      { key: "resnet", label: "ResNet", color: SERIES.tactical },
      { key: "linemlp", label: "line-MLP", color: SERIES.heuristic },
      { key: "transformer", label: "Transformer", color: SERIES.mcts },
    ]}
    unit="%"
    ylabel="heuristic win (%)"
    ymax={100}
  />
);

export const FigValidation = () => (
  <GroupedBar
    data={{
      random: VALIDATION.random,
      tactical: VALIDATION.tactical,
      heuristic: VALIDATION.heuristic,
      mcts: VALIDATION.mcts,
    }}
    groups={VALIDATION.groups}
    series={[
      { key: "random", label: "vs random", color: SERIES.random },
      { key: "tactical", label: "vs tactical", color: SERIES.tactical },
      { key: "heuristic", label: "vs heuristic", color: SERIES.heuristic },
      { key: "mcts", label: "vs MCTS-32", color: SERIES.mcts },
    ]}
    unit="%"
    ylabel="win rate (%)"
    ymax={100}
  />
);

// --- Candidate ranking (horizontal bars) -----------------------------------

function RankingTick({ x, y, payload }: any) {
  const row = RANKING[payload.index];
  return (
    <text
      dominantBaseline="middle"
      fill="hsl(var(--text-secondary))"
      fontFamily='"IBM Plex Mono", ui-monospace, monospace'
      fontSize={11}
      fontWeight={row?.pass ? 700 : 400}
      textAnchor="end"
      x={x}
      y={y}
    >
      {payload.value}
    </text>
  );
}

export const FigRanking = () => (
  <ResponsiveContainer height={232} width="100%">
    <BarChart
      barCategoryGap="28%"
      data={RANKING}
      layout="vertical"
      margin={{ top: 18, right: 64, bottom: 8, left: 8 }}
    >
      <CartesianGrid horizontal={false} stroke={GRID} />
      <XAxis
        axisLine={{ stroke: AXIS_LINE }}
        domain={[0, 0.9]}
        orientation="top"
        tick={TICK}
        tickLine={false}
        type="number"
      />
      <YAxis
        axisLine={false}
        dataKey="name"
        tick={<RankingTick />}
        tickLine={false}
        type="category"
        width={232}
      />
      <Tooltip
        content={<ChartTooltip />}
        cursor={{ fill: "hsl(var(--text-primary) / 0.05)" }}
      />
      <Bar dataKey="score" isAnimationActive radius={[0, 3, 3, 0]}>
        {RANKING.map((r) => (
          <Cell
            fill={r.pass ? SERIES.pass : SERIES.fail}
            fillOpacity={r.pass ? 0.95 : 0.6}
            key={r.name}
          />
        ))}
        <LabelList
          dataKey="score"
          fill={LABEL_FILL}
          fontSize={11}
          fontWeight={700}
          formatter={(v: any) => Number(v).toFixed(4)}
          position="right"
        />
        <LabelList dataKey="note" fill="#fff" fontSize={9.5} position="insideLeft" />
      </Bar>
    </BarChart>
  </ResponsiveContainer>
);

// --- Aggregate-score milestones --------------------------------------------

function milestoneColor(m: { label: string; good: boolean }) {
  if (m.good) return SERIES.pass;
  return m.label.startsWith("Curriculum") ? SERIES.fail : SERIES.random;
}

function MilestoneTick({ x, y, payload }: any) {
  const m = MILESTONES[payload.index];
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        dy={12}
        fill="hsl(var(--text-secondary))"
        fontFamily='"IBM Plex Mono", ui-monospace, monospace'
        fontSize={10.5}
        textAnchor="middle"
      >
        {payload.value}
      </text>
      <text
        dy={26}
        fill="hsl(var(--text-secondary))"
        fontFamily='"IBM Plex Mono", ui-monospace, monospace'
        fontSize={9}
        opacity={0.7}
        textAnchor="middle"
      >
        ({m?.protocol})
      </text>
    </g>
  );
}

export const FigMilestones = () => (
  <ResponsiveContainer height={330} width="100%">
    <BarChart data={MILESTONES} margin={{ top: 24, right: 16, bottom: 34, left: 8 }}>
      <CartesianGrid stroke={GRID} vertical={false} />
      <XAxis
        axisLine={{ stroke: AXIS_LINE }}
        dataKey="label"
        height={44}
        interval={0}
        tick={<MilestoneTick />}
        tickLine={false}
      />
      <YAxis
        axisLine={false}
        domain={[0, 1]}
        label={{
          angle: -90,
          position: "insideLeft",
          style: AXIS_LABEL,
          value: "aggregate score",
        }}
        tick={TICK}
        tickLine={false}
        width={52}
      />
      <Tooltip
        content={<ChartTooltip />}
        cursor={{ fill: "hsl(var(--text-primary) / 0.05)" }}
      />
      <Bar dataKey="score" isAnimationActive radius={[3, 3, 0, 0]}>
        {MILESTONES.map((m) => (
          <Cell fill={milestoneColor(m)} key={m.label} />
        ))}
        <LabelList
          dataKey="score"
          fill={LABEL_FILL}
          fontSize={11}
          fontWeight={600}
          formatter={(v: any) => Number(v).toFixed(3)}
          position="top"
        />
      </Bar>
    </BarChart>
  </ResponsiveContainer>
);

export const FigSeeds = () => (
  <GroupedBar
    data={{ tactical: SEEDS_4D.tactical, heuristic: SEEDS_4D.heuristic }}
    groups={SEEDS_4D.groups}
    series={[
      { key: "tactical", label: "vs tactical", color: SERIES.tactical },
      { key: "heuristic", label: "vs heuristic", color: SERIES.heuristic },
    ]}
    unit="%"
    ylabel="win rate (%)"
    ymax={100}
  />
);
