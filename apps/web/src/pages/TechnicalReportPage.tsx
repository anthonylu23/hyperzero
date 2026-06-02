import { Moon, Sun } from "lucide-react";
import { Link } from "react-router-dom";

import GlassSurface from "../components/GlassSurface";
import NavMenu from "../components/NavMenu";
import { VALIDATION } from "./reportData";
import {
  FigArch,
  FigComplexity,
  FigEval,
  FigLoss,
  FigMilestones,
  FigRanking,
  FigSeeds,
  FigValidation,
} from "./reportFigures";

type Theme = "dark" | "light";

const OPPS = ["random", "tactical", "heuristic", "mcts"] as const;

function ValidationTable() {
  return (
    <table className="report-table">
      <thead>
        <tr>
          <th>Variant</th>
          <th>Random</th>
          <th>Tactical</th>
          <th>Heuristic</th>
          <th>MCTS-32</th>
        </tr>
      </thead>
      <tbody>
        {VALIDATION.groups.map((g, gi) => (
          <tr key={g}>
            <td className="v">{g}</td>
            {OPPS.map((opp) => (
              <td key={opp}>
                {VALIDATION[opp][gi].toFixed(1)}
                <span className="cnt">{VALIDATION.counts[opp][gi]}</span>
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Figure({ children, caption }: { children: React.ReactNode; caption: React.ReactNode }) {
  return (
    <figure className="report-figure">
      {children}
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

export default function TechnicalReportPage({
  theme,
  onToggleTheme,
}: {
  theme: Theme;
  onToggleTheme: () => void;
}) {
  return (
    <>
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
          <nav className="glass-nav-inner" aria-label="Primary">
            <Link className="nav-brand nav-underline" to="/">
              HyperZero
            </Link>
            <div className="nav-controls">
              <NavMenu />
              <button
                aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
                aria-pressed={theme === "dark"}
                className="theme-toggle"
                onClick={onToggleTheme}
                type="button"
              >
                {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
              </button>
            </div>
          </nav>
        </GlassSurface>
      </header>

      <main className="page-shell report-page">
        <article className="report animate-fade-up">
          <h1>
            HyperZero: AlphaZero-Style Self-Play for{" "}
            <br />
            N-Dimensional Connect-K with a Universal Agent
          </h1>

          <div className="report-callout">
            <b>Abstract.</b> We study how AlphaZero-style self-play scales as the classic Connect-Four
            setting is generalized to a family of <i>N</i>-dimensional Connect-K games with gravity,
            tuned through dimensionality, board size, connect length, and gravity axis. We build a
            configurable game engine, non-neural baselines, and a PUCT self-play stack with a root
            tactical guard, then train specialist and cross-variant agents in 2D, 3D, and 4D. Our
            headline contribution is a <b>single universal policy–value transformer</b> that loads
            once and legally plays 2D, 3D, and 4D variants from one checkpoint. The promoted
            universal checkpoint scores{" "}
            <b>0.8221</b> on a held-out robust evaluation (4 variants × 4 opponents × 3 seeds × 64
            games) and is the only candidate in that block to clear every per-variant promotion floor.
            The enabling method is <b>teacher-replay anchored residual recovery</b>, which mixes a
            small fraction of teacher examples into repair training and stabilizes promotion where
            curriculum reweighting alone regressed. We report strong 3D results, a clear 4D
            fork-robustness limitation, and the controlled gaps that remain before paper-grade claims.
          </div>

          <div className="report-cards">
            <div className="report-card">
              <div className="n">0.8221</div>
              <div className="l">universal robust score, all promotion floors pass</div>
            </div>
            <div className="report-card">
              <div className="n">2D·3D·4D</div>
              <div className="l">variants from one checkpoint</div>
            </div>
            <div className="report-card">
              <div className="n">94.4%</div>
              <div className="l">3D specialist vs heuristic over 160 games</div>
            </div>
            <div className="report-card">
              <div className="n">1.46 GiB</div>
              <div className="l">peak validation VRAM on an 8 GiB RTX 3060 Ti</div>
            </div>
          </div>

          <nav className="report-toc" aria-label="Contents">
            <a href="#intro">1. Introduction</a>
            <a href="#game">2. Game family &amp; complexity</a>
            <a href="#system">3. System architecture</a>
            <a href="#method">4. Training methodology</a>
            <a href="#setup">5. Experimental setup</a>
            <a href="#results">6. Results</a>
            <a href="#ablations">7. Ablations &amp; negative results</a>
            <a href="#limits">8. Limitations</a>
            <a href="#gaps">9. Gaps &amp; future work</a>
            <a href="#repro">10. Reproducibility</a>
          </nav>

          <h2 id="intro">1. Introduction</h2>
          <p>
            AlphaZero showed that self-play reinforcement learning with neural-guided Monte Carlo Tree
            Search (MCTS) can master perfect-information games without handcrafted strategy. Most
            well-known demonstrations use fixed historical rule sets. HyperZero instead studies a{" "}
            <i>clean, parametric</i> game family — <b>N-dimensional Connect-K with gravity</b> — whose
            combinatorial complexity can be dialed directly through dimensionality, board shape,
            connect length, and gravity axis. This makes it a controlled testbed for the question:{" "}
            <em>
              how far does a practical AlphaZero-style implementation scale across dimensions, and
              which design choices make higher-D self-play tractable?
            </em>
          </p>
          <p>
            We make three contributions: (i) a configurable N-D Connect-K platform with correct
            line/symmetry machinery and a reusable evaluation harness; (ii) a clean scaling result —
            near-perfect play in 3D, and a concrete account of where 4D breaks down strategically
            rather than computationally; and (iii) a <b>dimension-conditioned universal agent</b> that
            plays 2D/3D/4D from one checkpoint, enabled by teacher-replay anchored recovery.
          </p>

          <h2 id="game">2. Game family &amp; complexity</h2>
          <p>
            A variant is defined by <code>(shape, K, gravity_axis)</code>. Moves always obey gravity: a
            player chooses an (N−1)-dimensional column and the piece falls to the lowest free cell. A
            win is K in a row along any axis, plane-diagonal, or hyper-diagonal direction. Complexity
            grows sharply with dimension: the action space and the number of winning lines both expand
            much faster than the board&apos;s linear size.
          </p>
          <Figure
            caption={
              <>
                <b>Figure 1.</b> Complexity scaling across the three primary variants. Legal actions
                grow from 7 (2D 6×7) to 16 (3D 4³) to 64 (4D 4⁴); board cells and the number of
                precomputed winning lines grow faster still. Counts are computed directly from{" "}
                <code>hyperzero.game.GameConfig</code>.
              </>
            }
          >
            <FigComplexity />
          </Figure>

          <h2 id="system">3. System architecture</h2>
          <p>
            The stack is layered and independently testable: a dependency-light engine (
            <code>hyperzero/game/</code>), four non-neural baselines (random, one-ply tactical,
            heuristic line-scorer, pure MCTS), a PUCT search (<code>hyperzero/search/puct.py</code>)
            with batched leaf inference, and policy–value models (
            <code>hyperzero/models/factory.py</code>).
          </p>
          <h3>3.1 Root tactical guard</h3>
          <p>
            PUCT root selection forces an immediately winning move when one exists and masks one-ply
            losing moves when a safe alternative remains. This eliminated the dominant one-ply blunder
            failure mode in 3D self-play and evaluation.
          </p>
          <h3>3.2 Specialist models</h3>
          <p>
            The factory provides MLP, CNN/ResNet, transformer, and <b>line-aware</b> variants (
            <code>line_mlp</code> and <code>line_resnet</code>) that add open-line feature planes.
            Line-aware features were the most repeated architectural win.
          </p>
          <h3>3.3 Universal policy–value transformer</h3>
          <p>
            The universal model (<code>hyperzero/models/universal_transformer.py</code>) runs one
            transformer over a global token, per-cell tokens, and per-action tokens carrying geometry
            metadata (normalized coordinates, rank, shape, gravity axis, connect length, ply, column
            fill). Policy logits are produced by <i>scoring action tokens</i>, so a single checkpoint
            emits 4, 7, 16, or 64 logits for different boards — no dimension-specific head.
          </p>

          <h2 id="method">4. Training methodology</h2>
          <p>
            Each iteration runs batched self-play, stores{" "}
            <code>(state, MCTS-visit policy, outcome)</code> with the originating{" "}
            <code>GameConfig</code>, and trains on a replay buffer sampled{" "}
            <b>balanced round-robin across variants</b> so cheap 2D games cannot dominate. Promotion is
            gated by a <b>weighted eval score with per-variant floors</b> — 2D strength cannot mask a
            4D regression. Key techniques studied:
          </p>
          <ul className="report-list">
            <li>
              <b>Symmetry augmentation</b> — gravity-preserving, action-label-safe.
            </li>
            <li>
              <b>Line-aware feature planes</b> and line-feature distillation runs.
            </li>
            <li>
              <b>Curriculum reweighting</b> of per-variant game counts across continuations.
            </li>
            <li>
              <b>Teacher-replay anchored &quot;residual recovery&quot;</b> — roughly 10% of each
              training batch sampled from a teacher replay at lr 2e-5, anchoring repair training to a
              stronger prior and reducing forgetting.
            </li>
            <li>
              <b>Heuristic-opponent injection</b> into self-play.
            </li>
            <li>Hyperparameter, capacity, replay, training-step, and sims-budget sweeps.</li>
          </ul>

          <h2 id="setup">5. Experimental setup</h2>
          <p>
            All training ran on a single NVIDIA <b>RTX 3060 Ti (8 GiB)</b> workstation (Fedora, conda{" "}
            <code>torch</code> env). Evaluation is isolated from training with fixed search budgets. The
            promotion-grade <b>robust evaluation</b> (2026-05-31) plays 5 candidate checkpoints × 4
            variants × 4 opponents × 3 seeds × 64 games (agent sims = 24, MCTS opponent sims = 32). The
            scored objective weights heuristic 0.55 / tactical 0.35 / random 0.10; MCTS-32 is reported
            as a diagnostic column. Floors: random ≥ 0.90, tactical ≥ 0.75, and per-variant heuristic
            floors (2D-6×7 ≥ 0.75, 3D ≥ 0.65, 4D ≥ 0.65).
          </p>
          <p className="report-foot">
            Resource envelope during validation: 391 GPU samples, mean util 93.3% (max 99%), max memory
            1458 MiB, max temp 63 °C. 3D training peaked at 3626 MiB; 4D self-play is CPU-bound at &lt;1
            GiB VRAM (≈ 8–10 s/game at 16 sims).
          </p>

          <h2 id="results">6. Results</h2>

          <h3>6.1 3D main target — near-perfect</h3>
          <p>
            The guarded 4×4×4 line-ResNet (120 iterations, 48 games/iter, 64 PUCT sims) beats every
            baseline under a 160-game-per-opponent final evaluation, and trains stably (total loss 3.44
            → 2.43).
          </p>
          <Figure
            caption={
              <>
                <b>Figure 2.</b> 3D stability run training losses (total / policy / value) per
                iteration, read from <code>phase4_3d_stability_guard…/train/metrics.jsonl</code>.
              </>
            }
          >
            <FigLoss />
          </Figure>
          <Figure
            caption={
              <>
                <b>Figure 3.</b> Train-time win rate vs baselines over the same run. Final 160-game
                evaluation: random 100%, tactical 97.5%, heuristic 94.4%, MCTS-32 99.4%.
              </>
            }
          >
            <FigEval />
          </Figure>

          <h3>6.2 Architecture comparison (v1, 2D)</h3>
          <p>
            Across 2D boards, ResNet is the strongest default; line-aware MLP helps the discriminating{" "}
            <i>heuristic</i> matchup; the plain transformer is competitive on 6×7 but collapses at 8×8
            under these budgets.
          </p>
          <Figure
            caption={
              <>
                <b>Figure 4.</b> Heuristic-opponent win rate (the discriminating baseline) by
                architecture. Random/tactical/MCTS often saturate; heuristic exposes the gap. The
                transformer&apos;s 8×8 collapse (0%) motivated ResNet/line-aware defaults for higher
                dimensions.
              </>
            }
          >
            <FigArch />
          </Figure>

          <h3>6.3 Universal agent — the headline</h3>
          <p>
            A single universal transformer legally plays all four configured variants and, after
            teacher-replay anchored recovery, clears every promotion floor. Figure 5 shows the
            winner&apos;s per-variant, per-opponent win rates under the 192-game robust protocol.
          </p>
          <Figure
            caption={
              <>
                <b>Figure 5.</b> Promoted universal checkpoint (
                <code>residual_recovery_teacher010_best</code>), 2026-05-31 robust evaluation, 192
                games/opponent. The one soft spot is 2D-4×4 vs MCTS-32 (57.8%).
              </>
            }
          >
            <FigValidation />
          </Figure>
          <ValidationTable />
          <Figure
            caption={
              <>
                <b>Figure 6.</b> Candidate ranking in the 2026-05-31 validation block. Green = passes
                all floors; red label = first failed floor. Only the teacher-replay anchored
                checkpoint passes everything, despite two rivals being within 0.02 on raw score —
                promotion is floor-limited, not score-limited.
              </>
            }
          >
            <FigRanking />
          </Figure>

          <h2 id="ablations">7. Ablations &amp; negative results</h2>
          <h3>7.1 Selection lineage &amp; the cost of curriculum-only repair</h3>
          <p>
            Curriculum reweighting alone (v2, v3) <i>reduced training loss but lowered external
            strength</i> below the earlier best, and never cleared floors. Teacher-replay anchored
            recovery is what finally produced a floor-passing checkpoint.
          </p>
          <Figure
            caption={
              <>
                <b>Figure 7.</b> Aggregate-score milestones along the universal track. Protocols
                differ: early/v2/v3 are train-time eval scores; the final bar is the 192-game robust
                validation score. The key pattern is curriculum-only regression followed by recovery;
                the absolute bar heights are not directly comparable.
              </>
            }
          >
            <FigMilestones />
          </Figure>
          <div className="report-callout">
            <b>Teacher-replay signal.</b> Under the strict 2026-05-31 block, the teacher-replay
            checkpoint passes all floors while the same recovery recipe without teacher replay fails
            the 2D-4×4 tactical floor, and the pure <code>teacher_anchor</code> /{" "}
            <code>distill</code> baselines fail the 4D heuristic floor. An <i>older</i> two-seed eval
            slightly favored no-teacher, so we frame this as a <b>promotion-stability</b> result, not
            yet a fully isolated multi-seed ablation (see §9).
          </div>

          <h3>7.2 Capacity scaling did not help</h3>
          <p>
            At fixed training budget, the medium universal model (192×3, ≈1.84 M params) matched or
            beat the large model (256×4, ≈4.21 M). Capacity was not the bottleneck; data/curriculum and
            anchoring were.
          </p>

          <h3>7.3 4D is a strategic wall, not a compute wall</h3>
          <p>
            4D training is numerically stable and cheap on VRAM, but specialist strength is
            seed-sensitive and fork-prone: the root guard stops one-ply blunders, not fork creation a
            few plies earlier.
          </p>
          <Figure
            caption={
              <>
                <b>Figure 8.</b> 4D specialist seed variance (best checkpoint, 40 games/opponent).
                Tactical/heuristic win rates swing widely across seeds 0–3; seed 2 is the promoted
                specialist. This motivated moving to the shared universal agent rather than open-ended
                specialist tuning.
              </>
            }
          >
            <FigSeeds />
          </Figure>

          <h2 id="limits">8. Limitations</h2>
          <ul className="report-list">
            <li>
              Promotion samples are modest (64 games/seed × 3 seeds, sims 24); <b>no confidence
              intervals yet</b> — the 2D-4×4 vs MCTS-32 cell (111/192) most needs them.
            </li>
            <li>4D heuristic/fork robustness is only partial; 4D specialists are seed-sensitive.</li>
            <li>&quot;Lower self-play loss ≠ stronger play&quot; recurred (v2/v3, train128).</li>
            <li>The teacher-replay comparison is not yet a matched multi-seed ablation.</li>
            <li>
              Reproducibility cleanup: one sweep trial failed to run; only the promoted checkpoint is
              synced into the public artifact set so far, while detailed residual-recovery metrics and
              the no-teacher control still need to be mirrored; <code>pip install -e .</code> needs
              flat-layout package discovery cleanup.
            </li>
          </ul>

          <h2 id="gaps">9. Gaps to close before the paper</h2>
          <ol className="report-list">
            <li>
              <b>Search-budget strength curve</b> (25/50/100/200/400 sims) on a fixed checkpoint —
              never run cleanly.
            </li>
            <li>
              <b>Symmetry on/off</b> and <b>curriculum-vs-direct</b> matched ablations.
            </li>
            <li>
              <b>Universal-vs-specialist transfer table</b> under one fixed budget.
            </li>
            <li>
              <b>Clean teacher-replay ablation</b> — matched multi-seed teacher vs no-teacher.
            </li>
            <li>
              <b>Confidence intervals</b> (Wilson/bootstrap) on every headline rate.
            </li>
            <li>
              <b>Value calibration &amp; policy-entropy</b> curves (already logged, never plotted).
            </li>
            <li>
              <b>Elo / cross-checkpoint matchup matrix</b> via <code>eval/tournament.py</code>.
            </li>
            <li>
              <b>4D throughput / memory-vs-N</b> extraction from telemetry CSVs.
            </li>
            <li>
              <b>Sync the residual-recovery + validation blocks</b> into the repository so all
              promotion evidence is reproducible locally.
            </li>
          </ol>

          <h2 id="repro">10. Reproducibility</h2>
          <p>
            Run artifacts record configs, seeds, <code>metrics.jsonl</code>, evaluation summaries, and
            paired GPU/CPU telemetry where available; sweeps also preserve full <code>command.json</code>{" "}
            / <code>trial.json</code> metadata. The promoted checkpoint shipped in the demo is{" "}
            <code>
              runs/universal_residual_followup_20260528/residual_recovery_teacher010_lr2e5_seed6604/checkpoints/best_by_eval_score.pt
            </code>
            , loaded by <code>hyperzero.server.agent_service.DEFAULT_CHECKPOINT</code>. The promotion
            evidence is the remote validation block{" "}
            <code>runs/universal_validation_block_20260531/</code> (
            <code>robust_s24_g64x3_seed7100.json</code>, <code>summary.md</code>).
          </p>

          <hr className="report-rule" />
          <p className="report-foot">
            Charts are rendered from static data extracted from checked artifacts and the remote
            2026-05-31 validation summary. Remaining paper work is to sync the raw validation JSON
            into this repository and add uncertainty intervals.
          </p>
        </article>
      </main>
    </>
  );
}
