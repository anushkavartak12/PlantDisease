import type { DiseaseSim } from '../../data/crops';

export interface PlantLook {
  crop: string;
  /** 0..1 growth progress */
  growth: number;
  /** strongest visible disease */
  disease?: { sim: DiseaseSim; severity: number };
  /** 0..1 how thirsty (droop + dull colour) */
  thirst?: number;
  /** 0..1 insect population around the plant */
  bugs?: number;
}

type Symptom = { sim: DiseaseSim; severity: number } | undefined;

const GROUND = 222;

export function Plant({ crop, growth, disease, thirst = 0, bugs = 0 }: PlantLook) {
  const sev = disease && disease.severity > 0.04 ? disease : undefined;
  const droop = Math.max(thirst * 0.6, sev?.sim.pattern === 'wilt' ? sev.severity : 0);
  const Body = BODIES[crop] ?? Tomato;
  return (
    <svg className="plant" viewBox="0 0 200 240" aria-hidden="true">
      <ellipse cx="100" cy={GROUND + 6} rx="62" ry="10" fill="#7a4b28" />
      <ellipse cx="100" cy={GROUND + 3} rx="54" ry="7" fill="#946036" />
      {growth < 0.06 ? (
        <Seed growth={growth} />
      ) : (
        <g className={droop > 0.4 ? 'sway slow' : 'sway'} style={{ transformOrigin: `100px ${GROUND}px` }}>
          <Body g={growth} s={sev} droop={droop} thirst={thirst} />
        </g>
      )}
      {bugs > 0.15 && <Bugs n={Math.round(bugs * 8)} />}
    </svg>
  );
}

function Seed({ growth }: { growth: number }) {
  return (
    <g>
      <ellipse cx="100" cy={GROUND - 2} rx="7" ry="5" fill="#c79a5b" />
      {growth > 0.02 && (
        <path d={`M100 ${GROUND - 4} q-2 -10 4 -16`} stroke="#6fbf4a" strokeWidth="3" fill="none" strokeLinecap="round" />
      )}
    </g>
  );
}

function Bugs({ n }: { n: number }) {
  return (
    <g className="bugs">
      {Array.from({ length: n }, (_, i) => (
        <g key={i} className="bug" style={{ animationDelay: `${-i * 0.7}s`, transformOrigin: `${70 + (i * 23) % 60}px ${80 + (i * 37) % 90}px` }}>
          <circle cx={70 + (i * 23) % 60} cy={80 + (i * 37) % 90} r="2.2" fill="#fafafa" stroke="#999" strokeWidth="0.5" />
        </g>
      ))}
    </g>
  );
}

// ---------- leaf primitives ----------

const HEALTHY = '#5fae3c';
const HEALTHY_DARK = '#3f8a2e';

function rand(seed: number) {
  const x = Math.sin(seed * 999.1) * 43758.5453;
  return x - Math.floor(x);
}

function leafColor(s: Symptom, thirst: number) {
  let c = HEALTHY;
  if (s && ['mottle', 'wilt', 'curl'].includes(s.sim.pattern)) c = mix(HEALTHY, '#d9cf4a', s.severity * 0.8);
  if (s?.sim.pattern === 'streaks' && s.sim.kind !== 'fungal') c = mix(HEALTHY, '#e0d27a', s.severity * 0.6);
  if (thirst > 0.3) c = mix(c, '#a7a24a', (thirst - 0.3) * 0.8);
  return c;
}

interface LeafProps {
  x: number;
  y: number;
  angle: number;
  len: number;
  width: number;
  seed: number;
  s: Symptom;
  thirst: number;
  droop: number;
}

/** Broad leaf pointing along `angle` (degrees, 0 = right). */
function Leaf({ x, y, angle, len, width, seed, s, thirst, droop }: LeafProps) {
  const curl = s?.sim.pattern === 'curl' ? s.severity : 0;
  const side = Math.cos((angle * Math.PI) / 180) >= 0 ? 1 : -1;
  const a = angle + side * droop * 50;
  const w = width * (1 - curl * 0.5);
  const l = len * (1 - curl * 0.3);
  return (
    <g transform={`translate(${x} ${y}) rotate(${a})`}>
      <path
        d={`M0 0 Q${l * 0.45} ${-w * (1 + curl)} ${l} ${curl * w * 0.6} Q${l * 0.5} ${w} 0 0 Z`}
        fill={leafColor(s, thirst)}
        stroke={HEALTHY_DARK}
        strokeWidth="0.8"
      />
      <path d={`M1 0 Q${l * 0.5} ${-w * 0.15} ${l * 0.92} 0`} stroke={HEALTHY_DARK} strokeWidth="0.8" fill="none" opacity="0.6" />
      {s && <Marks s={s} l={l} w={w} seed={seed} />}
    </g>
  );
}

/** Long grass blade (rice). */
function Blade({ x, y, angle, len, seed, s, thirst, droop, width = 5 }: LeafProps) {
  const bend = 0.35 + droop * 0.5;
  const dir = Math.cos((angle * Math.PI) / 180) >= 0 ? 1 : -1;
  const ex = Math.cos((angle * Math.PI) / 180) * len;
  const ey = Math.sin((angle * Math.PI) / 180) * len;
  const cx = ex * 0.4;
  const cy = ey * 0.8;
  const tipX = ex + dir * len * bend * 0.4;
  const tipY = ey + len * bend;
  const color = leafColor(s, thirst);
  return (
    <g transform={`translate(${x} ${y})`}>
      <path
        d={`M${-width / 2} 0 Q${cx} ${cy} ${tipX} ${tipY} Q${cx + width} ${cy + 2} ${width / 2} 0 Z`}
        fill={color}
        stroke={HEALTHY_DARK}
        strokeWidth="0.6"
      />
      {s && s.severity > 0.05 && <BladeMarks s={s} from={[cx, cy]} to={[tipX, tipY]} seed={seed} />}
    </g>
  );
}

function Marks({ s, l, w, seed }: { s: NonNullable<Symptom>; l: number; w: number; seed: number }) {
  const { pattern, color } = s.sim;
  const n = Math.round(s.severity * (pattern === 'pustules' ? 14 : 6));
  const pts = Array.from({ length: n }, (_, i) => {
    const px = l * (0.18 + rand(seed + i) * 0.65);
    const py = (rand(seed * 3 + i) - 0.5) * w * 0.9 * Math.sin((px / l) * Math.PI);
    return [px, py] as const;
  });
  switch (pattern) {
    case 'pustules':
      return <g fill={color}>{pts.map(([px, py], i) => <circle key={i} cx={px} cy={py} r={1.2} />)}</g>;
    case 'rings':
      return (
        <g>
          {pts.map(([px, py], i) => (
            <g key={i}>
              <circle cx={px} cy={py} r={3.4} fill="#e6d38a" opacity="0.8" />
              <circle cx={px} cy={py} r={2.4} fill={color} />
              <circle cx={px} cy={py} r={1.3} fill="none" stroke="#e6d38a" strokeWidth="0.5" />
            </g>
          ))}
        </g>
      );
    case 'patches':
      return (
        <g fill={color} opacity="0.85">
          {pts.slice(0, Math.ceil(n / 2)).map(([px, py], i) => (
            <ellipse key={i} cx={px} cy={py} rx={4 + s.severity * 5} ry={3 + s.severity * 3} />
          ))}
        </g>
      );
    case 'streaks':
      return (
        <g stroke={color} strokeWidth="1.6" strokeLinecap="round">
          {pts.map(([px, py], i) => <line key={i} x1={px - 4} y1={py} x2={px + 4} y2={py * 0.8} />)}
        </g>
      );
    case 'mottle':
      return (
        <g fill={color} opacity="0.7">
          {pts.map(([px, py], i) => <ellipse key={i} cx={px} cy={py} rx={4} ry={2.4} />)}
        </g>
      );
    case 'wilt':
    case 'curl':
      return (
        <path d={`M${l * 0.6} ${-w * 0.5} Q${l * 0.85} 0 ${l} 0`} stroke={color} strokeWidth={1 + s.severity * 2} fill="none" opacity="0.8" />
      );
    default:
      return (
        <g>
          {pts.map(([px, py], i) => (
            <g key={i}>
              <circle cx={px} cy={py} r={2.6} fill="#f0dc8a" opacity="0.7" />
              <circle cx={px} cy={py} r={1.7} fill={color} />
            </g>
          ))}
        </g>
      );
  }
}

function BladeMarks({ s, from, to, seed }: { s: NonNullable<Symptom>; from: [number, number]; to: [number, number]; seed: number }) {
  const n = Math.round(s.severity * 5);
  return (
    <g fill={s.sim.color} stroke={s.sim.color}>
      {Array.from({ length: n }, (_, i) => {
        const t = 0.2 + rand(seed + i) * 0.7;
        const x = from[0] + (to[0] - from[0]) * t;
        const y = from[1] + (to[1] - from[1]) * t;
        return s.sim.pattern === 'streaks' ? (
          <line key={i} x1={x} y1={y - 3} x2={x + 1} y2={y + 3} strokeWidth="2" strokeLinecap="round" />
        ) : (
          <ellipse key={i} cx={x} cy={y} rx="1.4" ry="2.6" stroke="none" />
        );
      })}
    </g>
  );
}

// ---------- crop bodies ----------

interface BodyProps {
  g: number;
  s: Symptom;
  droop: number;
  thirst: number;
}

function Tomato({ g, s, droop, thirst }: BodyProps) {
  const h = 30 + g * 150;
  const nodes = Math.max(1, Math.round(g * 7));
  const top = GROUND - h * (1 - droop * 0.15);
  return (
    <g>
      <path d={`M100 ${GROUND} C96 ${GROUND - h * 0.4} 104 ${GROUND - h * 0.7} 100 ${top}`} stroke="#4f8f2f" strokeWidth={3 + g * 3} fill="none" strokeLinecap="round" />
      {Array.from({ length: nodes }, (_, i) => {
        const y = GROUND - ((i + 1) / (nodes + 1)) * h;
        const side = i % 2 ? 1 : -1;
        const ang = side > 0 ? -25 : 205;
        return (
          <g key={i}>
            <Leaf x={100} y={y} angle={ang} len={30 + g * 24} width={12 + g * 6} seed={i + 1} s={s} thirst={thirst} droop={droop} />
            <Leaf x={100} y={y - 6} angle={ang + side * -35} len={20 + g * 12} width={9} seed={i + 11} s={s} thirst={thirst} droop={droop} />
          </g>
        );
      })}
      <Leaf x={100} y={top + 2} angle={-80} len={16} width={6} seed={42} s={s} thirst={thirst} droop={droop} />
      {g > 0.55 && [0, 1, 2].map((i) => <Flower key={i} x={88 + i * 12} y={top + 30 + i * 18} />)}
      {g > 0.75 &&
        [0, 1, 2, 3].map((i) => (
          <circle key={i} cx={82 + (i % 2) * 34} cy={top + 45 + i * 16} r={5 + (g - 0.75) * 24} fill={g > 0.9 ? '#e0432f' : mix('#7cbf4a', '#e0432f', (g - 0.75) / 0.15)} stroke="#a82e20" strokeWidth="0.6" />
        ))}
    </g>
  );
}

function Cotton({ g, s, droop, thirst }: BodyProps) {
  const h = 30 + g * 140;
  const nodes = Math.max(1, Math.round(g * 6));
  const top = GROUND - h;
  return (
    <g>
      <path d={`M100 ${GROUND} L100 ${top}`} stroke="#7a5a3a" strokeWidth={3 + g * 3} strokeLinecap="round" />
      {Array.from({ length: nodes }, (_, i) => {
        const y = GROUND - ((i + 1) / (nodes + 1)) * h;
        const side = i % 2 ? 1 : -1;
        const bx = 100 + side * (18 + g * 14);
        return (
          <g key={i}>
            <path d={`M100 ${y} Q${100 + side * 10} ${y - 4} ${bx} ${y - 10}`} stroke="#7a5a3a" strokeWidth="2" fill="none" />
            <Leaf x={bx} y={y - 10} angle={side > 0 ? -30 : 210} len={30 + g * 14} width={18} seed={i + 3} s={s} thirst={thirst} droop={droop} />
            <Leaf x={100 + side * 6} y={y - 2} angle={side > 0 ? 15 : 165} len={22 + g * 10} width={14} seed={i + 23} s={s} thirst={thirst} droop={droop} />
            {g > 0.55 && g < 0.8 && <Flower x={bx} y={y - 16} petal="#f7ecc0" />}
            {g >= 0.8 && <Boll x={bx} y={y - 16} open={Math.min(1, (g - 0.8) / 0.15)} />}
          </g>
        );
      })}
      <Leaf x={100} y={top} angle={-90} len={20} width={12} seed={77} s={s} thirst={thirst} droop={droop} />
    </g>
  );
}

function Soybean({ g, s, droop, thirst }: BodyProps) {
  const h = 25 + g * 100;
  const nodes = Math.max(1, Math.round(g * 6));
  return (
    <g>
      <path d={`M100 ${GROUND} L100 ${GROUND - h}`} stroke="#5f8f2f" strokeWidth={3 + g * 2} strokeLinecap="round" />
      {Array.from({ length: nodes }, (_, i) => {
        const y = GROUND - ((i + 1) / (nodes + 1)) * h;
        const side = i % 2 ? 1 : -1;
        const bx = 100 + side * (22 + g * 12);
        return (
          <g key={i}>
            <path d={`M100 ${y} L${bx} ${y - 12}`} stroke="#5f8f2f" strokeWidth="1.6" />
            {[-35, 0, 35].map((d, k) => (
              <Leaf key={k} x={bx} y={y - 12} angle={(side > 0 ? -20 : 200) + d} len={16 + g * 6} width={9} seed={i * 3 + k} s={s} thirst={thirst} droop={droop} />
            ))}
            {g > 0.7 && <Pod x={100 + side * 8} y={y + 2} ripe={g > 0.9} />}
          </g>
        );
      })}
    </g>
  );
}

function Rice({ g, s, droop, thirst }: BodyProps) {
  const tillers = 3 + Math.round(g * 6);
  const len = 30 + g * 120;
  return (
    <g>
      {Array.from({ length: tillers }, (_, i) => {
        const ang = -90 + (i - (tillers - 1) / 2) * (60 / tillers);
        return <Blade key={i} x={100 + (i - tillers / 2) * 2} y={GROUND} angle={ang} len={len * (0.8 + rand(i) * 0.3)} width={5} seed={i + 5} s={s} thirst={thirst} droop={droop} />;
      })}
      {g > 0.7 &&
        [-1, 0, 1].map((k) => {
          const x = 100 + k * 16;
          const y = GROUND - len * 0.72;
          const ripe = Math.min(1, (g - 0.7) / 0.25);
          return (
            <path key={k} d={`M${x} ${y + 20} q${k * 6 + 4} -24 ${k * 10 + 16} -6`} stroke={mix('#9ccf5a', '#e8b93a', ripe)} strokeWidth="5" strokeDasharray="3 2" fill="none" strokeLinecap="round" />
          );
        })}
    </g>
  );
}

function Potato({ g, s, droop, thirst }: BodyProps) {
  const h = 20 + g * 80;
  const stems = 2 + Math.round(g * 3);
  return (
    <g>
      {g > 0.75 &&
        [-24, 4, 26].map((dx, k) => (
          <ellipse key={k} cx={100 + dx} cy={GROUND + 4} rx={7 + k} ry={5} fill="#c9a26b" stroke="#8a6a3a" strokeWidth="1" />
        ))}
      {Array.from({ length: stems }, (_, i) => {
        const ang = -90 + (i - (stems - 1) / 2) * 22;
        const r = (ang * Math.PI) / 180;
        const ex = 100 + Math.cos(r) * h;
        const ey = GROUND + Math.sin(r) * h;
        return (
          <g key={i}>
            <path d={`M100 ${GROUND} Q${100 + Math.cos(r) * h * 0.3} ${GROUND - h * 0.5} ${ex} ${ey}`} stroke="#5f8f2f" strokeWidth={2 + g * 1.5} fill="none" />
            {[0.45, 0.7, 0.95].map((f, k) => {
              const lx = 100 + (ex - 100) * f;
              const ly = GROUND + (ey - GROUND) * f;
              return [-1, 1].map((side) => (
                <Leaf key={`${k}${side}`} x={lx} y={ly} angle={ang + side * 70} len={12 + g * 6} width={8} seed={i * 6 + k * 2 + side} s={s} thirst={thirst} droop={droop} />
              ));
            })}
            <Leaf x={ex} y={ey} angle={ang} len={13 + g * 6} width={8} seed={i + 40} s={s} thirst={thirst} droop={droop} />
            {g > 0.5 && g < 0.85 && <Flower x={ex} y={ey - 6} petal="#e8e0f5" />}
          </g>
        );
      })}
    </g>
  );
}

const BODIES: Record<string, (p: BodyProps) => JSX.Element> = {
  tomato: Tomato,
  cotton: Cotton,
  soybean: Soybean,
  rice: Rice,
  potato: Potato,
};

function Flower({ x, y, petal = '#ffd23f' }: { x: number; y: number; petal?: string }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      {[0, 72, 144, 216, 288].map((a) => (
        <ellipse key={a} cx="0" cy="-3.5" rx="2" ry="3.5" fill={petal} transform={`rotate(${a})`} />
      ))}
      <circle r="1.8" fill="#e08a1e" />
    </g>
  );
}

function Boll({ x, y, open }: { x: number; y: number; open: number }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <circle r={5} fill="#7a8f3a" />
      {open > 0 && [0, 1, 2, 3].map((i) => <circle key={i} cx={Math.cos(i * 1.6) * 4 * open} cy={Math.sin(i * 1.6) * 4 * open - 2} r={4 * open + 1} fill="#fffdf5" stroke="#e7e1d2" strokeWidth="0.5" />)}
    </g>
  );
}

function Pod({ x, y, ripe }: { x: number; y: number; ripe: boolean }) {
  return <ellipse cx={x} cy={y} rx="3.4" ry="8" fill={ripe ? '#c9a45a' : '#8fbf4a'} stroke="#6b7d2f" strokeWidth="0.6" transform={`rotate(20 ${x} ${y})`} />;
}

function mix(a: string, b: string, t: number) {
  t = Math.max(0, Math.min(1, t));
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const ch = (shift: number) => Math.round(((pa >> shift) & 255) * (1 - t) + ((pb >> shift) & 255) * t);
  return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
}
