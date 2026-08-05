/* ChessReasoner-120M — three.js scenes.
   Two canvases share one builder: a compact auto-orbiting version in the hero,
   and the full interactive pipeline in Fig. 2. Flow runs left to right along
   +X: board plane -> token span -> 18 decoder blocks -> four heads. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const INK = 0x07090c;
const AMBER = 0xe0a03c;
const CYAN = 0x52b5c4;
const BONE = 0xe8e3d8;
const SLATE = 0x38434f;

const FEN = 'r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R';
const OCC = (() => {
  const o = new Array(64).fill(null);
  FEN.split('/').forEach((rankStr, i) => {
    const rank = 7 - i; let file = 0;
    for (const ch of rankStr) {
      if (/\d/.test(ch)) { file += +ch; continue; }
      o[rank * 8 + file] = ch; file += 1;
    }
  });
  return o;
})();

function edged(mesh, color, opacity = 0.55) {
  const line = new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity })
  );
  mesh.add(line);
  return mesh;
}

function build(compact) {
  const root = new THREE.Group();
  const groups = {};

  /* ---- board plane ---------------------------------------------------- */
  const board = new THREE.Group();
  const cellGeo = new THREE.BoxGeometry(0.72, 0.72, 0.16);
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const sq = r * 8 + f;
      const occupied = !!OCC[sq];
      const m = new THREE.Mesh(cellGeo, new THREE.MeshStandardMaterial({
        color: occupied ? AMBER : ((r + f) % 2 ? 0x151d25 : 0x1e2831),
        emissive: occupied ? AMBER : 0x000000,
        emissiveIntensity: occupied ? 0.5 : 0,
        roughness: 0.6, metalness: 0.1,
        transparent: true, opacity: occupied ? 1 : 0.85
      }));
      m.position.set(0, (r - 3.5) * 0.8, (f - 3.5) * 0.8);
      m.scale.z = occupied ? 2.4 : 1;
      board.add(m);
    }
  }
  board.position.x = compact ? -6.2 : -10.6;
  root.add(board);
  groups.board = board;

  /* ---- token span ------------------------------------------------------ */
  const stream = new THREE.Group();
  const tokGeo = new THREE.BoxGeometry(0.34, 0.34, 0.34);
  const N_TOK = 72;
  for (let i = 0; i < N_TOK; i++) {
    const isMarker = i === 0 || i === N_TOK - 1;
    const occupied = i > 0 && i <= 64 && !!OCC[i - 1];
    const m = new THREE.Mesh(tokGeo, new THREE.MeshStandardMaterial({
      color: isMarker ? CYAN : occupied ? AMBER : SLATE,
      emissive: isMarker ? CYAN : occupied ? AMBER : 0x000000,
      emissiveIntensity: isMarker || occupied ? 0.6 : 0,
      roughness: 0.45, transparent: true, opacity: 0.95
    }));
    const t = i / (N_TOK - 1);
    const span = compact ? 4.2 : 6.2;
    m.position.set(t * span, Math.sin(t * Math.PI * 3) * 0.55, Math.cos(t * Math.PI * 3) * 0.55);
    m.userData.base = m.position.clone();
    m.userData.phase = t;
    stream.add(m);
  }
  stream.position.x = compact ? -5.4 : -9.6;
  root.add(stream);
  groups.stream = stream;

  /* ---- 18 decoder blocks ---------------------------------------------- */
  const stack = new THREE.Group();
  const plateGeo = new THREE.BoxGeometry(0.26, 5.4, 5.4);
  const N_L = 18;
  for (let i = 0; i < N_L; i++) {
    const m = new THREE.Mesh(plateGeo, new THREE.MeshStandardMaterial({
      color: 0x1d2935, roughness: 0.35, metalness: 0.25,
      transparent: true, opacity: 0.4
    }));
    m.position.x = i * (compact ? 0.42 : 0.5);
    edged(m, i === 0 || i === N_L - 1 ? AMBER : 0x2b3944, 0.5);
    m.userData.idx = i;
    stack.add(m);
  }
  stack.position.x = compact ? -2.2 : -2.6;
  root.add(stack);
  groups.stack = stack;

  /* ---- heads ----------------------------------------------------------- */
  const heads = new THREE.Group();
  const headGeo = new THREE.BoxGeometry(0.22, 1.5, 3.2);
  const defs = [
    { y: 2.4, c: AMBER, keep: true },
    { y: 0.8, c: CYAN, keep: false },
    { y: -0.8, c: CYAN, keep: false },
    { y: -2.4, c: CYAN, keep: false }
  ];
  defs.forEach((d) => {
    const m = new THREE.Mesh(headGeo, new THREE.MeshStandardMaterial({
      color: d.keep ? 0x2a2113 : 0x11242a,
      emissive: d.c, emissiveIntensity: d.keep ? 0.35 : 0.16,
      roughness: 0.4, transparent: true, opacity: d.keep ? 0.95 : 0.6
    }));
    m.position.set(0, d.y, 0);
    edged(m, d.c, 0.8);
    m.userData.keep = d.keep;
    heads.add(m);
  });
  heads.position.x = compact ? 6.2 : 8.2;
  root.add(heads);
  groups.heads = heads;

  /* ---- connectors ------------------------------------------------------ */
  const links = new THREE.Group();
  defs.forEach((d) => {
    const pts = [
      new THREE.Vector3(compact ? 5.4 : 6.9, 0, 0),
      new THREE.Vector3(compact ? 5.8 : 7.5, d.y * 0.55, 0),
      new THREE.Vector3(compact ? 6.08 : 8.05, d.y, 0)
    ];
    const curve = new THREE.CatmullRomCurve3(pts);
    const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
    links.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
      color: d.c, transparent: true, opacity: d.keep ? 0.85 : 0.4
    })));
  });
  root.add(links);
  groups.links = links;

  return { root, groups, stream, stack, heads };
}

function makeScene(canvas, compact) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(INK, compact ? 0.032 : 0.024);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
  camera.position.set(compact ? 13 : 17, compact ? 7 : 8, compact ? 15 : 20);

  scene.add(new THREE.AmbientLight(0x7c8b99, 1.05));
  const key = new THREE.DirectionalLight(BONE, 1.5); key.position.set(8, 14, 12); scene.add(key);
  const rim = new THREE.DirectionalLight(AMBER, 0.9); rim.position.set(-12, 4, -8); scene.add(rim);
  const fill = new THREE.PointLight(CYAN, 26, 40); fill.position.set(10, -4, 6); scene.add(fill);

  const built = build(compact);
  scene.add(built.root);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = false;   // see idle arc below
  controls.enablePan = false;
  controls.enableZoom = !compact;
  controls.minDistance = 12; controls.maxDistance = 52;
  controls.minPolarAngle = 0.5; controls.maxPolarAngle = 2.1;
  controls.target.set(compact ? -0.4 : -1.2, 0, 0);

  // Free auto-rotation swings a long, thin model through angles where the
  // board plane fills the foreground and the stack is edge-on. Instead the
  // camera drifts along a bounded arc that always keeps the pipeline readable,
  // and hands control over entirely once the user grabs it.
  const ARC = {
    radius: compact ? 19 : 26,
    az: compact ? -0.75 : -0.85, azSwing: compact ? 0.5 : 0.42,
    el: 0.38, elSwing: 0.1
  };
  let idle = true, idleTimer = null;
  controls.addEventListener('start', () => { idle = false; clearTimeout(idleTimer); });
  controls.addEventListener('end', () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { idle = true; }, 4500);
  });
  function driftCamera(t) {
    const az = ARC.az + Math.sin(t * 0.14) * ARC.azSwing;
    const el = ARC.el + Math.sin(t * 0.09) * ARC.elSwing;
    const r = ARC.radius;
    camera.position.set(
      controls.target.x + r * Math.cos(el) * Math.sin(az),
      controls.target.y + r * Math.sin(el),
      controls.target.z + r * Math.cos(el) * Math.cos(az)
    );
  }
  driftCamera(0);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== w * renderer.getPixelRatio() || canvas.height !== h * renderer.getPixelRatio()) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }

  let running = true;
  const vis = new IntersectionObserver((e) => { running = e[0].isIntersecting; }, { threshold: 0.01 });
  vis.observe(canvas);

  const clock = new THREE.Clock();
  function tick() {
    requestAnimationFrame(tick);
    if (!running) return;
    resize();
    const t = clock.getElapsedTime();

    // token stream: a slow travelling pulse along the span
    built.stream.children.forEach((m) => {
      const p = (t * 0.22 + m.userData.phase) % 1;
      const pulse = Math.exp(-Math.pow((p - 0.5) * 5, 2));
      m.position.y = m.userData.base.y + pulse * 0.35;
      m.material.emissiveIntensity = 0.15 + pulse * 1.2;
    });

    // stack: a wave of activation sweeping through the layers
    built.stack.children.forEach((m) => {
      const p = ((t * 0.3) - m.userData.idx * 0.045) % 1;
      const w = Math.exp(-Math.pow((p - 0.5) * 6, 2));
      m.material.opacity = 0.32 + w * 0.42;
    });

    built.heads.children.forEach((m, i) => {
      m.material.emissiveIntensity = (m.userData.keep ? 0.3 : 0.12)
        + Math.sin(t * 1.5 + i) * 0.08;
    });

    if (idle) driftCamera(t);
    controls.update();
    renderer.render(scene, camera);
  }
  resize();
  tick();

  return { built, controls, camera };
}

/* ---------- hero ------------------------------------------------------- */
const heroCanvas = document.getElementById('hero-canvas');
if (heroCanvas) makeScene(heroCanvas, true);

/* ---------- Fig. 2 interactive ----------------------------------------- */
const mainCanvas = document.getElementById('scene-canvas');
if (mainCanvas) {
  const { built, controls } = makeScene(mainCanvas, false);
  const readout = document.getElementById('scene-readout');

  const COPY = {
    all: '<b>Whole model.</b> The board enters as 72 tokens, is consumed by an 18-layer stack of 119.6M parameters, and exits through four heads. The three cold heads are auxiliary — they exist only while training.',
    board: '<b>Board plane.</b> All 64 squares, always emitted, always in a1→h8 order. Raised cells are occupied; 57% of a typical position is empty, which is why the loss weights the plane at 0.1 rather than masking or fully supervising it.',
    stream: '<b>Token span.</b> Exactly 72 tokens: one opening marker, 64 square contents, side to move, four castling slots, en passant, one closing marker. Fixed length means square i is always at a computable offset — file (+1), rank (+8) and diagonal (+9/+7) neighbours become constant strides.',
    stack: '<b>18 decoder blocks.</b> Pre-norm, GQA with 12 query and 4 key-value heads, SwiGLU 2048, RMSNorm accumulated in fp32. Deeper than the usual 120M configuration on purpose: serializing a search is sequential composition, and depth buys composition steps that width does not.',
    heads: '<b>Four heads.</b> Warm is the LM head — the only one that ships. Cold are board, value and policy: 2,251,632 parameters that shape the residual stream during training and are deleted before inference.',
    inference: '<b>Inference path.</b> What actually runs on the device: board in, 18 blocks, LM head, prose out. No engine, no search, no auxiliary heads, no scaffolded input. About 120 MB at int8 with an 18.9 MB KV cache.'
  };

  function setStage(stage) {
    const show = {
      all: ['board', 'stream', 'stack', 'heads', 'links'],
      board: ['board'],
      stream: ['stream'],
      stack: ['stack'],
      heads: ['heads', 'links'],
      inference: ['board', 'stream', 'stack', 'heads', 'links']
    }[stage];

    Object.entries(built.groups).forEach(([k, g]) => {
      const on = show.includes(k);
      g.visible = on;
      g.traverse((o) => {
        if (!o.material) return;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((mat) => {
          if (mat.userData.baseOpacity === undefined) mat.userData.baseOpacity = mat.opacity;
          mat.opacity = mat.userData.baseOpacity;
        });
      });
    });

    if (stage === 'inference') {
      // dim everything the device never loads
      built.heads.children.forEach((m) => {
        if (!m.userData.keep) { m.material.opacity = 0.07; m.material.emissiveIntensity = 0.02; }
      });
      built.groups.links.children.forEach((l, i) => { l.material.opacity = i === 0 ? 0.9 : 0.06; });
    }
    readout.innerHTML = COPY[stage];

  }

  document.querySelectorAll('.chip[data-stage]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.chip[data-stage]').forEach((b) => b.classList.remove('on'));
      btn.classList.add('on');
      setStage(btn.dataset.stage);
    });
  });
}
