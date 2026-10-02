import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { MeshBVH, acceleratedRaycast } from 'three-mesh-bvh';

const canvas = document.getElementById('scene');
const loaderEl = document.getElementById('loader');
const fillEl = document.getElementById('loader-fill');
const labelEl = document.getElementById('loader-label');

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const isMobile = window.matchMedia('(max-width: 900px)').matches;

/* ------------------------------------------------------------------ *
 * Renderer
 * ------------------------------------------------------------------ */

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: 'high-performance',
});
/* Starting quality; adaptQuality() below steps it down on GPUs that cannot
 * hold the frame rate. Above 1.5 the extra pixels cost more than they show. */
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;

const scene = new THREE.Scene();

/* Backdrop: a deep blue sweep. Also what the glass head refracts. */
scene.background = makeBackdrop();

const camera = new THREE.PerspectiveCamera(26, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 7);

/* ------------------------------------------------------------------ *
 * Lighting + environment
 * ------------------------------------------------------------------ */

/* Master brightness for the whole studio - panels and lamps alike. Turn this
 * down for a darker plate, up if the specimen disappears into the background.
 * Everything below is expressed relative to it. */
const STUDIO_LEVEL = 0.45;

/* A dark studio with a handful of bright panels, rather than three's evenly
 * lit RoomEnvironment. A glossy surface reflecting a uniformly white box goes
 * flat and pale; this keeps most of the reflection dark so the bright panels
 * land as streaks and the sculpted panel lines stay legible. */
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(makeStudioEnvironment(), 0.34).texture;
scene.environmentIntensity = 1.35 * STUDIO_LEVEL;

function makeStudioEnvironment() {
  const env = new THREE.Scene();

  const room = new THREE.Mesh(
    new THREE.BoxGeometry(24, 24, 24),
    new THREE.MeshBasicMaterial({ color: 0x03040a, side: THREE.BackSide })
  );
  env.add(room);

  /* [w, h, x, y, z, colour] - large, dim and heavily blurred. Small bright
   * panels mirror straight back off a smooth shell as legible rectangles;
   * these are wide and low-intensity so they read as soft gradients instead. */
  const panels = [
    [12, 7, 0, 7.5, 4.5, new THREE.Color(0.85, 0.92, 1.15)],  // main softbox, high front
    [3.0, 13, -8, 1.5, 3.5, new THREE.Color(0.55, 0.8, 1.05)],// cyan wash, left
    [2.4, 11, 8.4, 2.5, -1.5, new THREE.Color(0.8, 0.4, 0.9)],// magenta wash, right rear
    [9, 4, 0, -6, 5, new THREE.Color(0.22, 0.36, 0.52)],      // cool bounce from below
    [10, 8, 1.5, 0.5, 10, new THREE.Color(0.3, 0.34, 0.44)],  // frontal fill on the face
    [12, 7, 0, 2, -9, new THREE.Color(0.34, 0.2, 0.46)],      // violet rim wash behind
  ];

  for (const [w, h, x, y, z, color] of panels) {
    const panel = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({
        color: color.clone().multiplyScalar(STUDIO_LEVEL),
        side: THREE.DoubleSide,
      })
    );
    panel.position.set(x, y, z);
    panel.lookAt(0, 0, 0);
    env.add(panel);
  }

  return env;
}

const key = new THREE.DirectionalLight(0xdfe9ff, 0.9 * STUDIO_LEVEL);
key.position.set(3.0, 2.6, 3.2);
scene.add(key);

const rim = new THREE.DirectionalLight(0x9fb4ff, 0.8 * STUDIO_LEVEL);
rim.position.set(-3.4, 1.6, -2.6);
scene.add(rim);

const fill = new THREE.DirectionalLight(0xffffff, 0.3 * STUDIO_LEVEL);
fill.position.set(-1.8, -1.4, 2.6);
scene.add(fill);

/* A single magenta bounce for the thin-film shift on the lips and jaw. There
 * was a cyan point light opposite it, but sitting behind and above the right
 * temple it burned a hard blob into the skull, so it is gone - the cyan in the
 * picture now comes from the environment wash alone. */
/* Sits well back: at close range the inverse-square falloff concentrates into
 * a hotspot under the cheekbone, which reads as a stray lamp rather than a
 * tint. Far away with the intensity scaled up, the same light is a wash. */
const warmBounce = new THREE.PointLight(0xff5fa8, 26 * STUDIO_LEVEL, 22, 2);
warmBounce.position.set(-4.6, -3.2, 4.4);
scene.add(warmBounce);

/* ------------------------------------------------------------------ *
 * Materials
 * ------------------------------------------------------------------ */

/* The GLB ships its own material - a dark blue-slate base at roughness 0 - and
 * that darkness is what makes the sculpted panel lines read. It is kept and
 * dressed as thin-film glass: on a black stage the shell goes near invisible
 * except where the film splits into cyan and magenta, which is the look of the
 * reference plate. */
function dressBody(source) {
  const material = new THREE.MeshPhysicalMaterial({
    color: (source.color ? source.color.clone() : new THREE.Color(0x4a4f6f)).multiplyScalar(0.5),
    metalness: 0.12,
    /* the sculpt has small flat plates all over the skull; at mirror
     * roughness they reflect the studio panels back as hot rectangles, so the
     * surface is deliberately held softer than the GLB authored it */
    roughness: 0.24,
    clearcoat: 0.55,
    clearcoatRoughness: 0.14,
    transmission: 0.72,
    thickness: 1.15,
    ior: 1.42,
    attenuationColor: new THREE.Color(0x2c4f7a),
    attenuationDistance: 1.4,
    iridescence: 1,
    iridescenceIOR: 1.9,
    iridescenceThicknessRange: [180, 980],
    envMapIntensity: 0.5,
    transparent: true,
    side: THREE.FrontSide,
  });

  source.dispose();
  return material;
}

/* The wireframe laid over the shell - the mesh grid from the reference. Drawn
 * additively so it only ever brightens, and it reads as drafting linework
 * rather than a second solid object. */
/* Two line layers share this shader: crisp feature edges carry the structure,
 * and a faint full mesh grid underneath supplies the texture. Both also carry
 * the cursor glow, so their uniforms are updated together. */
const wireMaterials = [];

function makeWireMaterial(base, color) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: color },
      uGlow: { value: new THREE.Color(0.85, 1.9, 2.4) },
      uHit: { value: new THREE.Vector3(0, 999, 0) },
      uRadius: { value: 0.55 },
      uStrength: { value: 0 },
      uBase: { value: base },
    },
    vertexShader: `
      varying vec3 vWorld;
      void main() {
        vec4 world = modelMatrix * vec4(position, 1.0);
        vWorld = world.xyz;
        gl_Position = projectionMatrix * viewMatrix * world;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform vec3 uGlow;
      uniform vec3 uHit;
      uniform float uRadius;
      uniform float uStrength;
      uniform float uBase;
      varying vec3 vWorld;

      void main() {
        float falloff = 1.0 - smoothstep(0.0, uRadius, distance(vWorld, uHit));
        float lit = pow(falloff, 1.7) * uStrength;
        vec3 color = mix(uColor, uGlow, lit);
        gl_FragColor = vec4(color, uBase + lit * 1.7);
      }
    `,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  wireMaterials.push(material);
  return material;
}

const edgeMaterial = makeWireMaterial(0.34, new THREE.Color(0.16, 0.52, 0.7));
const meshMaterial = makeWireMaterial(0.18, new THREE.Color(0.1, 0.34, 0.46));

/* Angle in degrees between two faces before the edge between them is drawn.
 * Lower = more lines. Below about 8 it collapses back into triangle soup. */
const WIRE_ANGLE = 16;

function addWireframe(mesh) {
  /* EdgesGeometry, not WireframeGeometry, for the structural pass: drawing
   * every triangle edge of a 47k-tri sculpt puts more lines on screen than
   * there are pixels to hold them, so it aliases into noise. This keeps only
   * edges where the surface actually creases - panel outlines and plate seams.
   * The full grid still goes in behind it, but faint enough to read as
   * texture rather than detail. */
  const grid = new THREE.LineSegments(new THREE.WireframeGeometry(mesh.geometry), meshMaterial);
  grid.frustumCulled = false;
  grid.renderOrder = 4;
  mesh.add(grid);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry, WIRE_ANGLE),
    edgeMaterial
  );
  edges.frustumCulled = false;
  edges.renderOrder = 5;
  mesh.add(edges);
}

/* Eyes: unlit white. Values above 1 push them past the bloom threshold, which
 * is what turns them into the lit slits from the reference. */
const eyeMaterial = new THREE.MeshBasicMaterial({
  color: new THREE.Color(1.85, 1.92, 2.05),
  toneMapped: false,
  side: THREE.DoubleSide,
});

/* ------------------------------------------------------------------ *
 * Model
 * ------------------------------------------------------------------ */

let bodyMaterial = null;
let bodyMesh = null;

/* Resting yaw. 0 faces the camera squarely; raise it for a three-quarter
 * presentation. The idle drift and pointer parallax are applied on top. */
const BASE_YAW = 0;

const head = new THREE.Group();
const pivot = new THREE.Group();
head.rotation.y = BASE_YAW;
pivot.add(head);
scene.add(pivot);

new GLTFLoader().load(
  'head.glb',
  (gltf) => {
    const model = gltf.scene;
    const eyes = [];

    model.traverse((child) => {
      if (!child.isMesh) return;
      child.frustumCulled = false;

      if (/eye/i.test(child.name)) {
        /* The export mirrors both eyeballs onto the same socket, so they are
         * only used as a measurement for the glowing pair built below. */
        child.visible = false;
        eyes.push(child);
      } else {
        child.material = bodyMaterial = dressBody(child.material);
        child.renderOrder = 3;
        addWireframe(child);
        /* The cursor pick raycasts this 40k-triangle mesh on every pointer
         * move; brute force took ~14 ms on the main thread, which stalled
         * scrolling and CSS animation along with the render. A BVH makes it
         * sub-millisecond. Built after the wireframe because it reorders the
         * index buffer. */
        child.geometry.boundsTree = new MeshBVH(child.geometry);
        child.raycast = acceleratedRaycast;
        bodyMesh = child;
      }
    });

    head.add(model);
    frameModel(model);
    addEyes(eyes, model);

    /* Handy for tweaking the look from the console. */
    window.vessel = { scene, camera, renderer, composer, model, eyes, bodyMaterial, wireMaterials, eyeMaterial, bloom };

    document.body.classList.add('ready');
    labelEl.textContent = 'Ready';
  },
  (event) => {
    if (!event.lengthComputable) return;
    const pct = Math.round((event.loaded / event.total) * 100);
    fillEl.style.width = pct + '%';
    labelEl.textContent = 'Loading assembly — ' + pct + '%';
  },
  (error) => {
    console.error('Could not load head.glb', error);
    labelEl.textContent = 'Model failed to load';
    fillEl.style.width = '100%';
  }
);

/* Centre the model on the origin and size it to a predictable height. */
const FRAME_OFFSET = 0.01;

const modelSize = new THREE.Vector3(1.5, 2.55, 1.5);

function frameModel(model) {
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const scale = 2.55 / size.y;
  model.scale.setScalar(scale);
  model.position.copy(center).multiplyScalar(-scale);

  /* Vertical placement in the frame, as a fraction of the bust's height.
   * Positive lifts it above centre, negative drops it. */
  model.position.y += size.y * scale * FRAME_OFFSET;

  modelSize.copy(size).multiplyScalar(scale);
  fitCamera();
}

/* Pull the camera back far enough that the bust fits whatever the aspect is —
 * without this the ears get cropped on portrait screens. */
function fitCamera() {
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
  const forHeight = (modelSize.y * 0.5) / Math.tan(vFov / 2);
  const forWidth = (modelSize.x * 0.5) / Math.tan(hFov / 2);
  const portrait = camera.aspect < 1;
  const margin = portrait ? 1.12 : 1.34;

  camera.position.z = Math.max(forHeight, forWidth) * margin;
  camera.position.y = 0;

  /* The copy owns the lower part of the hero, so fit the bust into the band
   * between the top of the hero and the headline instead of the whole viewport -
   * otherwise the headline lands on the neck and shoulders. */
  const band = heroBand();
  if (band) {
    const tanHalf = Math.tan(vFov / 2);
    const forBand = (modelSize.y * window.innerHeight) / band.height / (2 * tanHalf);
    /* Split hero: the head (ear to ear, not the shoulders) must also clear
     * the text columns on either side. */
    const forGap = band.width
      ? (modelSize.x * HEAD_WIDTH_FRACTION * window.innerHeight) / band.width / (2 * tanHalf)
      : 0;
    camera.position.z = Math.max(forBand, forGap, forWidth * margin);

    const unitsPerPixel = (2 * camera.position.z * tanHalf) / window.innerHeight;
    camera.position.y = (band.centre - window.innerHeight / 2) * unitsPerPixel;
  }

  camera.updateProjectionMatrix();
}

/* Matches the split hero in styles.css, where the copy flanks the bust
 * instead of sitting under it. */
const splitHero = window.matchMedia('(min-width: 901px) and (min-aspect-ratio: 1/1)');

/* Ear-to-ear width of the bust as a fraction of its shoulder width. */
const HEAD_WIDTH_FRACTION = 0.6;

/* The free vertical band in the hero, in page pixels, measured from the
 * layout so it tracks font sizes and viewport height. On the split desktop
 * hero the copy is beside the bust, so the band runs to the hero's foot;
 * stacked, it stops at the headline. Null if the hero is missing or the band
 * is too thin to use. */
function heroBand() {
  const hero = document.querySelector('.hero');
  const floor = splitHero.matches ? hero : document.querySelector('.headline');
  if (!hero || !floor) return null;

  const scroll = window.scrollY;
  const gap = 14;
  const floorRect = floor.getBoundingClientRect();
  const top = hero.getBoundingClientRect().top + scroll + gap;
  const bottom = (splitHero.matches ? floorRect.bottom : floorRect.top) + scroll - gap;
  const height = bottom - top;
  if (height < 160) return null;

  /* Horizontal room between the copy and the readouts on the split hero. */
  let width = 0;
  if (splitHero.matches) {
    const text = document.querySelector('.hero-text');
    const readouts = document.querySelector('.hero-readouts');
    if (text && readouts) {
      width = readouts.getBoundingClientRect().left - text.getBoundingClientRect().right - 2 * gap;
    }
  }

  return { height, width: Math.max(width, 0), centre: (top + bottom) / 2 };
}

/* Build a mirrored pair of glowing eyes from the socket the model provides. */
function addEyes(sourceEyes, model) {
  if (!sourceEyes.length) return;

  model.updateWorldMatrix(true, true);

  /* Use the model's own eyeball - it is already the right almond shape and
   * sits properly inside the socket. The export mirrored both copies onto the
   * same side, so the second one is rebuilt here by reflecting the first
   * across the model's x = 0 symmetry plane. */
  const source = sourceEyes[0];
  const toModel = new THREE.Matrix4().copy(model.matrixWorld).invert();
  const inModel = new THREE.Matrix4().multiplyMatrices(toModel, source.matrixWorld);
  const mirror = new THREE.Matrix4().makeScale(-1, 1, 1);

  for (const flip of [false, true]) {
    const eye = new THREE.Mesh(source.geometry, eyeMaterial);
    eye.matrixAutoUpdate = false;
    eye.matrix.copy(flip ? new THREE.Matrix4().multiplyMatrices(mirror, inModel) : inModel);
    eye.frustumCulled = false;
    eye.renderOrder = 10;
    model.add(eye);

    /* A little light spilling out of the socket. */
    const glow = new THREE.PointLight(0xdfe9ff, 0.25, 0.9, 2);
    glow.position.setFromMatrixPosition(eye.matrix);
    model.add(glow);
  }
}

/* ------------------------------------------------------------------ *
 * Interaction
 * ------------------------------------------------------------------ */

const pointer = { x: 0, y: 0 };
const target = { x: 0, y: 0 };

if (!reducedMotion) {
  window.addEventListener('pointermove', (event) => {
    target.x = (event.clientX / window.innerWidth) * 2 - 1;
    target.y = (event.clientY / window.innerHeight) * 2 - 1;

    /* Same pointer in clip space, for picking the surface under the cursor. */
    ndc.set(target.x, -(event.clientY / window.innerHeight) * 2 + 1);
    ndcDirty = true;
    hasPointer = true;
  }, { passive: true });

  window.addEventListener('pointerleave', () => { hasPointer = false; }, { passive: true });

  window.addEventListener('deviceorientation', (event) => {
    if (event.gamma == null) return;
    target.x = THREE.MathUtils.clamp(event.gamma / 35, -1, 1);
    target.y = THREE.MathUtils.clamp((event.beta - 45) / 45, -1, 1);
  }, { passive: true });
}

let scrollProgress = 0;
const onScroll = () => {
  scrollProgress = Math.min(window.scrollY / Math.max(window.innerHeight, 1), 1);
  updateHeroInView();
};
window.addEventListener('scroll', onScroll, { passive: true });

/* ------------------------------------------------------------------ *
 * Cursor pick - lights the wireframe where the pointer touches the mesh
 * ------------------------------------------------------------------ */

const raycaster = new THREE.Raycaster();
raycaster.firstHitOnly = true;
const ndc = new THREE.Vector2(2, 2);
const hitPoint = new THREE.Vector3();
let ndcDirty = false;
let hasPointer = false;
let hovering = false;

function updateHover(delta) {
  /* Only re-cast when the pointer actually moved; the head itself drifts, but
   * a frame of lag on 40k triangles is not worth the cost every frame. */
  if (bodyMesh && hasPointer && ndcDirty) {
    ndcDirty = false;
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObject(bodyMesh, false)[0];
    hovering = Boolean(hit);
    if (hit) hitPoint.copy(hit.point);
  } else if (!hasPointer) {
    hovering = false;
  }

  for (const material of wireMaterials) {
    const uniforms = material.uniforms;
    uniforms.uHit.value.lerp(hitPoint, Math.min(delta * 14, 1));
    uniforms.uStrength.value += ((hovering ? 1 : 0) - uniforms.uStrength.value) * Math.min(delta * 6, 1);
  }
}

/* ------------------------------------------------------------------ *
 * Post-processing
 * ------------------------------------------------------------------ */

/* EffectComposer renders into its own target, which bypasses the canvas
 * antialiasing entirely - hence a multisampled target, or every line in the
 * drawing comes out stepped. */
const composerTarget = new THREE.WebGLRenderTarget(
  window.innerWidth * renderer.getPixelRatio(),
  window.innerHeight * renderer.getPixelRatio(),
  { type: THREE.HalfFloatType, samples: isMobile ? 2 : 4 }
);

const composer = new EffectComposer(renderer, composerTarget);
composer.addPass(new RenderPass(scene, camera));

const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.55, // strength
  0.5,  // radius
  1.35  // threshold — keeps everything but the eyes and hot highlights out
);
composer.addPass(bloom);
composer.addPass(new OutputPass());

/* Cinematic finish: a little lens colour split towards the corners, animated
 * film grain, and a vignette. Runs after OutputPass so it works on the graded
 * image rather than raw linear light. */
const FilmGrainShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uGrain: { value: 0.028 },
    uAberration: { value: 0.0022 },
    uVignette: { value: 1.05 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform float uGrain;
    uniform float uAberration;
    uniform float uVignette;
    varying vec2 vUv;

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
    }

    void main() {
      vec2 offset = vUv - 0.5;
      float radius = length(offset);
      float split = uAberration * (0.25 + radius * radius * 3.5);

      vec4 color;
      color.r = texture2D(tDiffuse, vUv + offset * split).r;
      color.g = texture2D(tDiffuse, vUv).g;
      color.b = texture2D(tDiffuse, vUv - offset * split).b;
      color.a = 1.0;

      float grain = hash(vUv * 1024.0 + fract(uTime) * 91.7) - 0.5;
      color.rgb += grain * uGrain * (0.4 + 0.75 * (1.0 - color.r));

      float vignette = smoothstep(uVignette, 0.28, radius);
      color.rgb *= mix(1.0, vignette, 0.8);

      gl_FragColor = color;
    }
  `,
};

const filmPass = new ShaderPass(FilmGrainShader);
composer.addPass(filmPass);
/* The composer takes its size in CSS pixels; setting both explicitly keeps it
 * in step with the renderer instead of inheriting the target's device size. */
composer.setPixelRatio(renderer.getPixelRatio());
composer.setSize(window.innerWidth, window.innerHeight);

/* ------------------------------------------------------------------ *
 * Loop
 * ------------------------------------------------------------------ */

const clock = new THREE.Clock();
let visible = true;
let heroInView = true;

document.addEventListener('visibilitychange', () => {
  visible = !document.hidden;
  if (visible) clock.getDelta();
});

/* Once the hero has scrolled away, the opaque sections cover the stage
 * completely, so rendering it (and animating the grain over it) is wasted
 * GPU time that the scrolling page needs. Checked from the scroll handler. */
const heroEl = document.querySelector('.hero');

function updateHeroInView() {
  const inView = !heroEl || heroEl.getBoundingClientRect().bottom > 0;
  if (inView === heroInView) return;
  heroInView = inView;
  document.body.classList.toggle('stage-paused', !heroInView);
  if (heroInView) {
    clock.getDelta();
    resetFrameStats();
  }
}

/* ------------------------------------------------------------------ *
 * Adaptive quality
 * ------------------------------------------------------------------ */

/* Each level is cheaper than the last. Measured on an integrated Radeon at
 * 1440x900: 4x MSAA on the half-float target costs ~8 ms, bloom and the film
 * pass ~4-9 ms each, and cost scales with pixel count. The film pass is
 * dropped only near the bottom, since it carries the vignette. */
const QUALITY = [
  { ratio: Math.min(window.devicePixelRatio, 1.5), samples: isMobile ? 2 : 4, film: true },
  { ratio: Math.min(window.devicePixelRatio, 1.25), samples: 2, film: true },
  { ratio: 1, samples: 2, film: false },
  { ratio: 0.8, samples: 0, film: false },
];
let qualityLevel = 0;

/* Step down when frames average slower than this. */
const SLOW_FRAME_MS = 1000 / 45;
const WARMUP_FRAMES = 60;
const SAMPLE_FRAMES = 60;
let frameCount = 0;
let frameTotal = 0;
let lastFrameAt = 0;

function resetFrameStats() {
  frameCount = -WARMUP_FRAMES;
  frameTotal = 0;
  lastFrameAt = 0;
}
resetFrameStats();

function applyQuality(level) {
  const q = QUALITY[level];
  renderer.setPixelRatio(q.ratio);
  composer.setPixelRatio(q.ratio);
  for (const target of [composer.renderTarget1, composer.renderTarget2]) {
    target.samples = q.samples;
    target.dispose();
  }
  composer.setSize(window.innerWidth, window.innerHeight);
  filmPass.enabled = q.film;
}

function adaptQuality() {
  const now = performance.now();
  const frameMs = lastFrameAt ? now - lastFrameAt : 0;
  lastFrameAt = now;

  /* A long gap is a stall (tab switch, throttled background), not a slow
   * GPU - start the window again rather than count it. */
  if (!frameMs || frameMs > 250) return;
  if (++frameCount <= 0) return;

  frameTotal += frameMs;
  if (frameCount < SAMPLE_FRAMES) return;

  const average = frameTotal / frameCount;
  if (average > SLOW_FRAME_MS && qualityLevel < QUALITY.length - 1) {
    applyQuality(++qualityLevel);
    resetFrameStats();
  } else {
    frameCount = 0;
    frameTotal = 0;
  }
}

function animate() {
  requestAnimationFrame(animate);
  if (!visible || !heroInView) return;

  /* Only judge the GPU on the real scene, not the empty one before load. */
  if (bodyMesh) adaptQuality();

  const delta = Math.min(clock.getDelta(), 0.05);
  const time = clock.elapsedTime;

  pointer.x += (target.x - pointer.x) * Math.min(delta * 3.2, 1);
  pointer.y += (target.y - pointer.y) * Math.min(delta * 3.2, 1);

  if (!reducedMotion) {
    pivot.rotation.y = pointer.x * 0.42 + Math.sin(time * 0.22) * 0.05;
    pivot.rotation.x = pointer.y * 0.2 + Math.sin(time * 0.31) * 0.02;
    head.position.y = Math.sin(time * 0.6) * 0.022;
  }

  /* Drift the head back and down a touch as the page scrolls away. */
  pivot.position.y = scrollProgress * 0.55;
  pivot.position.z = -scrollProgress * 1.1;

  updateHover(delta);

  filmPass.uniforms.uTime.value = time;
  composer.render();
}

onScroll();
animate();

/* ------------------------------------------------------------------ *
 * Resize
 * ------------------------------------------------------------------ */

/* The portrait fit measures the headline, which moves once web fonts land. */
document.fonts?.ready.then(fitCamera);

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    camera.aspect = width / height;
    fitCamera();
    renderer.setSize(width, height);
    composer.setSize(width, height);
    scene.background = makeBackdrop();
  }, 120);
});

/* ------------------------------------------------------------------ *
 * Backdrop texture
 * ------------------------------------------------------------------ */

function makeBackdrop() {
  const size = 1024;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');

  const gradient = ctx.createRadialGradient(
    size * 0.5, size * 0.42, size * 0.05,
    size * 0.5, size * 0.5, size * 0.78
  );
  gradient.addColorStop(0, '#101a2b');
  gradient.addColorStop(0.45, '#080d18');
  gradient.addColorStop(1, '#03040a');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/* ------------------------------------------------------------------ *
 * Section reveals
 * ------------------------------------------------------------------ */

const revealTargets = document.querySelectorAll('.card, .results-wrap, .section-title, .cta h2, .cta p');
revealTargets.forEach((el, i) => {
  el.classList.add('reveal');
  el.style.transitionDelay = (i % 4) * 70 + 'ms';
});

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('in');
      observer.unobserve(entry.target);
    }
  });
}, { rootMargin: '-10% 0px -10% 0px' });

revealTargets.forEach((el) => observer.observe(el));
