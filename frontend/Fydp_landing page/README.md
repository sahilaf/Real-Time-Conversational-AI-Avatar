# FYDP_V2 — landing page

Landing page for **Real-Time Bangla Speech Driven Avatar Generation** — a
real-time Bangla conversational avatar (speech in, lip-synced face out, 200–300
ms end to end) by Team Deep Thinkers, UIU CSE.

Built around `head.glb`: the mesh rendered as thin-film glass with a wireframe
overlay, sitting inside a draughting frame, finished with lens colour split,
film grain and a vignette.

## Run it

The model is fetched over HTTP, so it needs a server — opening `index.html`
from the filesystem will fail on CORS.

```bash
python -m http.server 5173
```

Then open http://localhost:5173.

## Files

| File | What it holds |
| --- | --- |
| `index.html` | Markup — hero, pipeline, research, status, team, report — plus the draughting overlay and the import map pointing `three` at `vendor/` |
| `styles.css` | All page styling; palette and mono stack live in the `:root` block |
| `head-scene.js` | Renderer, studio environment, materials, wireframe, film pass, interaction |
| `vendor/` | three.js r180 build plus the addons used (loader, bloom, shader passes), and three-mesh-bvh 0.9 for the cursor raycast |
| `head.glb` | The model — 47,280 triangles, one material, no textures |

`node_modules/` only exists because three.js and three-mesh-bvh were installed to copy `vendor/`
from — nothing at runtime reads it, and the site works offline as-is.

## Content

The page copy comes from the FYDP-1 main report (`UIU_E1_253_FYDP - 1 - Main
Report - (Team Deep Thinkers).pdf`): *Real-Time Bangla Speech Driven Avatar
Generation*, Group E1-253, United International University, submitted
28 February 2026.

Sections, and where each came from:

| Section | Source in the report |
| --- | --- |
| Hero | Abstract and §1.4 Methodology |
| Stats | §3.2 latency (200–300 ms), SyncTalk 2D frame rate, §3.4 team size |
| Pipeline | §3.2 — STT, LLM, TTS and lip-sync module descriptions |
| Research | §3.2 — the three proposed modifications to SyncTalk 2D |
| Status | §3.2 conclusion — English baseline validated, Bangla stages ahead |
| Team | Cover page authors and supervisors; §3.2 hardware environment |

**Still to fill in:** every link is `href="#"`. The report, slides, repository
and demo video have no URLs yet — point the "Open the report" button and the
nav "Report" link at them once they are hosted. The numbers in the drawing's
margin notes are real, so keep them in step with the system if it changes.

## Tuning the look

- **`STUDIO_LEVEL`** — one number at the top of the lighting block scaling every
  panel and lamp at once. It is the brightness dial: drop it to darken the whole
  plate, raise it if the specimen sinks into the background. Currently 0.45.
- **`makeStudioEnvironment()`** — the biggest lever on character. A dark room with six wide,
  dim panels (cool key, cyan wash, magenta wash, underlight, frontal fill, rear
  wash), PMREM'd into the scene environment with a heavy blur. They are large
  and low-intensity on purpose: small bright panels mirror off the shell as
  legible rectangles. Recolour them to re-light the whole piece.
- **Lights** — deliberately weak, since the environment does the lighting.
  `key` / `rim` / `fill` only shape the form, and `warmBounce` adds the magenta
  tint. It sits well back from the head on purpose — close in, inverse-square
  falloff concentrates it into a hotspot under the cheekbone that reads as a
  stray lamp; far away with the intensity scaled up, the same light is a wash. A cyan point light used to sit opposite it; from behind
  the right temple it burned a hard blob into the skull, so it was removed.
  Point lights on a smooth glossy surface read as blobs rather than washes —
  the cyan in the picture comes from the environment panels instead.
- **`dressBody()`** — the shell. The GLB's own dark blue-slate base is kept and
  dressed as thin-film glass: `transmission` 0.72 with strong `iridescence`, so
  on black the surface nearly vanishes except where the film splits into cyan
  and magenta. Roughness is held at 0.24 rather than the 0.05 the GLB asks for:
  the sculpt carries small flat plates across the skull and neck, and at mirror
  roughness those reflect the studio panels back as hot rectangles. Lowering it
  brings the boxes straight back.
- **`addWireframe()`** — two line layers, both drawn from the body geometry so
  they always match the sculpt:
  - **feature edges** (`edgeMaterial`, 34%) from `EdgesGeometry` at
    `WIRE_ANGLE` degrees — only edges where the surface creases, so panel
    outlines and plate seams. About 3,000 segments. Lower the angle for more
    lines; below roughly 8° it collapses back into triangle soup.
  - **mesh grid** (`meshMaterial`, 18%) from `WireframeGeometry` — every
    triangle edge. This is the layer that shimmers as the head drifts. It only
    holds together because the composer target is multisampled; without that,
    60k segments is far more line than the screen has pixels to resolve and it
    breaks up into noise. Drop it toward 0.05 for a quieter plate.

  Both come from `makeWireMaterial()`, a small `ShaderMaterial` that also
  carries the cursor glow: `uHit` is the point under the pointer, `uRadius` how
  far the glow spreads, `uGlow` its colour (kept above 1 so it crosses the
  bloom threshold), `uBase` the resting opacity. `updateHover()` drives every
  material in `wireMaterials`.
- **Antialiasing** — the composer renders into its own target, which bypasses
  the canvas `antialias` flag completely, so `composerTarget` is created
  multisampled (4× desktop, 2× mobile). Note the composer takes its size in CSS
  pixels while the target is in device pixels; both `setPixelRatio` and
  `setSize` are called explicitly at startup to keep them in step.
- **`FilmGrainShader`** — the cinematic finish, running after `OutputPass` so it
  works on the graded image. `uGrain` is grain strength (0.028), `uAberration`
  the lens colour split towards the corners, `uVignette` the falloff. A second,
  coarser grain layer sits in CSS as `.stage-grain` (opacity 0.07) on a slower
  step, so the noise does not look uniform — turn both down together.
- **`bloom`** — threshold sits at 1.35 so only the eyes bleed, not every
  specular highlight.
- **Eyes** — the export mirrors both eyeballs onto one socket, so `addEyes()`
  takes the model's own eye mesh and reflects a second copy across the x = 0
  symmetry plane. Colour values above 1 are what push them past the bloom
  threshold.
- **`FRAME_OFFSET`** — where the bust sits vertically, as a fraction of its own height. Positive lifts it above centre, negative drops it; currently 0.01.
- **`frameModel()` / `fitCamera()`** — size and crop, including the lifted
  framing used on portrait screens. `BASE_YAW` is the resting yaw — 0 faces the camera squarely, raise it for a three-quarter presentation.

While the page is open, `window.vessel` exposes the scene, camera, materials
and bloom pass, so values can be tried live from the console before editing.

## Behaviour

- Pointer (or device tilt) parallax with a slow idle drift; the specimen also
  recedes slightly as the page scrolls.
- Moving the cursor over the specimen lights the wireframe around the point
  under it, fading in and out as the pointer arrives and leaves. The pick is a
  raycast against the body mesh, re-cast only when the pointer moves.
- Loading progress bar driven by the GLB download, then a cross-fade in.
- Rendering pauses while the tab is hidden.
- `prefers-reduced-motion` disables parallax, drift and the grain animation.
- On narrow screens the margin notes collapse to the figure label so they do
  not collide with the hero copy.
