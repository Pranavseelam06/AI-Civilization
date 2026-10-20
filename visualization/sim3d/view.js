import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadModel } from './assets.js';

export async function start() {
  const $ = id => document.getElementById(id);
  async function api(path, data) {
    const response = await fetch(path, data === undefined ? {cache: 'no-store'} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `Server returned ${response.status}`);
    return body;
  }
  let state = await api('/api/state');
  const {width, height} = state.config;
  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#283d40');
  const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 250);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxDistance = 100;
  controls.minDistance = 2;
  controls.maxPolarAngle = Math.PI * 0.48;
  function resetCamera() {
    camera.position.set(26, 29, 34);
    controls.target.set(0, 0, 0);
    controls.update();
  }
  resetCamera();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6d7453, 2));
  const light = new THREE.DirectionalLight(0xfff2db, 3);
  light.position.set(10, 30, 15);
  scene.add(light);

  const names = ['man', 'woman', 'boy', 'girl'];
  const [tile, bush, ...people] = await Promise.all([
    loadModel('grass_tile', 1, false), loadModel('berry_bush', 0.42, true),
    ...names.map(name => loadModel(name, name === 'boy' || name === 'girl' ? 0.55 : 0.78, true))
  ]);
  const groundY = new THREE.Box3().setFromObject(tile).max.y;
  // Mesh instancing keeps 1,024 grass tiles and hundreds of bushes affordable.
  function instances(model, capacity) {
    model.updateMatrixWorld(true);
    const parts = [];
    model.traverse(node => {
      if (!node.isMesh) return;
      const mesh = new THREE.InstancedMesh(node.geometry, node.material, capacity);
      mesh.count = 0;
      mesh.frustumCulled = false;
      scene.add(mesh);
      parts.push({mesh, base: node.matrixWorld.clone()});
    });
    return positions => {
      const translation = new THREE.Matrix4();
      const matrix = new THREE.Matrix4();
      for (const {mesh, base} of parts) {
        positions.forEach(([x,y,z], i) => {
          translation.makeTranslation(x,y,z);
          matrix.multiplyMatrices(translation, base);
          mesh.setMatrixAt(i,matrix);
        });
        mesh.count = positions.length;
        mesh.instanceMatrix.needsUpdate = true;
      }
    };
  }
  const offsetX = (width-1)/2, offsetZ = (height-1)/2;
  const land = [];
  for (let y=0; y<height; y++) for (let x=0; x<width; x++) land.push([x-offsetX,0,y-offsetZ]);
  instances(tile,width*height)(land);
  const updateFood = instances(bush,width*height);
  const actors = new Map();
  const targets = new Map();
  let playing = false, busy = false, lastTick = 0;

  function updateButtons() {
    $('play').disabled = state.finished;
    $('play').textContent = playing ? 'Pause' : (state.finished ? 'Finished' : 'Start');
    $('step').disabled = busy || playing || state.finished;
    $('reset').disabled = busy;
  }
  function apply(next, snap=false) {
    state = next;
    updateFood(state.food.map(([y,x]) => [x-offsetX,groundY,y-offsetZ]));
    for (const a of state.agents) {
      if (!actors.has(a.id)) {
        const model = people[a.id % people.length].clone(true);
        scene.add(model);
        actors.set(a.id, model);
      }
      const model = actors.get(a.id);
      model.visible = a.alive;
      const target = new THREE.Vector3(a.x-offsetX,groundY+a.z,a.y-offsetZ);
      const previous = targets.get(a.id);
      if (previous && !previous.equals(target)) {
        model.rotation.y = Math.atan2(target.x-previous.x,target.z-previous.z);
      }
      targets.set(a.id,target);
      if (snap) {model.position.copy(target);model.rotation.y=0;}
    }
    const alive = state.agents.filter(a=>a.alive).length;
    const horizon = state.config.horizon ?? '∞';
    $('metrics').textContent = `Tick ${state.tick} / ${horizon} · Alive ${alive}/${state.agents.length} · Food ${state.food.length} · Eaten ${state.food_consumed}`;
    $('agents').replaceChildren(...state.agents.map(a => {
      const row = document.createElement('div');
      row.textContent = `Agent ${a.id} · ${names[a.id%4]} · ${a.alive ? `${a.energy} energy` : 'dead'}`;
      return row;
    }));
    if (state.finished) playing = false;
    updateButtons();
  }
  async function mutate(path, data, snap=false) {
    if (busy) return;
    busy = true;
    updateButtons();
    try {
      apply(await api(path,data),snap);
      $('status').className = '';
      $('status').textContent = state.finished ? 'Episode finished. Reset to start another.' : 'Live Python simulation · 3D display';
    } catch (error) {
      playing = false;
      $('status').className = 'error';
      $('status').textContent = error.message;
    } finally {busy=false;updateButtons();}
  }
  $('play').onclick = () => {if(!state.finished){playing=!playing;updateButtons();}};
  $('step').onclick = () => mutate('/api/step',{});
  $('reset').onclick = () => {
    if (busy) return;
    const seed = Number($('seed').value);
    if (!$('seed').value.trim() || !Number.isInteger(seed) || seed<0 || seed>4294967295) {
      $('status').textContent='Enter a whole-number seed from 0 to 4294967295.';
      return;
    }
    playing=false;
    mutate('/api/reset',{seed,policy:$('policy').value},true);
  };
  $('speed').oninput = () => {$('rate').textContent=$('speed').value;};
  $('camera').disabled=false;
  $('camera').onclick=resetCamera;
  window.addEventListener('keydown',event => {
    if (event.code==='Space' && !['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName)) {
      event.preventDefault();$('play').click();
    }
  });
  window.addEventListener('resize',()=>{
    camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);
  });
  apply(state,true);
  $('seed').value=state.seed;
  $('policy').value=state.policy;
  $('status').textContent='Ready. Press Start to run the simulation.';
  let previousTime=performance.now();
  renderer.setAnimationLoop(now=>{
    const dt=Math.min((now-previousTime)/1000,0.1);previousTime=now;
    if (playing && !busy && now-lastTick>=1000/Number($('speed').value)) {
      lastTick=now;mutate('/api/step',{});
    }
    for (const [id,model] of actors) model.position.lerp(targets.get(id),1-Math.exp(-18*dt));
    controls.update();renderer.render(scene,camera);
  });
}
