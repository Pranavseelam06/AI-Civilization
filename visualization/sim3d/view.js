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
  const catalog = await api('/api/checkpoints');
  function filterExperiments() {
    const previous = $('checkpoint').value;
    const query = $('experiment-search').value.trim().toLowerCase();
    const matches = catalog.checkpoints.filter(entry => /^\d+$/.test(query)
      ? entry.label.split(' ')[1] === query
      : entry.label.toLowerCase().includes(query));
    $('checkpoint').replaceChildren(...matches.map(entry => {
      const option = document.createElement('option');
      option.value = entry.name; option.textContent = entry.label;
      option.disabled = !entry.available;
      return option;
    }));
    if (matches.some(entry => entry.name === previous)) $('checkpoint').value = previous;
    $('search-result').textContent = matches.length ? '' : 'No matching experiments.';
    if (query && matches.some(entry => entry.available)) $('policy').value = 'ppo';
    updateCheckpointControl();
  }
  $('experiment-search').oninput = filterExperiments;
  $('experiment-search').onkeydown = event => {
    if (event.key === 'Enter' && $('checkpoint').value) {event.preventDefault(); $('reset').click();}
  };
  filterExperiments();
  if (!catalog.checkpoints.some(entry => entry.available)) {
    const option = document.createElement('option');
    option.textContent = 'No checkpoints — train a model first';option.value = '';
    $('checkpoint').appendChild(option);
    document.querySelector('#policy option[value="ppo"]').disabled = true;
  }
  function updateCheckpointControl() {$('checkpoint').disabled = $('policy').value !== 'ppo';}
  $('policy').onchange = updateCheckpointControl;
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
  let selectedAgent = null;
  const viewMaterial = new THREE.MeshBasicMaterial({color: '#f4cf60', transparent: true,
    opacity: 0.25, depthWrite: false, side: THREE.DoubleSide});
  const viewGeometry = new THREE.PlaneGeometry(0.94, 0.94);
  viewGeometry.rotateX(-Math.PI/2);
  const viewTiles = new THREE.InstancedMesh(viewGeometry, viewMaterial, 25);
  viewTiles.count = 0; viewTiles.frustumCulled = false;
  scene.add(viewTiles);
  const selectionRing = new THREE.Mesh(new THREE.RingGeometry(0.32, 0.42, 32),
    new THREE.MeshBasicMaterial({color: '#ffe28a', side: THREE.DoubleSide, depthWrite: false}));
  selectionRing.rotation.x = -Math.PI/2;
  selectionRing.visible = false;
  scene.add(selectionRing);

  function updateInspection() {
    const agent = state.agents.find(a => a.id === selectedAgent);
    $('inspect-agent').replaceChildren(...[
      {value: '', label: 'None'},
      ...state.agents.map(a => ({value: String(a.id), label: `Agent ${a.id} · ${a.alive ? a.energy + ' energy' : 'dead'}`}))
    ].map(entry => {
      const option = document.createElement('option');
      option.value = entry.value; option.textContent = entry.label;
      return option;
    }));
    $('inspect-agent').value = agent ? String(agent.id) : '';
    $('inspection').replaceChildren();
    viewTiles.count = 0;
    selectionRing.visible = Boolean(agent && agent.alive);
    const line = text => {
      const paragraph = document.createElement('p'); paragraph.className = 'note';
      paragraph.textContent = text; $('inspection').appendChild(paragraph);
    };
    if (!agent) {line('Select an agent to inspect it.'); return;}
    line(`Agent ${agent.id} · ${agent.energy}/100 energy · ${agent.alive ? 'alive' : 'dead'}`);
    line(`Latest action: ${agent.last_action ?? 'none yet'}${agent.last_outcome ? ' · ' + agent.last_outcome : ''}`);
    if (agent.alive) {
      const matrix = new THREE.Matrix4();
      let count = 0;
      for (let dy=-2; dy<=2; dy++) for (let dx=-2; dx<=2; dx++) {
        const x = agent.x+dx, y = agent.y+dy;
        if (x<0 || x>=width || y<0 || y>=height) continue;
        matrix.makeTranslation(x-offsetX, groundY+0.018, y-offsetZ);
        viewTiles.setMatrixAt(count++, matrix);
      }
      viewTiles.count = count; viewTiles.instanceMatrix.needsUpdate = true;
    }
    const probabilities = Object.entries(agent.next_action_probabilities ?? {});
    if (!probabilities.length) {
      line(!agent.alive ? 'This agent has died; no next action.' : state.finished
        ? 'Episode finished; no next action.' : 'Model probabilities are available with Trained PPO.');
      return;
    }
    line('Next-action probabilities at the current tick (latest action was chosen on the previous tick).');
    for (const [name, probability] of probabilities) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:6px;margin:5px 0;font-size:12px';
      const label = document.createElement('span'); label.textContent = name; label.style.width = '52px';
      const bar = document.createElement('progress'); bar.max = 1; bar.value = probability;
      bar.setAttribute('aria-label', `${name} probability`); bar.style.cssText = 'flex:1;min-width:0';
      const value = document.createElement('span'); value.textContent = `${(100*probability).toFixed(1)}%`;
      row.append(label, bar, value); $('inspection').appendChild(row);
    }
    line(state.eating_mode === 'automatic' ? 'Eating is automatic in this experiment; it is not a model action.'
      : 'These are action chances, not thoughts or an explanation of intent.');
  }
  $('inspect-agent').onchange = () => {
    selectedAgent = $('inspect-agent').value === '' ? null : Number($('inspect-agent').value);
    updateInspection();
  };
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerDown = null;
  renderer.domElement.addEventListener('pointerdown', event => {
    pointerDown = event.button === 0 ? {x:event.clientX, y:event.clientY, id:event.pointerId} : null;
  });
  renderer.domElement.addEventListener('pointercancel', () => {pointerDown=null;});
  renderer.domElement.addEventListener('pointerup', event => {
    const down = pointerDown; pointerDown = null;
    if (!down || down.id !== event.pointerId || Math.hypot(event.clientX-down.x, event.clientY-down.y)>5) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX-rect.left)/rect.width*2-1, -(event.clientY-rect.top)/rect.height*2+1);
    raycaster.setFromCamera(pointer, camera);
    const visibleActors = [...actors.values()].filter(model => model.visible);
    const hit = raycaster.intersectObjects(visibleActors, true)[0];
    selectedAgent = hit ? hit.object.userData.agentId : null;
    updateInspection();
  });


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
        model.traverse(node => {node.userData.agentId = a.id;});
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
    $('metrics').textContent = `Tick ${state.tick} / ${horizon} · Alive ${alive}/${state.agents.length} · Food ${state.food.length} · Eaten ${state.food_consumed} · Wasted energy ${state.food_energy_wasted}`;
    $('agents').replaceChildren(...state.agents.map(a => {
      const row = document.createElement('div');
      row.textContent = `Agent ${a.id} · ${names[a.id%4]} · ${a.alive ? `${a.energy} energy` : 'dead'}`;
      return row;
    }));
    if (state.finished) playing = false;
    updateInspection();
    updateButtons();
  }
  async function mutate(path, data, snap=false) {
    if (busy) return;
    busy = true;
    updateButtons();
    try {
      apply(await api(path,data),snap);
      $('status').className = '';
      $('status').textContent = state.finished ? 'Episode finished. Reset to start another.' : `Live ${state.policy}${state.checkpoint ? ' · ' + state.checkpoint + ' · ' + state.eating_mode + (state.action_masking ? ' · masked' : ' · legacy unmasked') : ''}`;
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
    mutate('/api/reset',{seed,policy:$('policy').value,checkpoint:$('checkpoint').value},true);
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
  if (state.checkpoint) $('checkpoint').value=state.checkpoint;
  updateCheckpointControl();
  $('status').textContent=`Ready: ${state.policy}${state.checkpoint ? ' · ' + state.checkpoint + ' · ' + state.eating_mode + (state.action_masking ? ' · masked' : ' · legacy unmasked') : ''}. Press Start.`;
  let previousTime=performance.now();
  renderer.setAnimationLoop(now=>{
    const dt=Math.min((now-previousTime)/1000,0.1);previousTime=now;
    if (playing && !busy && now-lastTick>=1000/Number($('speed').value)) {
      lastTick=now;mutate('/api/step',{});
    }
    for (const [id,model] of actors) model.position.lerp(targets.get(id),1-Math.exp(-18*dt));
    if (selectionRing.visible && actors.has(selectedAgent)) {
      selectionRing.position.copy(actors.get(selectedAgent).position);
      selectionRing.position.y += 0.025;
    }
    controls.update();renderer.render(scene,camera);
  });
}
