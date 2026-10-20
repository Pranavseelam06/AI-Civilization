import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export async function start() {
  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#283d40');
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxDistance = 20;
  controls.minDistance = 1;
  function resetCamera() {
    camera.position.set(5, 4, 6);
    controls.target.set(0, 0.6, 0);
    controls.update();
  }
  resetCamera();
  document.querySelector('#reset').onclick = resetCamera;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6d7453, 2));
  const sun = new THREE.DirectionalLight(0xfff2db, 3);
  sun.position.set(3, 7, 5);
  scene.add(sun);
  const loader = new GLTFLoader();
  const status = document.querySelector('#status');

  // These supplied assets are Z-up; Three.js uses Y-up.
  async function loadModel(name, targetSize, byHeight) {
    const gltf = await loader.loadAsync(`/models/${name}.glb`);
    const model = new THREE.Group();
    gltf.scene.rotation.x = -Math.PI / 2;
    model.add(gltf.scene);
    gltf.scene.traverse(node => {
      if (!node.isMesh) return;
      if (!node.geometry.attributes.normal) node.geometry.computeVertexNormals();
      // Some supplied meshes have vertex colors but no glTF material. Use a
      // matte surface so their colors remain visible without an environment map.
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        if (material.isMeshStandardMaterial) {
          material.metalness = 0;
          material.roughness = 1;
          material.needsUpdate = true;
        }
      }
    });
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const dimension = byHeight ? size.y : Math.max(size.x, size.z);
    if (!Number.isFinite(dimension) || dimension <= 0) throw new Error(`${name} has invalid bounds`);
    model.scale.multiplyScalar(targetSize/dimension);
    model.updateMatrixWorld(true);
    const fitted = new THREE.Box3().setFromObject(model);
    const center = fitted.getCenter(new THREE.Vector3());
    model.position.x -= center.x;
    model.position.z -= center.z;
    model.position.y -= fitted.min.y;
    model.updateMatrixWorld(true);
    return model;
  }
  const tile = await loadModel('grass_tile', 4, false);
  scene.add(tile);
  const groundHeight = new THREE.Box3().setFromObject(tile).max.y;
  const bush = await loadModel('berry_bush', 0.8, true);
  bush.position.add(new THREE.Vector3(0.9, groundHeight, 0.25));
  scene.add(bush);
  let person;
  let request = 0;
  function disposeModel(model) {
    model.traverse(node => {
      node.geometry?.dispose();
      const materials = node.material ? (Array.isArray(node.material) ? node.material : [node.material]) : [];
      for (const material of materials) {
        for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
        material.dispose();
      }
    });
  }
  async function setCharacter() {
    const current = ++request;
    const name = document.querySelector('#character').value;
    status.className = '';
    status.textContent = `Loading ${name}…`;
    try {
      const next = await loadModel(name, name === 'boy' || name === 'girl' ? 1.1 : 1.6, true);
      if (current !== request) {disposeModel(next);return;}
      next.position.add(new THREE.Vector3(-0.65, groundHeight, 0));
      if (person) {scene.remove(person);disposeModel(person);}
      person = next;
      scene.add(person);
      status.textContent = `Loaded ${name}, grass tile and berry bush.`;
    } catch(error) {
      if (current !== request) return;
      status.className = 'error';
      status.textContent = `Could not load ${name}: ${error.message}`;
    }
  }
  document.querySelector('#character').onchange = setCharacter;
  await setCharacter();
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
  renderer.setAnimationLoop(() => {controls.update();renderer.render(scene,camera);});
}
