import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
const loader = new GLTFLoader();
export async function loadModel(name, targetSize, byHeight) {
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
