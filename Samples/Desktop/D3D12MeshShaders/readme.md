---
page_type: sample
languages:
- cpp
products:
- windows-api-win32
name: Direct3D 12 mesh shader samples
urlFragment: d3d12-mesh-shader-samples-win32
description: This collection of projects act as an introduction to meshlets, and rendering with DirectX mesh shaders.
extendedZipContent:
- path: LICENSE
  target: LICENSE
---

# Modifications made from Microsoft Source:
This fork shows some changes that could be make to the meshlet baking process to add some efficiency. Overall impact of this change is somewhat minor, but it does increate meshlet cone culling efficiency in several use cases.

The included visualizations are purely as a 'Best Case' for this change. Real world improvement is largely depending on the size of meshlets produced (meshlets with a max amount of indices close to 128 behave best, max amount of 32 behaves predicatably poorer). 

Largely the benefits of this change come from the fact that this doesn't affect the actual runtime implementation of the DX12 Sample, just the baking (normal cone calculation) timing. Also this implementation is meant to be somewhat readable, so feel free to improve or use it however you want (so long as it complies with the original DX12-Samples license).

## Visual of Normal Cone Size Reduction
Model Used:

<img src = "https://i.imgur.com/PSOtQoR.png">


Original Code:

<img src = "https://i.imgur.com/ZaygB0E.png">

Alejandro Custom Divet Finder Code:

Note: Normal 'cone' becomes a single line.

<img src = "https://i.imgur.com/yLeBPaw.png">

## Algorithmic Changes
To increase the efficiency of the generated meshlet cones by this sample, rather than accept that all meshlets exist as an individual small 'model', we assume that any meshlet's backfaces are occluding (sort of a loose version of watertightness).

So for example, assume we have a simple meshlet comprised of points A through E that looks like the drawing below.


<img src ="https://i.imgur.com/NaFmkqi.png" alt="Meshlet Example">

Say we have a Viewer with a view direction of V who is viewing the meshlet from behind the normal of A->B and D->E.

<img src = "https://i.imgur.com/KpRwLh7.png">

If our model is guaranteed  to be self-occluding, we can assume that some other meshlet would've blocked the visibility of the face made of C->D. An example of this would be the meshlet created by the red lines below.

<img src = "https://i.imgur.com/7lfAaBp.png">

Because of this basic assumption, we can create a simpler meshlet before generating a normal cone, by moving 'divet' points outward to be an average with their neighbors, so in the original example, moving point C to C' like this:

<img src = "https://i.imgur.com/kt1LTFX.png">

Now whenever we generate a normal cone for this meshlet we obtain a single line, rather than a cone that encompasses all the angles that B->C or C->D which would be occluded by the red meshlet anyway.

## Usage Differences
Using the tool is almost completely unchanged (minus the addition of multithreading to speed up exports), with a `SHOULD_MOVE_OUTPUTTED_VERTICES` def in D3D12MeshletGenerator controlling if you want to see the shifted vertices outputted, or the intended vertices (only really useful for debugging).

---

# ORIGINAL MICROSOFT DOCUMENTATION FOR THIS SAMPLE FOLLOWS:

---
# Direct3D 12 mesh shader samples
This collection of projects act as an introduction to meshlets, and rendering with DirectX mesh shaders.

### Requirements
* GPU and driver with support for [DirectX 12 Ultimate](http://aka.ms/DirectX12UltimateDev)

  <img src="../../../Assets/DirectX12Ultimate.png" alt="DirectX 12 Ultimate" height="100">

### Getting Started
* DirectX Mesh Shader spec/documentation is available at [DirectX Mesh Shader Specs](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html) site.

# Projects
## 1. Wavefront Converter Command Line Tool
This project acts as an example of how to incorporate [DirectXMesh](https://github.com/microsoft/DirectXMesh) meshlet generation functionality into a complete mesh conversion application. The application is structured as a basic command line tool for loading and processing obj files. The processed mesh is exported using a simple binary runtime file format targeted at fast loading and rendering with DirectX apps.

## 2. Meshlet Viewer
This project demonstrates the basics of how to render a meshletized model using DirectX 12. This application loads the binary model files exported by the Wavefront Converter command line tool.

![D3D12 Meshlet Render Preview](src/MeshletRender/D3D12MeshletRender.png)

## 3. [Meshlet Instancing](src/MeshletInstancing/readme.md)
In the Mesh Shader Pipeline API there's no concept of instancing such as in the legacy pipeline. This leaves the logic of instancing meshes entirely up to application code. An inefficient implementation can waste precious threads within threadgroups. This sample demonstrates an implementation which aims to optimize instancing of meshletized meshes by packing the final, unfilled meshlets of multiple instances into a single threadgroup.

![D3D12 Meshlet Instancing Preview](src/MeshletInstancing/D3D12MeshletInstancing.png)

## 4. [Meshlet Culling](src/MeshletCull/readme.md)
The generic functionality of amplification shaders make them a useful tool for an innumerable number of tasks. This sample demonstrates the basics of amplification shaders by showcasing how to cull meshlests before ever dispatching a mesh shader threadgroup into the pipeline.

![D3D12 Meshlet Culling Preview](src/MeshletCull/D3D12MeshletCull.png)

## 5. [Instancing Culling & Dynamic LOD Selection](src/DynamicLOD/readme.md)
This sample presents an advanced shader technique using amplification shaders to do per-instance frustum culling and level-of-detail (LOD) selection entirely on the GPU for an arbitrary number of mesh instances.

![D3D12 Dynamic LOD Preview](src/DynamicLOD/D3D12DynamicLOD.png)

## Further resources
* [DirectX Mesh Shader Spec](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html)
* [DirectXMesh Repository](https://github.com/microsoft/DirectXMesh)
* [NVIDIA Mesh Shader Blog](https://devblogs.nvidia.com/introduction-turing-mesh-shaders/)

## Feedback and Questions
We welcome all feedback, questions and discussions about the mesh shader pipeline on our [discord server](http://discord.gg/directx).
