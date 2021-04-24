//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************
#include "D3D12MeshletGenerator.h"

#include "Generation.h"
#include "Utilities.h"
#include <iostream>
#include <thread>

using namespace DirectX;

namespace
{
	inline XMVECTOR QuantizeSNorm(XMVECTOR value) 
	{
		return (XMVectorClamp(value, g_XMNegativeOne, g_XMOne) * 0.5f + XMVectorReplicate(0.5f)) * 255.0f;
	}

	inline XMVECTOR QuantizeUNorm(XMVECTOR value) 
	{
		return (XMVectorClamp(value, g_XMZero, g_XMOne)) * 255.0f;
	}
}

namespace internal
{
	template <typename T>
	HRESULT ComputeMeshlets(
		uint32_t maxVerts, uint32_t maxPrims,
		const T* indices, uint32_t indexCount,
		const Subset* indexSubsets, uint32_t subsetCount,
		const XMFLOAT3* positions, uint32_t vertexCount,
		std::vector<Subset>& meshletSubsets,
		std::vector<Meshlet>& meshlets,
		std::vector<uint8_t>& uniqueVertexIndices,
		std::vector<PackedTriangle>& primitiveIndices);

	template<typename T>
	void DivetShift(std::vector<uint32_t>& divetCandidates, DirectX::XMFLOAT3  vertices[256], std::vector<std::vector<uint32_t>>& divetConnections, DirectX::XMFLOAT3  normals[256], const Meshlet& m, const PackedTriangle* primitiveIndices, const DWORD& flags, CullData* cullData, const Meshlet* meshlets, const uint32_t& mi, bool& degen);

	template<typename T>
	void ComputeCullDataForIndex(uint32_t mi, const Meshlet* meshlets, CullData* cullData, const T* uniqueVertexIndices, const uint32_t& vertexCount, DirectX::XMFLOAT3* positions, const PackedTriangle* primitiveIndices, DWORD& flags);

	template <typename T>
	HRESULT ComputeCullData(
		XMFLOAT3* positions, uint32_t vertexCount,
		const Meshlet* meshlets, uint32_t meshletCount,
		const T* uniqueVertexIndices,
		const PackedTriangle* primitiveIndices,
		DWORD flags,
		CullData* cullData
	);
}

HRESULT ComputeMeshlets(
	uint32_t maxVerts, uint32_t maxPrims,
	const uint16_t* indices, uint32_t indexCount,
	const Subset* indexSubsets, uint32_t subsetCount,
	const XMFLOAT3* positions, uint32_t vertexCount,
	std::vector<Subset>& meshletSubsets,
	std::vector<Meshlet>& meshlets,
	std::vector<uint8_t>& uniqueVertexIndices,
	std::vector<PackedTriangle>& primitiveIndices) 
{
	return internal::ComputeMeshlets(maxVerts, maxPrims, indices, indexCount, indexSubsets, subsetCount, positions, vertexCount, meshletSubsets, meshlets, uniqueVertexIndices, primitiveIndices);
}

HRESULT ComputeMeshlets(
	uint32_t maxVerts, uint32_t maxPrims,
	const uint32_t* indices, uint32_t indexCount,
	const Subset* indexSubsets, uint32_t subsetCount,
	const XMFLOAT3* positions, uint32_t vertexCount,
	std::vector<Subset>& meshletSubsets,
	std::vector<Meshlet>& meshlets,
	std::vector<uint8_t>& uniqueVertexIndices,
	std::vector<PackedTriangle>& primitiveIndices) 
{
	return internal::ComputeMeshlets(maxVerts, maxPrims, indices, indexCount, indexSubsets, subsetCount, positions, vertexCount, meshletSubsets, meshlets, uniqueVertexIndices, primitiveIndices);
}

HRESULT ComputeMeshlets(
	uint32_t maxVerts, uint32_t maxPrims,
	const uint16_t* indices, uint32_t indexCount,
	const XMFLOAT3* positions, uint32_t vertexCount,
	std::vector<Subset>& meshletSubsets,
	std::vector<Meshlet>& meshlets,
	std::vector<uint8_t>& uniqueVertexIndices,
	std::vector<PackedTriangle>& primitiveIndices) 
{
	Subset s = { 0, indexCount };
	return internal::ComputeMeshlets(maxVerts, maxPrims, indices, indexCount, &s, 1u, positions, vertexCount, meshletSubsets, meshlets, uniqueVertexIndices, primitiveIndices);
}

HRESULT ComputeMeshlets(
	uint32_t maxVerts, uint32_t maxPrims,
	const uint32_t* indices, uint32_t indexCount,
	const XMFLOAT3* positions, uint32_t vertexCount,
	std::vector<Subset>& meshletSubsets,
	std::vector<Meshlet>& meshlets,
	std::vector<uint8_t>& uniqueVertexIndices,
	std::vector<PackedTriangle>& primitiveIndices) 
{
	Subset s = { 0, indexCount };
	return internal::ComputeMeshlets(maxVerts, maxPrims, indices, indexCount, &s, 1u, positions, vertexCount, meshletSubsets, meshlets, uniqueVertexIndices, primitiveIndices);
}

HRESULT ComputeCullData(
	XMFLOAT3* positions, uint32_t vertexCount,
	const Meshlet* meshlets, uint32_t meshletCount,
	const uint16_t* uniqueVertexIndices,
	const PackedTriangle* primitiveIndices,
	DWORD flags,
	CullData* cullData
) 
{
	return internal::ComputeCullData(positions, vertexCount, meshlets, meshletCount, uniqueVertexIndices, primitiveIndices, flags, cullData);
}

HRESULT ComputeCullData(
	XMFLOAT3* positions, uint32_t vertexCount,
	const Meshlet* meshlets, uint32_t meshletCount,
	const uint32_t* uniqueVertexIndices,
	const PackedTriangle* primitiveIndices,
	DWORD flags,
	CullData* cullData
) 
{
	return internal::ComputeCullData(positions, vertexCount, meshlets, meshletCount, uniqueVertexIndices, primitiveIndices, flags, cullData);
}


template <typename T>
HRESULT internal::ComputeMeshlets(
	uint32_t maxVerts, uint32_t maxPrims,
	const T* indices, uint32_t indexCount,
	const Subset* indexSubsets, uint32_t subsetCount,
	const DirectX::XMFLOAT3* positions, uint32_t vertexCount,
	std::vector<Subset>& meshletSubsets,
	std::vector<Meshlet>& meshlets,
	std::vector<uint8_t>& uniqueVertexIndices,
	std::vector<PackedTriangle>& primitiveIndices) 
{
	UNREFERENCED_PARAMETER(indexCount);

	for (uint32_t i = 0; i < subsetCount; ++i) 
	{
		Subset s = indexSubsets[i];

		assert(s.Offset + s.Count <= indexCount);

		std::vector<InlineMeshlet<T>> builtMeshlets;
		Meshletize(maxVerts, maxPrims, indices + s.Offset, s.Count, positions, vertexCount, builtMeshlets);

		Subset meshletSubset;
		meshletSubset.Offset = static_cast<uint32_t>(meshlets.size());
		meshletSubset.Count = static_cast<uint32_t>(builtMeshlets.size());
		meshletSubsets.push_back(meshletSubset);

		// Determine final unique vertex index and primitive index counts & offsets.
		uint32_t startVertCount = static_cast<uint32_t>(uniqueVertexIndices.size()) / sizeof(T);
		uint32_t startPrimCount = static_cast<uint32_t>(primitiveIndices.size());

		uint32_t uniqueVertexIndexCount = startVertCount;
		uint32_t primitiveIndexCount = startPrimCount;

		// Resize the meshlet output array to hold the newly formed meshlets.
		uint32_t meshletCount = static_cast<uint32_t>(meshlets.size());
		meshlets.resize(meshletCount + builtMeshlets.size());

		for (uint32_t j = 0, dest = meshletCount; j < static_cast<uint32_t>(builtMeshlets.size()); ++j, ++dest)
		{
			meshlets[dest].VertOffset = uniqueVertexIndexCount;
			meshlets[dest].VertCount = static_cast<uint32_t>(builtMeshlets[j].UniqueVertexIndices.size());
			uniqueVertexIndexCount += static_cast<uint32_t>(builtMeshlets[j].UniqueVertexIndices.size());

			meshlets[dest].PrimOffset = primitiveIndexCount;
			meshlets[dest].PrimCount = static_cast<uint32_t>(builtMeshlets[j].PrimitiveIndices.size());
			primitiveIndexCount += static_cast<uint32_t>(builtMeshlets[j].PrimitiveIndices.size());
		}

		// Allocate space for the new data.
		uniqueVertexIndices.resize(uniqueVertexIndexCount * sizeof(T));
		primitiveIndices.resize(primitiveIndexCount);

		// Copy data from the freshly built meshlets into the output buffers.
		auto vertDest = reinterpret_cast<T*>(uniqueVertexIndices.data()) + startVertCount;
		auto primDest = reinterpret_cast<uint32_t*>(primitiveIndices.data()) + startPrimCount;

		for (uint32_t j = 0; j < static_cast<uint32_t>(builtMeshlets.size()); ++j)
		{
			std::memcpy(vertDest, builtMeshlets[j].UniqueVertexIndices.data(), builtMeshlets[j].UniqueVertexIndices.size() * sizeof(T));
			std::memcpy(primDest, builtMeshlets[j].PrimitiveIndices.data(), builtMeshlets[j].PrimitiveIndices.size() * sizeof(uint32_t));

			vertDest += builtMeshlets[j].UniqueVertexIndices.size();
			primDest += builtMeshlets[j].PrimitiveIndices.size();
		}
	}

	return S_OK;
}

void generateNormals(XMFLOAT3 vertices[256], XMFLOAT3 normals[256], const Meshlet& m, const PackedTriangle* primitiveIndices, DWORD flags) 
{
	// Generate primitive normals & cache
	for (uint32_t i = 0; i < m.PrimCount; ++i) 
	{
		auto primitive = primitiveIndices[m.PrimOffset + i];

		XMVECTOR triangle[3]
		{
			XMLoadFloat3(&vertices[primitive.indices.i0]),
			XMLoadFloat3(&vertices[primitive.indices.i1]),
			XMLoadFloat3(&vertices[primitive.indices.i2]),
		};

		XMVECTOR p10 = triangle[1] - triangle[0];
		XMVECTOR p20 = triangle[2] - triangle[0];
		XMVECTOR n = XMVector3Normalize(XMVector3Cross(p10, p20));

		XMStoreFloat3(&normals[i], (flags & CNORM_WIND_CW) != 0 ? -n : n);
	}
}


// Pulling out the normalCone generation to make it be able to use recursively (with modified data)
HRESULT ComputeNormalCone(
	const Meshlet* meshlets,
	const PackedTriangle* primitiveIndices,
	CullData* cullData,
	XMFLOAT3 vertices[256],
	XMFLOAT3 normals[256],
	uint32_t meshletIndex,
	bool& degen
) {
	auto& m = meshlets[meshletIndex];
	auto& c = cullData[meshletIndex];
	// Calculate spatial bounds
	XMVECTOR positionBounds = MinimumBoundingSphere(vertices, m.VertCount);
	XMStoreFloat4(&c.BoundingSphere, positionBounds);

	// Calculate the normal cone
	// 1. Normalized center point of minimum bounding sphere of unit normals == conic axis
	XMVECTOR normalBounds = MinimumBoundingSphere(normals, m.PrimCount);

	// 2. Calculate dot product of all normals to conic axis, selecting minimum
	XMVECTOR axis = XMVectorSetW(XMVector3Normalize(normalBounds), 0);

	XMVECTOR minDot = g_XMOne;
	for (uint32_t i = 0; i < m.PrimCount; ++i) 
	{
		XMVECTOR dot = XMVector3Dot(axis, XMLoadFloat3(&normals[i]));
		minDot = XMVectorMin(minDot, dot);
	}

	if (XMVector4Less(minDot, XMVectorReplicate(0.1f))) 
	{
		// Not setting directly since we will replace it later.
		degen = true;
	}

	// Find the point on center-t*axis ray that lies in negative half-space of all triangles
	float maxt = 0;

	for (uint32_t i = 0; i < m.PrimCount; ++i)
	{
		auto primitive = primitiveIndices[m.PrimOffset + i];

		uint32_t indices[3]
		{
			primitive.indices.i0,
			primitive.indices.i1,
			primitive.indices.i2,
		};

		XMVECTOR triangle[3]
		{
			XMLoadFloat3(&vertices[indices[0]]),
			XMLoadFloat3(&vertices[indices[1]]),
			XMLoadFloat3(&vertices[indices[2]]),
		};

		XMVECTOR c = positionBounds - triangle[0];

		XMVECTOR n = XMLoadFloat3(&normals[i]);
		float dc = XMVectorGetX(XMVector3Dot(c, n));
		float dn = XMVectorGetX(XMVector3Dot(axis, n));

		// dn should be larger than mindp cutoff above
		// This assertion actually can fail, since we're saving it later.
		//assert(dn > 0.0f);
		float t = dc / dn;

		maxt = (t > maxt) ? t : maxt;
	}

	// cone apex should be in the negative half-space of all cluster triangles by construction
	c.ApexOffset = maxt;

	// cos(a) for normal cone is minDot; we need to add 90 degrees on both sides and invert the cone
	// which gives us -cos(a+90) = -(-sin(a)) = sin(a) = sqrt(1 - cos^2(a))
	XMVECTOR coneCutoff = XMVectorSqrt(g_XMOne - minDot * minDot);

	// 3. Quantize to uint8
	XMVECTOR quantized = QuantizeSNorm(axis);
	c.NormalCone[0] = (uint8_t)XMVectorGetX(quantized);
	c.NormalCone[1] = (uint8_t)XMVectorGetY(quantized);
	c.NormalCone[2] = (uint8_t)XMVectorGetZ(quantized);

	XMVECTOR error = ((quantized / 255.0f) * 2.0f - g_XMOne) - axis;
	error = XMVectorSum(XMVectorAbs(error));

	quantized = QuantizeUNorm(coneCutoff + error);
	quantized = XMVectorMin(quantized + g_XMOne, XMVectorReplicate(255.0f));
	c.NormalCone[3] = (uint8_t)XMVectorGetX(quantized);
	degen = false;
	return S_OK;
}

//
// Strongly influenced by https://github.com/zeux/meshoptimizer - Thanks amigo!
//

// Process: If we assume that all meshlet models are watertight (or at least can't be viewed from behind)
// We can "flatten" the meshlets and generate the normal cone in a much smaller area
// Sort of assuming that the other faces in a meshlet 'occlude' the 'divet' points
// And we're only concerned with any point that is a 'peak' or 'edge' of a meshlet.
bool DivetShift(std::vector<uint32_t>& divetCandidates, DirectX::XMFLOAT3  vertices[256], std::vector<std::vector<uint32_t>>& divetConnections, DirectX::XMFLOAT3  normals[256], const Meshlet& m, const PackedTriangle* primitiveIndices, const DWORD& flags, CullData* cullData, const Meshlet* meshlets, const uint32_t& mi, bool& degen) {
	// Try to shift the verts.
	uint32_t index = 0;

	for (auto& i : divetCandidates) {
		XMVECTOR neighborSum = XMVectorZero();
		XMFLOAT3 vertexBackup = vertices[i];
		XMVECTOR vertexBackupVec = XMLoadFloat3(&vertices[i]);
		std::vector<uint32_t>& connections = divetConnections[index];
		for (auto& j : connections) {
			neighborSum = XMVectorAdd(neighborSum, XMLoadFloat3(&vertices[j]));
		}
		XMVECTOR neighborAvg = XMVectorDivide(neighborSum, XMVectorReplicate((float)connections.size()));

		if (XMVectorGetX(XMVector3LengthSq(XMVectorSubtract(XMLoadFloat3(&vertices[i]), neighborAvg))) <= 0.00000001f) {
			index++;
			continue;
		}


		XMStoreFloat3(&vertices[i], neighborAvg);

		// Changes normals (not all, but still fine)
		generateNormals(vertices, normals, m, primitiveIndices, flags);

		//if passes test, keep shift and recur
		//if fails, revert
		ComputeNormalCone(meshlets, primitiveIndices, cullData, vertices, normals, mi, degen);
		bool shouldShift = true;
		// Have to check that all b->a and b->c vecs are in the 'cullCopy'
		for (auto& j : connections) {
			XMVECTOR vecOut = XMVector3Normalize(XMVectorSubtract(XMLoadFloat3(&vertices[j]), vertexBackupVec));
			// How it's done with shaders...
			/*if (dot(view, -axis) > normalCone.w) {
			return false;
			}*/
			XMVECTOR axis = XMVectorSet((float)cullData->NormalCone[0], (float)cullData->NormalCone[1], (float)cullData->NormalCone[2], 0.0f);
			axis = XMVectorDivide(axis, XMVectorSet(255.0f, 255.0f, 255.0f, 1.0f));
			axis = XMVectorMultiplyAdd(axis, XMVectorSet(2.0f, 2.0f, 2.0f, 1.0f), XMVectorSet(-1.0f, -1.0f, -1.0f, 0.0f));
			axis = XMVector3Normalize(axis);
			// Test fails...
			if (XMVectorGetX(XMVector3Dot(XMVectorNegate(axis), vecOut)) > cullData->NormalCone[3] / 255.0f) {

				shouldShift = false;
				break;
			}
		}

		if (shouldShift) {
			//std::cout << "Shifting Vert: " << i << std::endl;
			return true;
		}
		else {
			vertices[i] = vertexBackup;
		}

		index++;
	}
	return false;
}

template<typename T>
void internal::ComputeCullDataForIndex(uint32_t mi, const Meshlet* meshlets, CullData* cullData, const T* uniqueVertexIndices, const uint32_t& vertexCount, DirectX::XMFLOAT3* positions, const PackedTriangle* primitiveIndices, DWORD& flags) {
	XMFLOAT3 vertices[256];
	XMFLOAT3 normals[256];
	auto& m = meshlets[mi];
	auto& c = cullData[mi];
	//std::cout << "On Meshlet: " << mi << std::endl;
	// Cache vertices
	for (uint32_t i = 0; i < m.VertCount; ++i) {
		uint32_t vIndex = uniqueVertexIndices[m.VertOffset + i];

		assert(vIndex < vertexCount);
		vertices[i] = positions[vIndex];
	}

	generateNormals(vertices, normals, m, primitiveIndices, flags);

	bool degen = false;

	ComputeNormalCone(meshlets, primitiveIndices, cullData, vertices, normals, mi, degen);

	// ************ Alejandro's Custom Divet Findy Fixer *************
	// Identify all divet candidate points
	std::vector<uint32_t> divetCandidates;
	std::vector<std::vector<uint32_t>> divetConnections;
	for (uint32_t i = 0; i < m.VertCount; i++) {
		// Vertex index i is only a candidate if every point in a triangle with it occurs either
		// 0 times, or 2 times.
		std::vector<uint32_t> occurenceCount(m.VertCount, 0);
		std::vector<uint32_t> connectedVerts;
		for (uint32_t j = 0; j < m.PrimCount; j++) {
			auto& indices = primitiveIndices[m.PrimOffset + j].indices;
			if ((indices.i0 == i) || (indices.i1 == i) || (indices.i2 == i)) {
				occurenceCount[indices.i0]++;
				occurenceCount[indices.i1]++;
				occurenceCount[indices.i2]++;
			}
		}
		bool candidate = true;
		// Check if all occurences are 0 or 2
		for (uint32_t j = 0; j < m.VertCount; j++) {
			// Current point to check is exempt from 0 or 2 rule
			if (j == i) continue;
			if (occurenceCount[j] == 2) {
				connectedVerts.push_back(j);
			}
			// No need to check if more if it violates the condition once.
			if ((occurenceCount[j] != 0) && (occurenceCount[j] != 2)) {
				candidate = false;
				break;
			}
		}

		if (candidate == true) {
			divetCandidates.push_back(i);
			divetConnections.push_back(connectedVerts);
		}
	}
	while (DivetShift(divetCandidates, vertices, divetConnections, normals, m, primitiveIndices, flags, cullData, meshlets, mi, degen)) {}

	// Just for debug purposes...
	// Uncache vertices
#define SHOULD_MOVE_OUTPUTTED_VERTICES false
#if SHOULD_MOVE_OUTPUTTED_VERTICES
	for (uint32_t i = 0; i < m.VertCount; ++i) {
		uint32_t vIndex = uniqueVertexIndices[m.VertOffset + i];

		assert(vIndex < vertexCount);
		positions[vIndex] = vertices[i];
	}
#endif
	// ************ Alejandro's Custom Divet Findy Fixer END *************
	degen = false;
	ComputeNormalCone(meshlets, primitiveIndices, cullData, vertices, normals, mi, degen);
	if (degen) {
		// Degenerate cone
		c.NormalCone[0] = 127;
		c.NormalCone[1] = 127;
		c.NormalCone[2] = 127;
		c.NormalCone[3] = 255;
	}
}

// Just threading this to make it a bit more practical for huge meshes.
// Definitely much better ways to do this, but this is certainly a way.
template <typename T>
HRESULT internal::ComputeCullData(
	XMFLOAT3* positions, uint32_t vertexCount,
	const Meshlet* meshlets, uint32_t meshletCount,
	const T* uniqueVertexIndices,
	const PackedTriangle* primitiveIndices,
	DWORD flags,
	CullData* cullData
) {
	UNREFERENCED_PARAMETER(vertexCount);
	std::cout << "Generating Culling Data For Meshlet Count: " << meshletCount << std::endl;
	unsigned int nThreads = std::thread::hardware_concurrency();
	std::cout << "Using Thread Count: " << nThreads << std::endl;
	// Jobs are launched in batches, slow, but simple and doesn't need to be debugged.
	for (uint32_t i = 0; i < (uint32_t)std::ceil(meshletCount / (float)nThreads); ++i) {
		uint32_t mi = i * nThreads;
		if (i % 100 == 0) {
			std::cout << "On Meshlet: " << mi << std::endl;
		}
		std::vector<std::thread> threadGroup;
		for (uint32_t threadIndex = 0; (threadIndex < nThreads) && ((threadIndex + mi) < meshletCount); threadIndex++) {
			threadGroup.push_back(std::thread(ComputeCullDataForIndex<T>, mi + threadIndex, meshlets, cullData, uniqueVertexIndices, std::ref(vertexCount), positions, primitiveIndices, std::ref(flags)));
		}
		for (auto& thread : threadGroup) {
			thread.join();
		}
	}

	return S_OK;
}
