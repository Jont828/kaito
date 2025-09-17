// Copyright (c) KAITO authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package utils

import (
	"fmt"
	"sort"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
)

const (
	// NVIDIA GPU labels for extracting GPU information
	NvidiaGPUCountLabel  = "nvidia.com/gpu.count"
	NvidiaGPUMemoryLabel = "nvidia.com/gpu.memory"
)

// GPUNodeType represents a group of nodes with identical GPU features
type GPUNodeType struct {
	// GPUProduct is the GPU model/product name.
	GPUProduct string
	// GPUCount is the number of GPUs per node.
	GPUCount int64
	// GPUMemory is the memory per GPU.
	GPUMemory resource.Quantity
	// Nodes is the list of node pointers in this bucket
	Nodes []*corev1.Node
}

// ExtractGPUFeaturesFromNode extracts GPU features from a single node using nvidia.com labels
func ExtractGPUFeaturesFromNode(node *corev1.Node) (*GPUNodeType, error) {
	var gpuProduct string
	var gpuCount int64
	var gpuMemoryMB int64

	// Extract GPU product
	if product, exists := node.Labels["nvidia.com/gpu.product"]; exists {
		gpuProduct = product
	}

	// Extract GPU count
	if countStr, exists := node.Labels[NvidiaGPUCountLabel]; exists {
		count, err := strconv.ParseInt(countStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid GPU count label: %v", err)
		}
		gpuCount = count
	} else {
		return nil, fmt.Errorf("missing nvidia.com/gpu.count label")
	}

	// Extract GPU memory per GPU (in MB)
	if memStr, exists := node.Labels[NvidiaGPUMemoryLabel]; exists {
		memMB, err := strconv.ParseInt(memStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid GPU memory label: %v", err)
		}
		gpuMemoryMB = memMB
	} else {
		return nil, fmt.Errorf("missing nvidia.com/gpu.memory label")
	}

	// Convert memory from MB to resource.Quantity (bytes)
	gpuMemoryBytes := gpuMemoryMB * 1024 * 1024
	gpuMemory := resource.NewQuantity(gpuMemoryBytes, resource.BinarySI)

	return &GPUNodeType{
		GPUProduct: gpuProduct,
		GPUCount:   gpuCount,
		GPUMemory:  *gpuMemory,
		Nodes:      []*corev1.Node{node},
	}, nil
}

// BucketGPUNodes groups nodes by their GPU features and excludes nodes with workspace labels.
// It takes a pre-filtered list of nodes and returns only the GPU node types.
func BucketGPUNodes(nodes []*corev1.Node) ([]GPUNodeType, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no nodes provided")
	}

	// Group nodes by GPU features
	bucketMap := make(map[string]*GPUNodeType)

	for _, node := range nodes {
		// Skip nodes that are already assigned to a workspace for validation purposes
		if _, exists := node.Labels["workspace.kaito.io/name"]; exists {
			continue
		}

		gpuBucket, err := ExtractGPUFeaturesFromNode(node)
		if err != nil {
			klog.Warningf("Failed to extract GPU features from node %s: %v", node.Name, err)
			continue
		}

		// Create a unique key for bucketing based on GPU features
		key := fmt.Sprintf("%s-%d-%s", gpuBucket.GPUProduct, gpuBucket.GPUCount, gpuBucket.GPUMemory.String())

		if bucket, exists := bucketMap[key]; exists {
			bucket.Nodes = append(bucket.Nodes, node)
		} else {
			bucketMap[key] = gpuBucket
		}
	}

	// Convert map to slice and sort by total memory (descending)
	var buckets []GPUNodeType
	for _, bucket := range bucketMap {
		buckets = append(buckets, *bucket)
	}

	// Sort buckets by total memory capacity (descending): memory per GPU * GPUs per node * node count
	sort.Slice(buckets, func(i, j int) bool {
		totalMemoryI := buckets[i].GPUMemory.Value() * buckets[i].GPUCount * int64(len(buckets[i].Nodes))
		totalMemoryJ := buckets[j].GPUMemory.Value() * buckets[j].GPUCount * int64(len(buckets[j].Nodes))
		return totalMemoryI > totalMemoryJ // Descending order
	})

	return buckets, nil
}
